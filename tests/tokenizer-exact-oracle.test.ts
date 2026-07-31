import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { renderQwen35Chat, type Qwen35ChatMessage } from "../src/qwen-chat-template.js";
import {
  Qwen35Tokenizer,
  type UnsafeTokenizerReferenceFixture,
} from "../src/qwen-tokenizer.js";

test(
  "always matches the pinned official Transformers oracle",
  async () => {
    const oracleBytes = await readFile(
      new URL("./fixtures/qwen35-tokenizer-oracle.json", import.meta.url),
    );
    const sliceBytes = await readFile(
      new URL(
        "./fixtures/qwen35-tokenizer-oracle-slice.json",
        import.meta.url,
      ),
    );
    assert.equal(
      createHash("sha256").update(sliceBytes).digest("hex"),
      "76272646681825d15accd7d406fdddf7cd229c3b44abeec01919d153d496f7ee",
    );
    const fixture = JSON.parse(oracleBytes.toString("utf8")) as {
      source: string;
      revision: string;
      tokenizerSha256: string;
      text: { name: string; input: string; ids: number[] }[];
      chat: {
        name: string;
        messages: Qwen35ChatMessage[];
        addGenerationPrompt: boolean;
        enableThinking: boolean;
        rendered: string;
        ids: number[];
      }[];
    };
    const slice = JSON.parse(sliceBytes.toString("utf8")) as {
      source: string;
      revision: string;
      tokenizerSha256: string;
      runtimeSlice: UnsafeTokenizerReferenceFixture;
      closureProof: {
        version: number;
        algorithm: string;
        sourceMergeCount: number;
        oracleCaseCount: number;
        pieceCount: number;
        stateCount: number;
        adjacentPairObservations: number;
        mergeCandidateObservations: number;
        losingCandidateObservations: number;
        staleCandidateOpportunities: number;
        uniqueCandidateMergeCount: number;
        traceSha256: string;
      };
      closureTraces: readonly [
        "text" | "chat",
        string,
        number,
        readonly number[],
        readonly (readonly (readonly number[])[])[],
        readonly number[],
      ][];
    };
    const provenance = {
      source: "Qwen/Qwen3.5-4B",
      revision: "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
      tokenizerSha256:
        "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
    };
    assert.deepEqual(
      {
        source: fixture.source,
        revision: fixture.revision,
        tokenizerSha256: fixture.tokenizerSha256,
      },
      provenance,
    );
    assert.deepEqual(
      {
        source: slice.source,
        revision: slice.revision,
        tokenizerSha256: slice.tokenizerSha256,
      },
      provenance,
    );
    const tokenizer =
      Qwen35Tokenizer.fromUnsafeReferenceFixtureForTests(
        slice.runtimeSlice,
      );

    assert.deepEqual(
      {
        version: slice.closureProof.version,
        algorithm: slice.closureProof.algorithm,
        sourceMergeCount: slice.closureProof.sourceMergeCount,
        oracleCaseCount: slice.closureProof.oracleCaseCount,
        pieceCount: slice.closureProof.pieceCount,
        uniqueCandidateMergeCount:
          slice.closureProof.uniqueCandidateMergeCount,
      },
      {
        version: 1,
        algorithm: "all-adjacent-candidates-leftmost-min-rank-v1",
        sourceMergeCount: 247_587,
        oracleCaseCount: fixture.text.length + fixture.chat.length,
        pieceCount: slice.closureTraces.length,
        uniqueCandidateMergeCount: slice.runtimeSlice.merges.length,
      },
    );
    assert.equal(
      createHash("sha256")
        .update(JSON.stringify(slice.closureTraces))
        .digest("hex"),
      slice.closureProof.traceSha256,
    );
    validateClosureProof(slice);
    assert.ok(slice.closureProof.losingCandidateObservations > 0);
    assert.ok(slice.closureProof.staleCandidateOpportunities > 0);
    assert.equal(tokenizer.decodableTokenCount, 248_070);
    assert.equal(tokenizer.undecodableLogitRows, 250);
    for (const fixtureCase of fixture.text) {
      const ids = tokenizer.encode(fixtureCase.input);
      assert.deepEqual(ids, fixtureCase.ids, fixtureCase.name);
      assert.equal(tokenizer.decode(ids), fixtureCase.input.normalize("NFC"));
    }
    for (const fixtureCase of fixture.chat) {
      const rendered = renderQwen35Chat(fixtureCase.messages, {
        addGenerationPrompt: fixtureCase.addGenerationPrompt,
        enableThinking: fixtureCase.enableThinking,
      });
      assert.equal(rendered, fixtureCase.rendered, fixtureCase.name);
      assert.deepEqual(
        tokenizer.encode(rendered, { addedTokens: "allow" }),
        fixtureCase.ids,
        fixtureCase.name,
      );
    }
  },
);

function validateClosureProof(slice: {
  runtimeSlice: UnsafeTokenizerReferenceFixture;
  closureProof: {
    stateCount: number;
    adjacentPairObservations: number;
    mergeCandidateObservations: number;
    losingCandidateObservations: number;
    staleCandidateOpportunities: number;
  };
  closureTraces: readonly [
    "text" | "chat",
    string,
    number,
    readonly number[],
    readonly (readonly (readonly number[])[])[],
    readonly number[],
  ][];
}): void {
  const merges = new Map(
    slice.runtimeSlice.merges.map((merge) => [
      `${merge.left}:${merge.right}`,
      merge,
    ]),
  );
  let stateCount = 0;
  let adjacentPairObservations = 0;
  let mergeCandidateObservations = 0;
  let losingCandidateObservations = 0;
  let staleCandidateOpportunities = 0;

  for (const [, name, pieceIndex, initial, states, expected] of
    slice.closureTraces) {
    const symbols = [...initial];
    for (const recorded of states) {
      stateCount += 1;
      adjacentPairObservations += Math.max(0, symbols.length - 1);
      const discovered: number[][] = [];
      for (let leftIndex = 0; leftIndex + 1 < symbols.length; leftIndex += 1) {
        const left = symbols[leftIndex]!;
        const right = symbols[leftIndex + 1]!;
        const merge = merges.get(`${left}:${right}`);
        if (merge !== undefined) {
          discovered.push([
            leftIndex,
            left,
            right,
            merge.rank,
            merge.result,
          ]);
        }
      }
      assert.deepEqual(
        discovered,
        recorded,
        `${name} piece ${pieceIndex} closure state ${stateCount}`,
      );
      mergeCandidateObservations += discovered.length;
      if (discovered.length === 0) {
        continue;
      }
      losingCandidateObservations += discovered.length - 1;
      const selected = discovered.reduce((best, candidate) =>
        candidate[3]! < best[3]! ||
        (candidate[3] === best[3] && candidate[0]! < best[0]!)
          ? candidate
          : best,
      );
      staleCandidateOpportunities += discovered.filter(
        (candidate) =>
          candidate !== selected &&
          Math.abs(candidate[0]! - selected[0]!) <= 1,
      ).length;
      symbols.splice(selected[0]!, 2, selected[4]!);
    }
    assert.deepEqual(symbols, expected, `${name} piece ${pieceIndex} result`);
  }
  assert.deepEqual(
    {
      stateCount,
      adjacentPairObservations,
      mergeCandidateObservations,
      losingCandidateObservations,
      staleCandidateOpportunities,
    },
    {
      stateCount: slice.closureProof.stateCount,
      adjacentPairObservations: slice.closureProof.adjacentPairObservations,
      mergeCandidateObservations:
        slice.closureProof.mergeCandidateObservations,
      losingCandidateObservations:
        slice.closureProof.losingCandidateObservations,
      staleCandidateOpportunities:
        slice.closureProof.staleCandidateOpportunities,
    },
  );
}
