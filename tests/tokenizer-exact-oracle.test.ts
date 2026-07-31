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
      "07ceaf9bbbd406b4d015f6ba28843fa7e80cd4f7669f112ae0016c6d558ec089",
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
