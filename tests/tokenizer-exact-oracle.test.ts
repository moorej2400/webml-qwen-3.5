import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { renderQwen35Chat, type Qwen35ChatMessage } from "../src/qwen-chat-template.js";
import { Qwen35Tokenizer } from "../src/qwen-tokenizer.js";

const artifactPath = process.env.QWEN35_TOKENIZER_ARTIFACT;

test(
  "matches official Transformers oracle ids for the exact pinned artifact",
  { skip: artifactPath === undefined },
  async () => {
    const fixture = JSON.parse(
      await readFile(
        new URL("./fixtures/qwen35-tokenizer-oracle.json", import.meta.url),
        "utf8",
      ),
    ) as {
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
    const tokenizer = Qwen35Tokenizer.fromCompiledArtifact(
      await readFile(artifactPath!),
    );

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
