import assert from "node:assert/strict";
import test from "node:test";
import { parseTokenizerCompilerArguments } from "../tools/compile-qwen35-tokenizer.js";

test("parses explicit tokenizer compiler paths without environment fallbacks", () => {
  assert.deepEqual(
    parseTokenizerCompilerArguments([
      "--tokenizer",
      "fixtures/tokenizer.json",
      "--config",
      "fixtures/tokenizer_config.json",
      "--output",
      "artifacts/compiled-tokenizer",
    ]),
    {
      tokenizerPath: "fixtures/tokenizer.json",
      configPath: "fixtures/tokenizer_config.json",
      outputDirectory: "artifacts/compiled-tokenizer",
    },
  );
});

test("rejects incomplete or unknown tokenizer compiler arguments", () => {
  assert.throws(
    () =>
      parseTokenizerCompilerArguments([
        "--tokenizer",
        "fixtures/tokenizer.json",
      ]),
    /requires --tokenizer, --config, and --output/u,
  );
  assert.throws(
    () =>
      parseTokenizerCompilerArguments([
        "--tokenizer",
        "fixtures/tokenizer.json",
        "--config",
        "fixtures/tokenizer_config.json",
        "--output",
        "artifacts/compiled-tokenizer",
        "--secret",
        "value",
      ]),
    /unknown argument/u,
  );
});
