import assert from "node:assert/strict";
import { mkdtemp, readFile, readdir } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";
import test from "node:test";

const collectFiles = async (directory: string): Promise<string[]> => {
  const entries = await readdir(directory, { withFileTypes: true });
  const nested = await Promise.all(
    entries.map((entry) => {
      const absolute = path.join(directory, entry.name);
      return entry.isDirectory() ? collectFiles(absolute) : [absolute];
    }),
  );
  return nested.flat();
};

test("public build contains no local control agent or private connection material", async () => {
  const output = await mkdtemp(path.join(tmpdir(), "qwen-public-build-"));
  const result = spawnSync(
    process.execPath,
    ["tools/build-public.mjs", "--out-dir", output],
    { cwd: process.cwd(), encoding: "utf8" },
  );
  assert.equal(result.status, 0, `${result.stdout}\n${result.stderr}`);

  const files = await collectFiles(output);
  const relativeFiles = files
    .map((file) => path.relative(output, file))
    .sort();
  assert.deepEqual(relativeFiles, [
    "allocation-ledger.d.ts",
    "allocation-ledger.js",
    "browser.d.ts",
    "browser.js",
    "byte-level.d.ts",
    "byte-level.js",
    "device-profile.d.ts",
    "device-profile.js",
    "diagnostics.d.ts",
    "diagnostics.js",
    "gguf.d.ts",
    "gguf.js",
    "gpu-arena.d.ts",
    "gpu-arena.js",
    "http-range-reader.d.ts",
    "http-range-reader.js",
    "hybrid-state.d.ts",
    "hybrid-state.js",
    "incremental-sha256.d.ts",
    "incremental-sha256.js",
    "kernel-registry.d.ts",
    "kernel-registry.js",
    "manifest.d.ts",
    "manifest.js",
    "mixed-gemv.d.ts",
    "mixed-gemv.js",
    "mixed-quant.d.ts",
    "mixed-quant.js",
    "opfs-model-cache.d.ts",
    "opfs-model-cache.js",
    "origin-model-lock.d.ts",
    "origin-model-lock.js",
    "q3k.d.ts",
    "q3k.js",
    "qwen-chat-template.d.ts",
    "qwen-chat-template.js",
    "qwen-tokenizer.d.ts",
    "qwen-tokenizer.js",
    "qwen35-config.d.ts",
    "qwen35-config.js",
    "qwen35-model-loader.d.ts",
    "qwen35-model-loader.js",
    "qwen35-program.d.ts",
    "qwen35-program.js",
    "qwen35-session.d.ts",
    "qwen35-session.js",
    "tensor-policy.d.ts",
    "tensor-policy.js",
    "tokenizer-binary.d.ts",
    "tokenizer-binary.js",
  ]);
  const content = (
    await Promise.all(files.map(async (file) => `${file}\n${await readFile(file, "utf8")}`))
  ).join("\n");
  for (const forbidden of [
    "node:",
    "tokenizer-compiler",
    "compileTokenizerSource",
    "dev/control",
    "qwen-control.v1",
    "LOCAL_CONTROL",
    "operatorToken",
    "phoneToken",
    "127.0.0.1",
    "localhost",
    "/v1/command",
    "/.local-agent.js",
  ]) {
    assert.doesNotMatch(content, new RegExp(forbidden.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")));
  }
});
