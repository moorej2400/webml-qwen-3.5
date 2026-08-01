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
    "hybrid-kernels.d.ts",
    "hybrid-kernels.js",
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
    "qwen-embedding.d.ts",
    "qwen-embedding.js",
    "qwen-primitives.d.ts",
    "qwen-primitives.js",
    "qwen-tokenizer.d.ts",
    "qwen-tokenizer.js",
    "qwen35-activation-workspace.d.ts",
    "qwen35-activation-workspace.js",
    "qwen35-allocation-clear.d.ts",
    "qwen35-allocation-clear.js",
    "qwen35-config.d.ts",
    "qwen35-config.js",
    "qwen35-deltanet-dispatch.d.ts",
    "qwen35-deltanet-dispatch.js",
    "qwen35-final-dispatch.d.ts",
    "qwen35-final-dispatch.js",
    "qwen35-forward-dispatch.d.ts",
    "qwen35-forward-dispatch.js",
    "qwen35-full-attention-dispatch.d.ts",
    "qwen35-full-attention-dispatch.js",
    "qwen35-greedy-driver.d.ts",
    "qwen35-greedy-driver.js",
    "qwen35-logits-dispatch.d.ts",
    "qwen35-logits-dispatch.js",
    "qwen35-logits-reduction.d.ts",
    "qwen35-logits-reduction.js",
    "qwen35-model-loader.d.ts",
    "qwen35-model-loader.js",
    "qwen35-program.d.ts",
    "qwen35-program.js",
    "qwen35-session.d.ts",
    "qwen35-session.js",
    "qwen35-uniform-arena.d.ts",
    "qwen35-uniform-arena.js",
    "qwen35-vision-package-bootstrap.d.ts",
    "qwen35-vision-package-bootstrap.js",
    "qwen35-vision-package-loader.d.ts",
    "qwen35-vision-package-loader.js",
    "qwen35-vision-preprocess.d.ts",
    "qwen35-vision-preprocess.js",
    "qwen35-vision-program.d.ts",
    "qwen35-vision-program.js",
    "qwen35-webgpu-executor.d.ts",
    "qwen35-webgpu-executor.js",
    "qwen35-weight-directory.d.ts",
    "qwen35-weight-directory.js",
    "qwen35-weight-upload.d.ts",
    "qwen35-weight-upload.js",
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
    "__QWEN_LOCAL_CONTROL__",
    "local-runtime-config",
    "dev/browser",
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
