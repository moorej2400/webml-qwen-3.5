import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdtemp, readFile, readdir, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import { GgmlType, GgufMetadataType, parseGguf } from "../src/gguf.js";
import {
  PINNED_QWEN35_LANGUAGE_CONVERTER_SOURCE,
  convertQwen35LanguagePackage,
  parseQwen35LanguageConverterArguments,
  type Qwen35LanguageSourceContract,
} from "../tools/convert-qwen35-language.js";
import { BinaryWriter, memoryReader } from "./fixture-utils.js";

function ggufFixture(): Uint8Array {
  const writer = new BinaryWriter()
    .bytesFrom([0x47, 0x47, 0x55, 0x46])
    .u32(3)
    .u64(3)
    .u64(3)
    .string("general.alignment")
    .u32(GgufMetadataType.Uint32)
    .u32(32)
    .string("general.architecture")
    .u32(GgufMetadataType.String)
    .string("qwen35")
    .string("qwen35.block_count")
    .u32(GgufMetadataType.Uint32)
    .u32(33)
    .string("output_norm.weight")
    .u32(1)
    .u64(4)
    .u32(GgmlType.F32)
    .u64(0)
    .string("blk.0.attn_q.weight")
    .u32(1)
    .u64(256)
    .u32(GgmlType.Q3_K)
    .u64(32)
    .string("blk.32.nextn.weight")
    .u32(1)
    .u64(32)
    .u32(GgmlType.Q8_0)
    .u64(160);
  writer.pad(32).bytesFrom(new Uint8Array(194));
  return writer.build();
}

async function fixtureContract(bytes: Uint8Array): Promise<Qwen35LanguageSourceContract> {
  const parsed = await parseGguf(memoryReader(bytes));
  return {
    source: {
      repository: "https://example.invalid/qwen-fixture",
      revision: "1".repeat(40),
      file: "fixture.gguf",
      size: String(bytes.byteLength),
      sha256: createHash("sha256").update(bytes).digest("hex"),
    },
    tokenizer: {
      repository: "https://example.invalid/qwen-fixture",
      revision: "2".repeat(40),
      file: "tokenizer.json",
      size: "1",
      sha256: "3".repeat(64),
    },
    dataOffset: parsed.dataOffset,
    alignment: 32,
    blockCount: 33,
    tensorTypeCounts: Object.freeze({ F32: 1, Q8_0: 1, Q3_K: 1 }),
  };
}

test("parses only explicit source and output paths with one optional mode", () => {
  assert.deepEqual(
    parseQwen35LanguageConverterArguments([
      "--source",
      "artifacts/model.gguf",
      "--output",
      "artifacts/package",
      "--dry-run",
    ]),
    {
      sourcePath: "artifacts/model.gguf",
      outputDirectory: "artifacts/package",
      mode: "dry-run",
    },
  );
  assert.throws(
    () => parseQwen35LanguageConverterArguments(["--source", "model.gguf"]),
    /requires --source and --output/u,
  );
  assert.throws(
    () => parseQwen35LanguageConverterArguments([
      "--source", "model.gguf", "--output", "package", "--inventory", "--dry-run",
    ]),
    /one mode/u,
  );
  assert.throws(
    () => parseQwen35LanguageConverterArguments([
      "--source", "model.gguf", "--output", "package", "--token", "private",
    ]),
    /unknown argument/u,
  );
});

test("locks the executable converter to the machine-readable source pins", async () => {
  const descriptor = JSON.parse(await readFile(
    new URL("../model-sources.json", import.meta.url),
    "utf8",
  ));
  assert.deepEqual(PINNED_QWEN35_LANGUAGE_CONVERTER_SOURCE, {
    source: {
      repository: descriptor.language.repository,
      revision: descriptor.language.revision,
      file: descriptor.language.file,
      size: String(descriptor.language.size),
      sha256: descriptor.language.sha256,
    },
    tokenizer: {
      repository: descriptor.tokenizer.repository,
      revision: descriptor.tokenizer.revision,
      file: descriptor.tokenizer.file,
      size: String(descriptor.tokenizer.size),
      sha256: descriptor.tokenizer.sha256,
    },
    dataOffset: BigInt(descriptor.language.dataOffset),
    alignment: descriptor.language.alignment,
    blockCount: descriptor.language.blockCount,
    tensorTypeCounts: descriptor.language.tensorTypeCounts,
  });
});

test("authenticates inventory and dry-run without creating output", async () => {
  const bytes = ggufFixture();
  const contract = await fixtureContract(bytes);
  const parent = await mkdtemp(join(tmpdir(), "qwen35-converter-modes-"));
  const sourcePath = join(parent, "fixture.gguf");
  await writeFile(sourcePath, bytes);

  const inventory = await convertQwen35LanguagePackage({
    sourcePath,
    outputDirectory: join(parent, "inventory-output"),
    mode: "inventory",
  }, contract);
  const dryRun = await convertQwen35LanguagePackage({
    sourcePath,
    outputDirectory: join(parent, "dry-run-output"),
    mode: "dry-run",
  }, contract);

  assert.deepEqual(inventory.inventory.map(({ ggmlType, tensorCount }) => ({
    ggmlType,
    tensorCount,
  })), [
    { ggmlType: "F32", tensorCount: 1 },
    { ggmlType: "Q8_0", tensorCount: 1 },
    { ggmlType: "Q3_K", tensorCount: 1 },
  ]);
  assert.deepEqual(inventory.excludedTensors, [{
    name: "blk.32.nextn.weight",
    reason: "excluded-by-mtp-name-policy-v1",
  }]);
  assert.equal(inventory.plan, undefined);
  assert.equal(dryRun.plan?.shardCount, 1);
  assert.equal(dryRun.published, undefined);
  assert.deepEqual(await readdir(parent), ["fixture.gguf"]);
});

test("publishes shards and deterministic metadata by one atomic directory rename", async () => {
  const bytes = ggufFixture();
  const contract = await fixtureContract(bytes);
  const parent = await mkdtemp(join(tmpdir(), "qwen35-converter-publish-"));
  const sourcePath = join(parent, "fixture.gguf");
  const outputDirectory = join(parent, "package");
  await writeFile(sourcePath, bytes);

  const report = await convertQwen35LanguagePackage({
    sourcePath,
    outputDirectory,
    mode: "convert",
  }, contract, {
    maxShardBytes: 512n,
    tensorAlignment: 16,
    maxBlocksPerRead: 1,
  });

  assert.deepEqual(report.published?.files, [
    "LICENSES.json",
    "SHA256SUMS",
    "manifest.json",
    "shards/model-00000.bin",
    "source-provenance.json",
  ]);
  assert.deepEqual((await readdir(parent)).sort(), ["fixture.gguf", "package"]);
  const outputFiles = await readdir(outputDirectory);
  assert.deepEqual(outputFiles.sort(), [
    "LICENSES.json",
    "SHA256SUMS",
    "manifest.json",
    "shards",
    "source-provenance.json",
  ]);
  const manifest = JSON.parse(await readFile(join(outputDirectory, "manifest.json"), "utf8"));
  assert.equal(manifest.format, "webml-qwen-package");
  assert.equal(manifest.source.sha256, contract.source.sha256);
  assert.deepEqual(manifest.excludedTensors, [{
    name: "blk.32.nextn.weight",
    reason: "excluded-by-mtp-name-policy-v1",
  }]);
  const checksums = await readFile(join(outputDirectory, "SHA256SUMS"), "utf8");
  assert.match(checksums, /^[a-f0-9]{64}  LICENSES\.json$/mu);
  assert.match(checksums, /^[a-f0-9]{64}  shards\/model-00000\.bin$/mu);
  assert.equal(checksums.includes(sourcePath), false);
  const provenance = await readFile(join(outputDirectory, "source-provenance.json"), "utf8");
  assert.equal(provenance.includes(sourcePath), false);
  assert.match(provenance, /"runtimeAbi": "qwen35-webgpu-v1"/u);
  const licenses = await readFile(join(outputDirectory, "LICENSES.json"), "utf8");
  assert.match(licenses, /"spdx": "Apache-2.0"/u);
});

test("rejects Git-contained output and sanitizes source authentication failures", async () => {
  const bytes = ggufFixture();
  const contract = await fixtureContract(bytes);
  const parent = await mkdtemp(join(tmpdir(), "qwen35-converter-private-marker-"));
  const sourcePath = join(parent, "private-marker-source.gguf");
  await writeFile(sourcePath, bytes);
  await writeFile(join(parent, ".git"), "gitdir: elsewhere\n");

  await assert.rejects(convertQwen35LanguagePackage({
    sourcePath,
    outputDirectory: join(parent, "package"),
    mode: "dry-run",
  }, contract), {
    code: "converter-output-inside-git",
    message: "Converter output must be outside a Git worktree",
  });

  const outsideParent = await mkdtemp(join(tmpdir(), "qwen35-converter-bad-source-"));
  const badSourcePath = join(outsideParent, "private-marker-source.gguf");
  await writeFile(badSourcePath, bytes.subarray(0, bytes.byteLength - 1));
  await assert.rejects(convertQwen35LanguagePackage({
    sourcePath: badSourcePath,
    outputDirectory: join(outsideParent, "package"),
    mode: "convert",
  }, contract), (error: unknown) => {
    assert.equal((error as { code?: string }).code, "converter-source-size-mismatch");
    assert.equal(String(error).includes("private-marker"), false);
    return true;
  });
  assert.deepEqual(await readdir(outsideParent), ["private-marker-source.gguf"]);
});

test("rejects a same-size hash mismatch and never overwrites output", async () => {
  const bytes = ggufFixture();
  const contract = await fixtureContract(bytes);
  const parent = await mkdtemp(join(tmpdir(), "qwen35-converter-integrity-"));
  const sourcePath = join(parent, "fixture.gguf");
  const changed = bytes.slice();
  changed[changed.length - 1] ^= 0xff;
  await writeFile(sourcePath, changed);

  await assert.rejects(convertQwen35LanguagePackage({
    sourcePath,
    outputDirectory: join(parent, "package"),
    mode: "convert",
  }, contract), {
    code: "converter-source-hash-mismatch",
    message: "Converter source SHA-256 does not match the pinned identity",
  });

  const existingOutput = await mkdtemp(join(parent, "existing-package-"));
  await writeFile(join(existingOutput, "sentinel.txt"), "preserve\n");
  await assert.rejects(convertQwen35LanguagePackage({
    sourcePath,
    outputDirectory: existingOutput,
    mode: "convert",
  }, contract), {
    code: "converter-output-exists",
    message: "Converter output directory already exists",
  });
  assert.equal(await readFile(join(existingOutput, "sentinel.txt"), "utf8"), "preserve\n");
});

test("publishes one repository script and a public-safe operator guide", async () => {
  const packageJson = JSON.parse(await readFile(
    new URL("../package.json", import.meta.url),
    "utf8",
  ));
  const readme = await readFile(new URL("../README.md", import.meta.url), "utf8");
  const guide = await readFile(
    new URL("../docs/model-package-conversion.md", import.meta.url),
    "utf8",
  );

  assert.equal(
    packageJson.scripts["convert:qwen35-language"],
    "node --import tsx tools/convert-qwen35-language.ts",
  );
  assert.match(readme, /model package conversion/iu);
  assert.match(guide, /--source <gguf-path> --output <package-path>/u);
  assert.match(guide, /--inventory/u);
  assert.match(guide, /--dry-run/u);
  assert.match(guide, /outside a Git worktree/u);
  assert.match(guide, /retained.*inspection/iu);
  assert.equal(guide.includes("/Users/"), false);
});
