import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdir, mkdtemp, readFile, readdir, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import { GgmlType, GgufMetadataType, parseGguf } from "../src/gguf.js";
import {
  PINNED_QWEN35_VISION_CONVERTER_SOURCE,
  convertQwen35VisionPackage,
  orderQwen35VisionLayerGroups,
  parseQwen35VisionConverterArguments,
  type Qwen35VisionSourceContract,
} from "../tools/convert-qwen35-vision.js";
import { BinaryWriter, memoryReader } from "./fixture-utils.js";

function ggufFixture(): Uint8Array {
  const writer = new BinaryWriter()
    .bytesFrom([0x47, 0x47, 0x55, 0x46])
    .u32(3)
    .u64(4)
    .u64(14)
    .string("general.alignment")
    .u32(GgufMetadataType.Uint32)
    .u32(32)
    .string("general.architecture")
    .u32(GgufMetadataType.String)
    .string("clip")
    .string("general.type")
    .u32(GgufMetadataType.String)
    .string("mmproj")
    .string("clip.vision.block_count")
    .u32(GgufMetadataType.Uint32)
    .u32(2)
    .string("clip.vision.embedding_length")
    .u32(GgufMetadataType.Uint32)
    .u32(4)
    .string("clip.vision.feed_forward_length")
    .u32(GgufMetadataType.Uint32)
    .u32(16)
    .string("clip.vision.attention.head_count")
    .u32(GgufMetadataType.Uint32)
    .u32(2)
    .string("clip.vision.projection_dim")
    .u32(GgufMetadataType.Uint32)
    .u32(8)
    .string("clip.vision.image_size")
    .u32(GgufMetadataType.Uint32)
    .u32(16)
    .string("clip.vision.patch_size")
    .u32(GgufMetadataType.Uint32)
    .u32(2)
    .string("clip.vision.spatial_merge_size")
    .u32(GgufMetadataType.Uint32)
    .u32(1)
    .string("clip.projector_type")
    .u32(GgufMetadataType.String)
    .string("fixture_merger")
    .string("clip.use_gelu")
    .u32(GgufMetadataType.Bool)
    .u8(1)
    .string("clip.vision.attention.layer_norm_epsilon")
    .u32(GgufMetadataType.Float32)
    .f32(0.0000009999999974752427)
    .string("v.patch_embd.weight")
    .u32(1)
    .u64(4)
    .u32(GgmlType.BF16)
    .u64(0)
    .string("v.blk.0.attn_q.weight")
    .u32(1)
    .u64(4)
    .u32(GgmlType.BF16)
    .u64(32)
    .string("v.blk.1.attn_q.weight")
    .u32(1)
    .u64(4)
    .u32(GgmlType.BF16)
    .u64(64)
    .string("mm.0.weight")
    .u32(1)
    .u64(2)
    .u32(GgmlType.F32)
    .u64(96);
  writer.pad(32).bytesFrom(new Uint8Array(104));
  return writer.build();
}

async function fixtureContract(bytes: Uint8Array): Promise<Qwen35VisionSourceContract> {
  const parsed = await parseGguf(memoryReader(bytes));
  return {
    source: {
      repository: "https://example.invalid/vision-fixture",
      revision: "1".repeat(40),
      file: "fixture.gguf",
      size: String(bytes.byteLength),
      sha256: createHash("sha256").update(bytes).digest("hex"),
    },
    tokenizer: {
      repository: "https://example.invalid/vision-fixture",
      revision: "2".repeat(40),
      file: "tokenizer.json",
      size: "1",
      sha256: "3".repeat(64),
    },
    processor: {
      repository: "https://example.invalid/vision-fixture",
      revision: "2".repeat(40),
      file: "preprocessor_config.json",
      size: "1",
      sha256: "4".repeat(64),
    },
    processorSettings: {
      processorClass: "Qwen3VLProcessor",
      imageProcessorType: "Qwen2VLImageProcessorFast",
      patchSize: 16,
      temporalPatchSize: 2,
      mergeSize: 2,
      shortestEdge: 65_536,
      longestEdge: 16_777_216,
      imageMean: [0.5, 0.5, 0.5],
      imageStd: [0.5, 0.5, 0.5],
    },
    dataOffset: parsed.dataOffset,
    alignment: 32,
    tensorCount: 4,
    layerCount: 2,
    tensorTypeCounts: Object.freeze({ F32: 1, BF16: 3 }),
    ggufSettings: {
      embeddingLength: 4,
      feedForwardLength: 16,
      headCount: 2,
      projectionDim: 8,
      imageSize: 16,
      patchSize: 2,
      spatialMergeSize: 1,
      projectorType: "fixture_merger",
      useGelu: true,
      attentionLayerNormEpsilon: 0.0000009999999974752427,
    },
  };
}

test("parses the vision converter's explicit source and output paths", () => {
  assert.deepEqual(
    parseQwen35VisionConverterArguments([
      "--source", "vision.gguf", "--output", "vision-package", "--inventory",
    ]),
    { sourcePath: "vision.gguf", outputDirectory: "vision-package", mode: "inventory" },
  );
  assert.throws(
    () => parseQwen35VisionConverterArguments(["--source", "vision.gguf"]),
    /requires --source and --output/u,
  );
});

test("locks the vision source and processor identities to model-sources.json", async () => {
  const descriptor = JSON.parse(await readFile(
    new URL("../model-sources.json", import.meta.url), "utf8",
  ));
  assert.deepEqual(PINNED_QWEN35_VISION_CONVERTER_SOURCE, {
    source: {
      repository: descriptor.vision.repository,
      revision: descriptor.vision.revision,
      file: descriptor.vision.file,
      size: String(descriptor.vision.size),
      sha256: descriptor.vision.sha256,
    },
    tokenizer: {
      repository: descriptor.tokenizer.repository,
      revision: descriptor.tokenizer.revision,
      file: descriptor.tokenizer.file,
      size: String(descriptor.tokenizer.size),
      sha256: descriptor.tokenizer.sha256,
    },
    processor: {
      repository: descriptor.vision.processor.repository,
      revision: descriptor.vision.processor.revision,
      file: descriptor.vision.processor.file,
      size: String(descriptor.vision.processor.size),
      sha256: descriptor.vision.processor.sha256,
    },
    processorSettings: descriptor.vision.processorSettings,
    dataOffset: BigInt(descriptor.vision.dataOffset),
    alignment: descriptor.vision.alignment,
    tensorCount: descriptor.vision.tensorCount,
    layerCount: descriptor.vision.layerCount,
    tensorTypeCounts: descriptor.vision.tensorTypeCounts,
    ggufSettings: descriptor.vision.ggufSettings,
  });
});

test("orders numbered vision layers numerically after the bootstrap group", () => {
  assert.deepEqual(orderQwen35VisionLayerGroups([
    "v.blk.10.attn_q.weight",
    "v.blk.2.attn_q.weight",
    "mm.0.weight",
    "v.blk.0.attn_q.weight",
  ]), ["bootstrap", "0", "2", "10"]);
});

test("authenticates inventory and dry-run without creating vision output", async () => {
  const bytes = ggufFixture();
  const contract = await fixtureContract(bytes);
  const parent = await mkdtemp(join(tmpdir(), "qwen35-vision-modes-"));
  const sourcePath = join(parent, "fixture.gguf");
  await writeFile(sourcePath, bytes);

  const inventory = await convertQwen35VisionPackage({
    sourcePath,
    outputDirectory: join(parent, "inventory-package"),
    mode: "inventory",
  }, contract);
  const dryRun = await convertQwen35VisionPackage({
    sourcePath,
    outputDirectory: join(parent, "dry-run-package"),
    mode: "dry-run",
  }, contract);

  assert.equal(inventory.plan, undefined);
  assert.deepEqual(inventory.layerShards, [
    { layer: "bootstrap", shards: [0] },
    { layer: "0", shards: [1] },
    { layer: "1", shards: [2] },
  ]);
  assert.equal(dryRun.plan?.shardCount, 3);
  assert.equal(dryRun.published, undefined);
  assert.deepEqual(await readdir(parent), ["fixture.gguf"]);
});

test("rejects a same-size source mutation after planning before dry-run", async () => {
  const bytes = ggufFixture();
  const contract = await fixtureContract(bytes);
  const parent = await mkdtemp(join(tmpdir(), "qwen35-vision-mutated-source-"));
  const sourcePath = join(parent, "fixture.gguf");
  await writeFile(sourcePath, bytes);

  await assert.rejects(convertQwen35VisionPackage({
    sourcePath,
    outputDirectory: join(parent, "package"),
    mode: "dry-run",
  }, contract, {
    afterPlanning: async () => {
      const changed = await readFile(sourcePath);
      changed[changed.byteLength - 1]! ^= 0xff;
      await writeFile(sourcePath, changed);
    },
  }), {
    code: "vision-converter-source-changed",
  });
  assert.deepEqual(await readdir(parent), ["fixture.gguf"]);
});

test("rejects a changed structural GGUF contract before creating output", async () => {
  const bytes = ggufFixture();
  const contract = await fixtureContract(bytes);
  const parent = await mkdtemp(join(tmpdir(), "qwen35-vision-contract-"));
  const sourcePath = join(parent, "fixture.gguf");
  await writeFile(sourcePath, bytes);

  await assert.rejects(convertQwen35VisionPackage({
    sourcePath,
    outputDirectory: join(parent, "package"),
    mode: "dry-run",
  }, {
    ...contract,
    ggufSettings: { ...contract.ggufSettings, projectionDim: 9 },
  }), {
    code: "vision-converter-source-contract-mismatch",
  });
  assert.deepEqual(await readdir(parent), ["fixture.gguf"]);
});

test("writes a vision manifest with independently streamable layer shards", async () => {
  const bytes = ggufFixture();
  const contract = await fixtureContract(bytes);
  const parent = await mkdtemp(join(tmpdir(), "qwen35-vision-converter-"));
  const sourcePath = join(parent, "fixture.gguf");
  const outputDirectory = join(parent, "package");
  await writeFile(sourcePath, bytes);

  const report = await convertQwen35VisionPackage({
    sourcePath,
    outputDirectory,
    mode: "convert",
  }, contract, {
    maxShardBytes: 32n,
    tensorAlignment: 16,
    maxBlocksPerRead: 1,
    afterOutputReservation: async (reservedOutput) => {
      assert.deepEqual(await readdir(reservedOutput), []);
      await assert.rejects(mkdir(reservedOutput), { code: "EEXIST" });
    },
  });

  assert.deepEqual(report.inventory.map(({ ggmlType, tensorCount }) => ({ ggmlType, tensorCount })), [
    { ggmlType: "F32", tensorCount: 1 },
    { ggmlType: "BF16", tensorCount: 3 },
  ]);
  assert.deepEqual(report.layerShards, [
    { layer: "bootstrap", shards: [0] },
    { layer: "0", shards: [1] },
    { layer: "1", shards: [2] },
  ]);
  const manifest = JSON.parse(await readFile(join(outputDirectory, "manifest.json"), "utf8"));
  assert.equal(manifest.packageKind, "vision");
  assert.deepEqual(manifest.processor, contract.processor);
  assert.deepEqual(manifest.processorSettings, contract.processorSettings);
  assert.equal(manifest.runtime.abi, "qwen35-webgpu-vision-v1");
  assert.equal(manifest.tensorLayout.every((entry: { shardOffset: string }) =>
    BigInt(entry.shardOffset) % 16n === 0n,
  ), true);
  const layerIndex = JSON.parse(await readFile(join(outputDirectory, "layer-index.json"), "utf8"));
  assert.deepEqual(layerIndex.groups, report.layerShards);
  assert.equal(new Set(layerIndex.groups.flatMap((group: { shards: number[] }) => group.shards)).size, 3);
  const shardFiles = (await readdir(join(outputDirectory, "shards"))).map(
    (file) => `shards/${file}`,
  );
  const expectedChecksummedFiles = [
    "LICENSES.json",
    "layer-index.json",
    "manifest.json",
    ...shardFiles,
    "source-provenance.json",
  ].sort();
  const checksums = await readFile(join(outputDirectory, "SHA256SUMS"), "utf8");
  const checksumEntries = checksums.trim().split("\n").map((line) => {
    const match = /^([a-f0-9]{64})  (.+)$/u.exec(line);
    assert.notEqual(match, null);
    return { sha256: match![1]!, file: match![2]! };
  });
  assert.deepEqual(checksumEntries.map((entry) => entry.file), expectedChecksummedFiles);
  for (const entry of checksumEntries) {
    const actual = createHash("sha256")
      .update(await readFile(join(outputDirectory, entry.file)))
      .digest("hex");
    assert.equal(actual, entry.sha256);
  }
  assert.deepEqual((await readdir(outputDirectory)).sort(), [
    "LICENSES.json",
    "SHA256SUMS",
    "layer-index.json",
    "manifest.json",
    "shards",
    "source-provenance.json",
  ]);
  assert.deepEqual(await readdir(parent).then((entries) => entries.sort()), ["fixture.gguf", "package"]);
  const provenance = await readFile(join(outputDirectory, "source-provenance.json"), "utf8");
  assert.equal(provenance.includes(sourcePath), false);
  assert.match(provenance, /"imageProcessorType": "Qwen2VLImageProcessorFast"/u);
});

test("fails safely when output already exists and leaves no model material in the repository", async () => {
  const bytes = ggufFixture();
  const contract = await fixtureContract(bytes);
  const parent = await mkdtemp(join(tmpdir(), "qwen35-vision-existing-"));
  const sourcePath = join(parent, "fixture.gguf");
  const outputDirectory = join(parent, "package");
  await writeFile(sourcePath, bytes);
  await writeFile(outputDirectory, "preserve");

  await assert.rejects(convertQwen35VisionPackage({
    sourcePath,
    outputDirectory,
    mode: "convert",
  }, contract), {
    code: "vision-converter-output-exists",
  });
  assert.equal(await readFile(outputDirectory, "utf8"), "preserve");
});
