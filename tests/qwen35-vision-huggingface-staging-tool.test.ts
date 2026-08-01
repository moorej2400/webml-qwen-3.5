import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import {
  appendFile,
  lstat,
  mkdir,
  mkdtemp,
  readFile,
  readdir,
  writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import { GgmlType } from "../src/gguf.js";
import {
  stringifyManifest,
  type ImmutableArtifactIdentity,
  type VisionProcessorSettings,
} from "../src/manifest.js";
import {
  parseQwen35VisionHuggingFaceStagingArguments,
  stageQwen35VisionHuggingFacePackage,
  type Qwen35VisionHuggingFaceStageContract,
} from "../tools/stage-qwen35-vision-huggingface.js";

const REVISION = "1".repeat(40);

function sha256(bytes: Uint8Array | string): string {
  return createHash("sha256").update(bytes).digest("hex");
}

function identity(file: string): ImmutableArtifactIdentity {
  return {
    repository: "https://example.invalid/vision-fixture",
    revision: REVISION,
    file,
    size: "1",
    sha256: "2".repeat(64),
  };
}

const processorSettings: VisionProcessorSettings = {
  processorClass: "Qwen3VLProcessor",
  imageProcessorType: "Qwen2VLImageProcessorFast",
  patchSize: 16,
  temporalPatchSize: 2,
  mergeSize: 2,
  shortestEdge: 65_536,
  longestEdge: 16_777_216,
  imageMean: [0.5, 0.5, 0.5],
  imageStd: [0.5, 0.5, 0.5],
};

async function packageFixture(parent: string): Promise<{
  packageDirectory: string;
  contract: Qwen35VisionHuggingFaceStageContract;
}> {
  const packageDirectory = join(parent, "converted-vision-package");
  const shardDirectory = join(packageDirectory, "shards");
  await mkdir(shardDirectory, { recursive: true });
  const source = identity("vision.gguf");
  const tokenizer = identity("tokenizer.json");
  const processor = identity("preprocessor_config.json");
  const shards = await Promise.all(Array.from({ length: 26 }, async (_, index) => {
    const bytes = Uint8Array.of(index, index + 1, index + 2, index + 3);
    const name = `vision-${String(index).padStart(5, "0")}.bin`;
    await writeFile(join(shardDirectory, name), bytes);
    return {
      url: `shards/${name}`,
      offset: String(index * 4),
      length: "4",
      sha256: sha256(bytes),
    };
  }));
  const manifest = stringifyManifest({
    format: "webml-qwen-package" as const,
    version: 1 as const,
    packageKind: "vision" as const,
    source,
    runtime: { abi: "qwen35-webgpu-vision-v1" },
    tokenizer,
    processor,
    processorSettings,
    tensorLayout: shards.map((shard, index) => ({
      name: `v.blk.${index}.weight`,
      shape: ["1"],
      ggmlType: GgmlType.F32,
      storageType: "f32" as const,
      shard: index,
      shardOffset: "0",
      tensorOffset: "0",
      length: "4",
    })),
    shards,
    excludedTensors: [],
  });
  const layerIndex = `${JSON.stringify({
    format: "webml-qwen-vision-layer-index",
    version: 1,
    groups: [
      { layer: "bootstrap", shards: [0, 1] },
      ...Array.from({ length: 24 }, (_, index) => ({
        layer: String(index),
        shards: [index + 2],
      })),
    ],
  }, null, 2)}\n`;
  const licenses = "{\n  \"format\": \"webml-qwen-license-metadata\",\n  \"version\": 1\n}\n";
  const provenance = "{\n  \"format\": \"webml-qwen-vision-source-provenance\",\n  \"version\": 1\n}\n";
  await writeFile(join(packageDirectory, "manifest.json"), manifest);
  await writeFile(join(packageDirectory, "layer-index.json"), layerIndex);
  await writeFile(join(packageDirectory, "LICENSES.json"), licenses);
  await writeFile(join(packageDirectory, "source-provenance.json"), provenance);
  const files = [
    ["LICENSES.json", licenses],
    ["layer-index.json", layerIndex],
    ["manifest.json", manifest],
    ...shards.map((shard) => [shard.url, readFile(join(packageDirectory, shard.url))] as const),
    ["source-provenance.json", provenance],
  ] as const;
  const lines: string[] = [];
  for (const [file, content] of files) {
    lines.push(`${sha256(await content)}  ${file}`);
  }
  await writeFile(join(packageDirectory, "SHA256SUMS"), `${lines.join("\n")}\n`);
  return {
    packageDirectory,
    contract: { source, tokenizer, processor, processorSettings, shardCount: 26 },
  };
}

async function rewriteFixtureChecksums(packageDirectory: string): Promise<void> {
  const shardNames = (await readdir(join(packageDirectory, "shards"))).sort();
  const files = [
    "LICENSES.json",
    "layer-index.json",
    "manifest.json",
    ...shardNames.map((name) => `shards/${name}`),
    "source-provenance.json",
  ];
  const lines: string[] = [];
  for (const file of files) {
    lines.push(`${sha256(await readFile(join(packageDirectory, file)))}  ${file}`);
  }
  await writeFile(join(packageDirectory, "SHA256SUMS"), `${lines.join("\n")}\n`);
}

function layerIndex(groups: readonly unknown[]): string {
  return `${JSON.stringify({
    format: "webml-qwen-vision-layer-index",
    version: 1,
    groups,
  }, null, 2)}\n`;
}

test("accepts only explicit converted-package and output paths", () => {
  assert.deepEqual(parseQwen35VisionHuggingFaceStagingArguments([
    "--package", "converted-vision", "--output", "upload-root",
  ]), { packageDirectory: "converted-vision", outputDirectory: "upload-root" });
  assert.throws(
    () => parseQwen35VisionHuggingFaceStagingArguments(["--package", "converted-vision"]),
    /requires --package and --output/u,
  );
});

test("stages an authenticated vision package as collision-safe flat files", async () => {
  const parent = await mkdtemp(join(tmpdir(), "qwen35-vision-hf-stage-"));
  const fixture = await packageFixture(parent);
  const outputDirectory = join(parent, "upload-root");
  const report = await stageQwen35VisionHuggingFacePackage({
    packageDirectory: fixture.packageDirectory,
    outputDirectory,
  }, fixture.contract);

  const expectedShards = Array.from({ length: 26 }, (_, index) =>
    `vision-${String(index).padStart(5, "0")}.bin`,
  );
  assert.deepEqual(report.files, [
    "vision-LICENSES.json",
    "vision-SHA256SUMS",
    "vision-layer-index.json",
    "vision-manifest.json",
    "vision-publication-provenance.json",
    "vision-source-provenance.json",
    ...expectedShards,
  ].sort());
  assert.deepEqual((await readdir(outputDirectory)).sort(), report.files);
  const staged = JSON.parse(await readFile(join(outputDirectory, "vision-manifest.json"), "utf8"));
  assert.deepEqual(staged.shards.map((shard: { url: string }) => shard.url), expectedShards);
  assert.deepEqual(
    await readFile(join(outputDirectory, "vision-layer-index.json"), "utf8"),
    await readFile(join(fixture.packageDirectory, "layer-index.json"), "utf8"),
  );
  const checksums = await readFile(join(outputDirectory, "vision-SHA256SUMS"), "utf8");
  assert.equal(checksums.trim().split("\n").length, report.files.length - 1);
  assert.equal((await readFile(join(outputDirectory, "vision-publication-provenance.json"), "utf8")).includes(parent), false);
});

test("rejects source checksum drift before creating output and never overwrites", async () => {
  const parent = await mkdtemp(join(tmpdir(), "qwen35-vision-hf-drift-"));
  const fixture = await packageFixture(parent);
  const outputDirectory = join(parent, "upload-root");
  await writeFile(join(fixture.packageDirectory, "shards/vision-00000.bin"), Uint8Array.of(9, 9, 9, 9));
  await assert.rejects(stageQwen35VisionHuggingFacePackage({
    packageDirectory: fixture.packageDirectory,
    outputDirectory,
  }, fixture.contract), { code: "vision-hf-stage-source-checksum-mismatch" });
  await assert.rejects(lstat(outputDirectory), { code: "ENOENT" });

  const secondParent = await mkdtemp(join(tmpdir(), "qwen35-vision-hf-existing-"));
  const second = await packageFixture(secondParent);
  const existing = join(secondParent, "upload-root");
  await mkdir(existing);
  await writeFile(join(existing, "sentinel.txt"), "preserve\n");
  await assert.rejects(stageQwen35VisionHuggingFacePackage({
    packageDirectory: second.packageDirectory,
    outputDirectory: existing,
  }, second.contract), { code: "vision-hf-stage-output-exists" });
  assert.equal(await readFile(join(existing, "sentinel.txt"), "utf8"), "preserve\n");
});

test("accepts metadata at the eight-mebibyte limit and rejects a larger metadata file before output creation", async () => {
  const exactParent = await mkdtemp(join(tmpdir(), "qwen35-vision-hf-metadata-exact-"));
  const exact = await packageFixture(exactParent);
  await writeFile(join(exact.packageDirectory, "LICENSES.json"), Buffer.alloc(8 * 1024 * 1024, 0x20));
  await rewriteFixtureChecksums(exact.packageDirectory);
  const exactOutput = join(exactParent, "upload-root");
  await stageQwen35VisionHuggingFacePackage({
    packageDirectory: exact.packageDirectory,
    outputDirectory: exactOutput,
  }, exact.contract);
  assert.equal((await lstat(join(exactOutput, "vision-LICENSES.json"))).size, 8 * 1024 * 1024);

  const tooLargeParent = await mkdtemp(join(tmpdir(), "qwen35-vision-hf-metadata-large-"));
  const tooLarge = await packageFixture(tooLargeParent);
  await writeFile(join(tooLarge.packageDirectory, "LICENSES.json"), Buffer.alloc((8 * 1024 * 1024) + 1, 0x20));
  await rewriteFixtureChecksums(tooLarge.packageDirectory);
  const tooLargeOutput = join(tooLargeParent, "upload-root");
  await assert.rejects(stageQwen35VisionHuggingFacePackage({
    packageDirectory: tooLarge.packageDirectory,
    outputDirectory: tooLargeOutput,
  }, tooLarge.contract), { code: "vision-hf-stage-input-invalid" });
  await assert.rejects(lstat(tooLargeOutput), { code: "ENOENT" });
});

test("rejects metadata growth after handle stat without reserving output", async () => {
  const parent = await mkdtemp(join(tmpdir(), "qwen35-vision-hf-metadata-growth-"));
  const fixture = await packageFixture(parent);
  const outputDirectory = join(parent, "upload-root");
  let grewAfterStat = false;
  await assert.rejects(stageQwen35VisionHuggingFacePackage({
    packageDirectory: fixture.packageDirectory,
    outputDirectory,
  }, fixture.contract, {
    afterMetadataStat: async ({ label }) => {
      if (label === "Vision layer index") {
        grewAfterStat = true;
        await appendFile(join(fixture.packageDirectory, "layer-index.json"), Buffer.alloc(8 * 1024 * 1024, 0x20));
      }
    },
  }), { code: "vision-hf-stage-input-mutated" });
  assert.equal(grewAfterStat, true);
  await assert.rejects(lstat(outputDirectory), { code: "ENOENT" });
});

test("rejects coverage-complete layer indexes that do not match the fixed release structure", async () => {
  const correctGroups = [
    { layer: "bootstrap", shards: [0, 1] },
    ...Array.from({ length: 24 }, (_, index) => ({ layer: String(index), shards: [index + 2] })),
  ];
  const invalidGroups = [
    [
      { layer: "bootstrap", shards: [1, 0] },
      ...correctGroups.slice(1),
    ],
    [
      correctGroups[0],
      correctGroups[2],
      correctGroups[1],
      ...correctGroups.slice(3),
    ],
    [
      correctGroups[0],
      { layer: "not-a-release-layer", shards: [2] },
      ...correctGroups.slice(2),
    ],
  ];

  for (const [index, groups] of invalidGroups.entries()) {
    const parent = await mkdtemp(join(tmpdir(), `qwen35-vision-hf-index-${index}-`));
    const fixture = await packageFixture(parent);
    await writeFile(join(fixture.packageDirectory, "layer-index.json"), layerIndex(groups));
    await rewriteFixtureChecksums(fixture.packageDirectory);
    const outputDirectory = join(parent, "upload-root");
    await assert.rejects(stageQwen35VisionHuggingFacePackage({
      packageDirectory: fixture.packageDirectory,
      outputDirectory,
    }, fixture.contract), { code: "vision-hf-stage-layer-index-invalid" });
    await assert.rejects(lstat(outputDirectory), { code: "ENOENT" });
  }
});

test("publishes the vision staging command without local environment details", async () => {
  const packageJson = JSON.parse(await readFile("package.json", "utf8")) as {
    scripts: Record<string, string>;
  };
  assert.equal(
    packageJson.scripts["stage:qwen35-vision-huggingface"],
    "node --import tsx tools/stage-qwen35-vision-huggingface.ts",
  );
  const guide = await readFile("docs/model-package-conversion.md", "utf8");
  assert.match(guide, /stage:qwen35-vision-huggingface/u);
  assert.match(guide, /collision-safe flat names/u);
  assert.equal(guide.includes("/Users/"), false);
});
