import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import {
  copyFile,
  lstat,
  mkdir,
  mkdtemp,
  readFile,
  readdir,
  stat,
  writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import { GgmlType } from "../src/gguf.js";
import { stringifyManifest, type ImmutableArtifactIdentity } from "../src/manifest.js";
import {
  TOKENIZER_ARTIFACT_VERSION,
  TOKENIZER_BINARY_HEADER_BYTES,
  TOKENIZER_BINARY_MAGIC,
} from "../src/tokenizer-binary.js";
import {
  parseQwen35HuggingFaceStagingArguments,
  stageQwen35HuggingFacePackage,
  type Qwen35HuggingFaceStageContract,
} from "../tools/stage-qwen35-huggingface.js";

const REVISION = "1".repeat(40);

function sha256(bytes: Uint8Array | string): string {
  return createHash("sha256").update(bytes).digest("hex");
}

function artifact(file: string): ImmutableArtifactIdentity {
  return {
    repository: "https://example.invalid/qwen-fixture",
    revision: REVISION,
    file,
    size: "1",
    sha256: "2".repeat(64),
  };
}

function tokenizerBinary(): Uint8Array {
  const binary = new Uint8Array(TOKENIZER_BINARY_HEADER_BYTES + 8 + 1);
  binary.set(new TextEncoder().encode(TOKENIZER_BINARY_MAGIC), 0);
  const view = new DataView(binary.buffer);
  view.setUint32(8, TOKENIZER_ARTIFACT_VERSION, true);
  view.setUint32(12, 1, true);
  view.setUint32(16, 1, true);
  view.setUint32(20, 0, true);
  view.setUint32(24, 0, true);
  view.setUint32(28, 1, true);
  view.setUint32(TOKENIZER_BINARY_HEADER_BYTES, 0, true);
  view.setUint32(TOKENIZER_BINARY_HEADER_BYTES + 4, 1, true);
  binary[binary.length - 1] = 97;
  return binary;
}

async function packageFixture(parent: string): Promise<{
  packageDirectory: string;
  tokenizerBinPath: string;
  tokenizerManifestPath: string;
  contract: Qwen35HuggingFaceStageContract;
  manifest: Record<string, unknown>;
}> {
  const packageDirectory = join(parent, "converted-package");
  const shardDirectory = join(packageDirectory, "shards");
  const tokenizerDirectory = join(parent, "compiled-tokenizer");
  await mkdir(shardDirectory, { recursive: true });
  await mkdir(tokenizerDirectory, { recursive: true });
  const source = artifact("model.gguf");
  const tokenizer = artifact("tokenizer.json");
  const tokenizerConfig = artifact("tokenizer_config.json");
  const shard = Uint8Array.of(1, 2, 3, 4);
  const shardHash = sha256(shard);
  const manifest = {
    format: "webml-qwen-package" as const,
    version: 1 as const,
    packageKind: "language" as const,
    source,
    runtime: { abi: "qwen35-webgpu-v2" },
    tokenizer,
    tensorLayout: [{
      name: "output_norm.weight",
      shape: ["1"],
      ggmlType: GgmlType.F32,
      storageType: "f32" as const,
      shard: 0,
      shardOffset: "0",
      tensorOffset: "0",
      length: "4",
    }],
    shards: [{
      url: "shards/model-00000.bin",
      offset: "0",
      length: "4",
      sha256: shardHash,
    }],
    excludedTensors: [],
  };
  const manifestText = stringifyManifest(manifest);
  const licenses = "{\n  \"format\": \"webml-qwen-license-metadata\",\n  \"version\": 1\n}\n";
  const provenance = "{\n  \"format\": \"webml-qwen-source-provenance\",\n  \"version\": 1\n}\n";
  await writeFile(join(shardDirectory, "model-00000.bin"), shard);
  await writeFile(join(packageDirectory, "manifest.json"), manifestText);
  await writeFile(join(packageDirectory, "LICENSES.json"), licenses);
  await writeFile(join(packageDirectory, "source-provenance.json"), provenance);
  const sourceFiles = [
    ["LICENSES.json", licenses],
    ["manifest.json", manifestText],
    ["shards/model-00000.bin", shard],
    ["source-provenance.json", provenance],
  ] as const;
  await writeFile(
    join(packageDirectory, "SHA256SUMS"),
    `${sourceFiles.map(([name, bytes]) => `${sha256(bytes)}  ${name}`).join("\n")}\n`,
  );

  const binary = tokenizerBinary();
  const tokenizerManifest = {
    format: "webml-qwen35-tokenizer",
    version: 1,
    runtimeAbi: "qwen35-tokenizer-v1",
    source: { ...tokenizer, size: Number(tokenizer.size) },
    chatTemplateSource: tokenizerConfig,
    chatTemplateSha256: "3".repeat(64),
    normalization: "NFC",
    preTokenizer: "qwen35-bytelevel-v1",
    baseVocabSize: 1,
    tokenCount: 1,
    decodableTokenCount: 1,
    modelLogitRows: 248_320,
    undecodableLogitRows: 248_319,
    mergeCount: 0,
    addedTokenCount: 0,
    artifactByteLength: binary.byteLength,
    artifactSha256: sha256(binary),
  };
  const tokenizerBinPath = join(tokenizerDirectory, "tokenizer.bin");
  const tokenizerManifestPath = join(tokenizerDirectory, "tokenizer.manifest.json");
  await writeFile(tokenizerBinPath, binary);
  await writeFile(
    tokenizerManifestPath,
    `${JSON.stringify(tokenizerManifest, null, 2)}\n`,
  );
  return {
    packageDirectory,
    tokenizerBinPath,
    tokenizerManifestPath,
    manifest,
    contract: {
      source,
      tokenizer,
      tokenizerConfig,
      chatTemplateSha256: "3".repeat(64),
      tokenizerArtifact: {
        byteLength: binary.byteLength,
        sha256: sha256(binary),
        baseVocabSize: 1,
        mergeCount: 0,
        addedTokenCount: 0,
        decodableTokenCount: 1,
        modelLogitRows: 248_320,
      },
    },
  };
}

test("requires explicit package, tokenizer, and outside-Git output paths", () => {
  assert.deepEqual(parseQwen35HuggingFaceStagingArguments([
    "--package", "converted",
    "--tokenizer-bin", "compiled/tokenizer.bin",
    "--tokenizer-manifest", "compiled/tokenizer.manifest.json",
    "--output", "upload-root",
  ]), {
    packageDirectory: "converted",
    tokenizerBinPath: "compiled/tokenizer.bin",
    tokenizerManifestPath: "compiled/tokenizer.manifest.json",
    outputDirectory: "upload-root",
  });
  assert.throws(
    () => parseQwen35HuggingFaceStagingArguments([
      "--package", "converted", "--output", "upload-root",
    ]),
    /requires --package, --tokenizer-bin, --tokenizer-manifest, and --output/u,
  );
  assert.throws(
    () => parseQwen35HuggingFaceStagingArguments([
      "--package", "converted", "--tokenizer-bin", "tokenizer.bin",
      "--tokenizer-manifest", "tokenizer.json", "--output", "upload-root",
      "--token", "private",
    ]),
    /unknown argument/u,
  );
});

test("authenticates and creates one flat independently owned upload root", async () => {
  const parent = await mkdtemp(join(tmpdir(), "qwen35-hf-stage-link-"));
  const fixture = await packageFixture(parent);
  const outputDirectory = join(parent, "upload-root");
  const report = await stageQwen35HuggingFacePackage({
    packageDirectory: fixture.packageDirectory,
    tokenizerBinPath: fixture.tokenizerBinPath,
    tokenizerManifestPath: fixture.tokenizerManifestPath,
    outputDirectory,
  }, fixture.contract);

  assert.deepEqual(report, {
    format: "webml-qwen-huggingface-stage-report",
    version: 1,
    files: [
      "LICENSES.json",
      "SHA256SUMS",
      "manifest.json",
      "model-00000.bin",
      "publication-provenance.json",
      "source-provenance.json",
      "tokenizer.bin",
      "tokenizer.manifest.json",
    ],
    clonedShardCount: 1,
  });
  assert.deepEqual((await readdir(outputDirectory)).sort(), report.files);
  assert.notEqual(
    (await stat(join(fixture.packageDirectory, "shards/model-00000.bin"))).ino,
    (await stat(join(outputDirectory, "model-00000.bin"))).ino,
  );
  const stagedManifest = JSON.parse(
    await readFile(join(outputDirectory, "manifest.json"), "utf8"),
  );
  assert.deepEqual(stagedManifest, {
    ...fixture.manifest,
    shards: [{
      ...(fixture.manifest.shards as Array<Record<string, unknown>>)[0],
      url: "model-00000.bin",
    }],
  });
  const checksums = await readFile(join(outputDirectory, "SHA256SUMS"), "utf8");
  assert.equal(checksums.trim().split("\n").length, report.files.length - 1);
  assert.match(checksums, /^[a-f0-9]{64}  tokenizer\.bin$/mu);
  assert.match(checksums, /^[a-f0-9]{64}  model-00000\.bin$/mu);
  const provenance = await readFile(
    join(outputDirectory, "publication-provenance.json"),
    "utf8",
  );
  assert.equal(provenance.includes(parent), false);
  assert.match(provenance, /"layout": "huggingface-flat-root-v1"/u);
  assert.equal(
    (await readdir(parent)).some((name) => name.includes(".staging-")),
    false,
  );
});

test("stages independently owned shard bytes through an injected clone", async () => {
  const parent = await mkdtemp(join(tmpdir(), "qwen35-hf-stage-copy-"));
  const fixture = await packageFixture(parent);
  const outputDirectory = join(parent, "upload-root");
  let cloneAttempts = 0;
  const report = await stageQwen35HuggingFacePackage({
    packageDirectory: fixture.packageDirectory,
    tokenizerBinPath: fixture.tokenizerBinPath,
    tokenizerManifestPath: fixture.tokenizerManifestPath,
    outputDirectory,
  }, fixture.contract, {
    async cloneFile(source, destination) {
      cloneAttempts += 1;
      await copyFile(source, destination);
    },
  });

  assert.equal(cloneAttempts, 1);
  assert.equal(report.clonedShardCount, 1);
  assert.notEqual(
    (await stat(join(fixture.packageDirectory, "shards/model-00000.bin"))).ino,
    (await stat(join(outputDirectory, "model-00000.bin"))).ino,
  );
  assert.deepEqual(
    await readFile(join(outputDirectory, "model-00000.bin")),
    await readFile(join(fixture.packageDirectory, "shards/model-00000.bin")),
  );
});

test("rejects source mutation during staging and reserves output before cloning", async () => {
  const parent = await mkdtemp(join(tmpdir(), "qwen35-hf-stage-mutation-"));
  const fixture = await packageFixture(parent);
  const outputDirectory = join(parent, "upload-root");
  let outputWasReserved = false;
  await assert.rejects(stageQwen35HuggingFacePackage({
    packageDirectory: fixture.packageDirectory,
    tokenizerBinPath: fixture.tokenizerBinPath,
    tokenizerManifestPath: fixture.tokenizerManifestPath,
    outputDirectory,
  }, fixture.contract, {
    async cloneFile(source, destination) {
      await assert.rejects(mkdir(outputDirectory), { code: "EEXIST" });
      outputWasReserved = true;
      await writeFile(source, Uint8Array.of(9, 9, 9, 9));
      await copyFile(source, destination);
    },
  }), {
    code: "hf-stage-staged-shard-mismatch",
    message: "Staged shard does not match its manifest",
  });
  assert.equal(outputWasReserved, true);
});

test("rejects checksum and tokenizer drift before creating output", async () => {
  const parent = await mkdtemp(join(tmpdir(), "qwen35-hf-stage-private-marker-"));
  const fixture = await packageFixture(parent);
  const outputDirectory = join(parent, "private-marker-output");
  await writeFile(
    join(fixture.packageDirectory, "shards/model-00000.bin"),
    Uint8Array.of(9, 9, 9, 9),
  );

  await assert.rejects(stageQwen35HuggingFacePackage({
    packageDirectory: fixture.packageDirectory,
    tokenizerBinPath: fixture.tokenizerBinPath,
    tokenizerManifestPath: fixture.tokenizerManifestPath,
    outputDirectory,
  }, fixture.contract), (error: unknown) => {
    assert.equal((error as { code?: string }).code, "hf-stage-source-checksum-mismatch");
    assert.equal(String(error).includes("private-marker"), false);
    return true;
  });
  await assert.rejects(lstat(outputDirectory), { code: "ENOENT" });

  const secondParent = await mkdtemp(join(tmpdir(), "qwen35-hf-tokenizer-drift-"));
  const second = await packageFixture(secondParent);
  await writeFile(second.tokenizerBinPath, new Uint8Array(tokenizerBinary().length));
  await assert.rejects(stageQwen35HuggingFacePackage({
    packageDirectory: second.packageDirectory,
    tokenizerBinPath: second.tokenizerBinPath,
    tokenizerManifestPath: second.tokenizerManifestPath,
    outputDirectory: join(secondParent, "upload-root"),
  }, second.contract), {
    code: "hf-stage-tokenizer-checksum-mismatch",
    message: "Compiled tokenizer SHA-256 does not match its manifest",
  });
});

test("rejects extra package files and noncanonical tokenizer metadata", async () => {
  const parent = await mkdtemp(join(tmpdir(), "qwen35-hf-stage-schema-"));
  const fixture = await packageFixture(parent);
  const extra = "private local data\n";
  await writeFile(join(fixture.packageDirectory, "local-notes.txt"), extra);
  const checksumPath = join(fixture.packageDirectory, "SHA256SUMS");
  await writeFile(
    checksumPath,
    `${await readFile(checksumPath, "utf8")}${sha256(extra)}  local-notes.txt\n`,
  );
  await assert.rejects(stageQwen35HuggingFacePackage({
    packageDirectory: fixture.packageDirectory,
    tokenizerBinPath: fixture.tokenizerBinPath,
    tokenizerManifestPath: fixture.tokenizerManifestPath,
    outputDirectory: join(parent, "upload-root"),
  }, fixture.contract), {
    code: "hf-stage-checksums-invalid",
    message: "Converted package checksum set is not exact",
  });

  const secondParent = await mkdtemp(join(tmpdir(), "qwen35-hf-stage-tokenizer-schema-"));
  const second = await packageFixture(secondParent);
  const tokenizerManifest = JSON.parse(
    await readFile(second.tokenizerManifestPath, "utf8"),
  ) as Record<string, unknown>;
  tokenizerManifest.localPath = "private-marker";
  tokenizerManifest.chatTemplateSha256 = "4".repeat(64);
  await writeFile(
    second.tokenizerManifestPath,
    `${JSON.stringify(tokenizerManifest, null, 2)}\n`,
  );
  await assert.rejects(stageQwen35HuggingFacePackage({
    packageDirectory: second.packageDirectory,
    tokenizerBinPath: second.tokenizerBinPath,
    tokenizerManifestPath: second.tokenizerManifestPath,
    outputDirectory: join(secondParent, "upload-root"),
  }, second.contract), {
    code: "hf-stage-tokenizer-manifest-invalid",
    message: "Compiled tokenizer manifest schema is invalid",
  });
});

test("never overwrites an existing output directory", async () => {
  const parent = await mkdtemp(join(tmpdir(), "qwen35-hf-stage-existing-"));
  const fixture = await packageFixture(parent);
  const outputDirectory = join(parent, "upload-root");
  await mkdir(outputDirectory);
  await writeFile(join(outputDirectory, "sentinel.txt"), "preserve\n");

  await assert.rejects(stageQwen35HuggingFacePackage({
    packageDirectory: fixture.packageDirectory,
    tokenizerBinPath: fixture.tokenizerBinPath,
    tokenizerManifestPath: fixture.tokenizerManifestPath,
    outputDirectory,
  }, fixture.contract), {
    code: "hf-stage-output-exists",
    message: "Hugging Face staging output already exists",
  });
  assert.equal(await readFile(join(outputDirectory, "sentinel.txt"), "utf8"), "preserve\n");
});

test("publishes a public-safe repository command and operator guide", async () => {
  const packageJson = JSON.parse(await readFile("package.json", "utf8")) as {
    scripts: Record<string, string>;
  };
  assert.equal(
    packageJson.scripts["stage:qwen35-huggingface"],
    "node --import tsx tools/stage-qwen35-huggingface.ts",
  );
  const guide = await readFile("docs/model-package-conversion.md", "utf8");
  assert.match(guide, /stage:qwen35-huggingface/u);
  assert.match(guide, /copy-on-write/u);
  assert.equal(guide.includes("/Users/"), false);
});
