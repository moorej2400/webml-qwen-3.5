import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import test from "node:test";

import { GgmlType } from "../src/gguf.js";
import {
  stringifyManifest,
  type ModelPackageManifest,
  type TensorLayoutEntry,
} from "../src/manifest.js";
import {
  createIntegrityValidatedQwen35VisionPackage,
  type Qwen35IntegrityValidatedVisionPackage,
} from "../src/qwen35-vision-package-loader.js";
import {
  assertIntegrityValidatedQwen35VisionProgram,
  createQwen35VisionProgram,
} from "../src/qwen35-vision-program.js";

const F32 = GgmlType.F32;
const BF16 = GgmlType.BF16;
const BASE_URL = "https://huggingface.co/public-fixtures/qwen-vision/resolve/0123456789abcdef0123456789abcdef01234567/";

function sha256(bytes: Uint8Array): string {
  return createHash("sha256").update(bytes).digest("hex");
}

function byteLength(shape: readonly number[], ggmlType: number): number {
  return shape.reduce((product, dimension) => product * dimension, 1) * (ggmlType === F32 ? 4 : 2);
}

function segment(
  name: string,
  shape: readonly number[],
  ggmlType: number,
  shard: number,
  shardOffset: number,
  tensorOffset = 0,
  length = byteLength(shape, ggmlType),
): TensorLayoutEntry {
  return {
    name,
    shape: shape.map(String),
    ggmlType,
    storageType: ggmlType === F32 ? "f32" : "raw",
    shard,
    shardOffset: String(shardOffset),
    tensorOffset: String(tensorOffset),
    length: String(length),
  };
}

function completeVisionManifest(): ModelPackageManifest {
  const tensorLayout: TensorLayoutEntry[] = [];
  const bootstrap: Array<readonly [string, readonly number[], number]> = [
    ["mm.0.bias", [4_096], F32],
    ["mm.0.weight", [4_096, 4_096], BF16],
    ["mm.2.bias", [2_560], F32],
    ["mm.2.weight", [4_096, 2_560], BF16],
    ["v.patch_embd.bias", [1_024], F32],
    ["v.patch_embd.weight", [16, 16, 3, 1_024], F32],
    ["v.patch_embd.weight.1", [16, 16, 3, 1_024], F32],
  ];
  let bootstrapOffset = 0;
  for (const [name, shape, ggmlType] of bootstrap) {
    tensorLayout.push(segment(name, shape, ggmlType, 0, bootstrapOffset));
    bootstrapOffset += byteLength(shape, ggmlType);
  }
  const positionLength = byteLength([1_024, 2_304], F32);
  const positionFirstSegment = 6_258_688;
  tensorLayout.push(segment(
    "v.position_embd.weight", [1_024, 2_304], F32, 0, bootstrapOffset, 0, positionFirstSegment,
  ));
  tensorLayout.push(segment(
    "v.position_embd.weight", [1_024, 2_304], F32, 1, 0, positionFirstSegment,
    positionLength - positionFirstSegment,
  ));
  tensorLayout.push(segment("v.post_ln.bias", [1_024], F32, 1, positionLength - positionFirstSegment));
  tensorLayout.push(segment("v.post_ln.weight", [1_024], F32, 1, positionLength - positionFirstSegment + 4_096));

  for (let layer = 0; layer < 24; layer += 1) {
    const prefix = `v.blk.${layer}`;
    const entries: Array<readonly [string, readonly number[], number]> = [
      [`${prefix}.attn_out.bias`, [1_024], F32],
      [`${prefix}.attn_out.weight`, [1_024, 1_024], BF16],
      [`${prefix}.attn_qkv.bias`, [3_072], F32],
      [`${prefix}.attn_qkv.weight`, [1_024, 3_072], BF16],
      [`${prefix}.ffn_down.bias`, [1_024], F32],
      [`${prefix}.ffn_down.weight`, [4_096, 1_024], BF16],
      [`${prefix}.ffn_up.bias`, [4_096], F32],
      [`${prefix}.ffn_up.weight`, [1_024, 4_096], BF16],
      [`${prefix}.ln1.bias`, [1_024], F32],
      [`${prefix}.ln1.weight`, [1_024], F32],
      [`${prefix}.ln2.bias`, [1_024], F32],
      [`${prefix}.ln2.weight`, [1_024], F32],
    ];
    let shardOffset = 0;
    for (const [name, shape, ggmlType] of entries) {
      tensorLayout.push(segment(name, shape, ggmlType, layer + 2, shardOffset));
      shardOffset += byteLength(shape, ggmlType);
    }
  }

  const shardLengths = [67_106_816, 3_186_688, ...Array.from({ length: 24 }, () => 25_219_072)];
  let packageOffset = 0;
  return {
    format: "webml-qwen-package",
    version: 1,
    packageKind: "vision",
    source: {
      repository: "https://huggingface.co/bartowski/Qwen_Qwen3.5-4B-GGUF",
      revision: "4168f45a16a1290d65a4ec0fa312ae917a4c15d6",
      file: "mmproj-Qwen_Qwen3.5-4B-bf16.gguf",
      size: "675569216",
      sha256: "463f39bd1c291c1186c319a8c90ff8640aafa678b14cbee2232d695113dfbb66",
    },
    runtime: { abi: "qwen35-webgpu-vision-v1" },
    tokenizer: {
      repository: "https://huggingface.co/Qwen/Qwen3.5-4B",
      revision: "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
      file: "tokenizer.json", size: "12807982",
      sha256: "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
    },
    processor: {
      repository: "https://huggingface.co/Qwen/Qwen3.5-4B",
      revision: "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
      file: "preprocessor_config.json", size: "390",
      sha256: "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
    },
    processorSettings: {
      processorClass: "Qwen3VLProcessor", imageProcessorType: "Qwen2VLImageProcessorFast",
      patchSize: 16, temporalPatchSize: 2, mergeSize: 2,
      shortestEdge: 65_536, longestEdge: 16_777_216,
      imageMean: [0.5, 0.5, 0.5], imageStd: [0.5, 0.5, 0.5],
    },
    tensorLayout,
    shards: shardLengths.map((length, index) => {
      const shard = {
        url: `vision-${String(index).padStart(5, "0")}.bin`,
        offset: String(packageOffset), length: String(length), sha256: String(index % 10).repeat(64),
      };
      packageOffset += length;
      return shard;
    }),
    excludedTensors: [],
  };
}

function completeVisionPackage(): Qwen35IntegrityValidatedVisionPackage {
  const manifest = completeVisionManifest();
  const manifestBytes = new TextEncoder().encode(stringifyManifest(manifest));
  const layerIndexBytes = new TextEncoder().encode(JSON.stringify({
    format: "webml-qwen-vision-layer-index",
    version: 1,
    groups: [
      { layer: "bootstrap", shards: [0, 1] },
      ...Array.from({ length: 24 }, (_, layer) => ({ layer: String(layer), shards: [layer + 2] })),
    ],
  }));
  return createIntegrityValidatedQwen35VisionPackage({
    manifestBytes,
    layerIndexBytes,
    pins: {
      packageBaseUrl: BASE_URL,
      expectedPackageBaseUrl: BASE_URL,
      expectedManifestSha256: sha256(manifestBytes),
      expectedLayerIndexSha256: sha256(layerIndexBytes),
    },
  });
}

test("derives immutable bootstrap and 24-layer descriptors from the exact Qwen3.5 vision directory", () => {
  const program = createQwen35VisionProgram(completeVisionPackage());

  assert.equal(program.architecture.layerCount, 24);
  assert.equal(program.architecture.hiddenSize, 1_024);
  assert.equal(program.architecture.headCount, 16);
  assert.equal(program.architecture.headDimension, 64);
  assert.equal(program.architecture.feedForwardSize, 4_096);
  assert.equal(program.architecture.projectedHiddenSize, 2_560);
  assert.equal(program.bootstrap.positionEmbedding.segments.length, 2);
  assert.deepEqual(program.bootstrap.positionEmbedding.segments.map((entry) => entry.shard), [0, 1]);
  assert.equal(program.bootstrap.patchEmbedding.temporalWeights.length, 2);
  assert.equal(program.bootstrap.merger.output.weight.name, "mm.2.weight");
  assert.equal(program.layers.length, 24);
  assert.equal(program.layers[0]!.shard, 2);
  assert.equal(program.layers[23]!.shard, 25);
  assert.equal(program.layers[0]!.attention.qkv.weight.precision, "bf16");
  assert.equal(program.layers[0]!.attention.qkv.weight.storageType, "raw");
  assert.equal(program.layers[0]!.normalization.preAttention.weight.precision, "f32");
  assert.equal(program.layers[0]!.activation, "gelu-pytorch-tanh");
  assert.ok(Object.isFrozen(program));
  assert.ok(Object.isFrozen(program.bootstrap.positionEmbedding.segments));
  assert.ok(Object.isFrozen(program.layers));
  assert.ok(Object.isFrozen(program.layers[0]!));
});

test("encodes the official position, rotary, attention, residual, merger, and tensor-orientation semantics", () => {
  const program = createQwen35VisionProgram(completeVisionPackage());

  assert.deepEqual(program.architecture.learnedPosition, {
    tableShape: [48, 48, 1_024],
    interpolation: "bilinear",
    alignCorners: true,
    tapsPerPatch: 4,
    patchOrder: "temporal-then-spatial-merge-block-major",
  });
  assert.deepEqual(program.architecture.rotary, {
    coordinateAxes: ["height", "width"],
    base: 10_000,
    frequencyCountPerAxis: 16,
    rotatedDimensionsPerHead: 64,
    frequencyLayout: "height-16-then-width-16-duplicated-for-rotate-half",
    qkOnly: true,
    arithmetic: "f32-then-cast-to-input",
    applicationOrder: "qkv-split-then-qk-rotate-half",
    positionOrder: "temporal-then-spatial-merge-block-major",
  });
  assert.deepEqual(program.architecture.attention, {
    segmentation: "one-non-causal-segment-per-image-frame",
    scaling: 0.125,
    dropout: 0,
  });
  assert.equal(program.architecture.layerNormEpsilon, 1e-6);
  assert.deepEqual(program.executionSequence, [
    "patch-conv3d", "interpolate-learned-position", "add-learned-position",
    "prepare-2d-rotary", "prepare-per-image-attention-segments", "layers-0-through-23",
    "merge-2x2-patches", "project-to-language-hidden-size",
  ]);
  assert.deepEqual(program.layers[0]!.operationSequence, [
    "save-attention-residual", "layernorm-1", "qkv-linear", "split-q-k-v",
    "apply-2d-rope-to-q-k", "per-image-non-causal-attention", "attention-output-linear",
    "add-attention-residual", "save-mlp-residual", "layernorm-2", "mlp-up-linear",
    "gelu-pytorch-tanh", "mlp-down-linear", "add-mlp-residual",
  ]);
  assert.deepEqual(program.bootstrap.merger.operationSequence, [
    "layernorm-before-spatial-shuffle", "reshape-2x2-merge-blocks", "input-linear",
    "gelu-exact", "output-linear",
  ]);
  assert.deepEqual(program.architecture.tensorOrientation.linear, {
    manifestShape: ["input-width", "output-rows"],
    contiguousDimension: "input-width",
  });
  assert.deepEqual(program.architecture.tensorOrientation.patchConv3d, {
    manifestShape: ["kernel-width", "kernel-height", "input-channel", "output-channel"],
    temporalSliceOrder: ["v.patch_embd.weight", "v.patch_embd.weight.1"],
    accumulation: "sum-slice-0-then-slice-1-then-add-one-bias",
    kernel: [2, 16, 16],
    stride: [2, 16, 16],
  });
  assert.ok(Object.isFrozen(program.layers[0]!.operationSequence));
  assert.ok(Object.isFrozen(program.architecture.tensorOrientation.patchConv3d));
});

test("rejects an unbranded structural clone and a schema-valid raw manifest", () => {
  const authenticated = completeVisionPackage();
  assert.throws(
    () => createQwen35VisionProgram({ ...authenticated }),
    { code: "vision-package-integrity-unvalidated" },
  );
  assert.throws(
    () => createQwen35VisionProgram(completeVisionManifest() as unknown as Qwen35IntegrityValidatedVisionPackage),
    { code: "vision-package-integrity-unvalidated" },
  );
});

test("binds each vision program to the exact authenticated package that created it", () => {
  const firstPackage = completeVisionPackage();
  const secondPackage = completeVisionPackage();
  const program = createQwen35VisionProgram(firstPackage);
  assert.equal(assertIntegrityValidatedQwen35VisionProgram(program, firstPackage), program);
  assert.throws(
    () => assertIntegrityValidatedQwen35VisionProgram({ ...program } as Qwen35VisionProgram, firstPackage),
    { code: "vision-program-integrity-unvalidated" },
  );
  assert.throws(
    () => assertIntegrityValidatedQwen35VisionProgram(program, secondPackage),
    { code: "vision-program-package-mismatch" },
  );
});

test("rejects missing, unknown, duplicated, drifted, unaligned, and cross-owner vision tensors", () => {
  const cases: Array<readonly [string, (manifest: ModelPackageManifest) => void]> = [
    ["missing", (manifest) => { manifest.tensorLayout = manifest.tensorLayout.filter((entry) => entry.name !== "v.blk.7.ln2.weight"); }],
    ["unknown", (manifest) => { manifest.tensorLayout.push(segment("v.extra.weight", [1], F32, 25, 25_218_816)); }],
    ["duplicated", (manifest) => { manifest.tensorLayout.push({ ...manifest.tensorLayout[0]! }); }],
    ["drifted", (manifest) => { manifest.tensorLayout.find((entry) => entry.name === "v.blk.0.attn_qkv.weight")!.shape = ["1_024", "1"]; }],
    ["unaligned", (manifest) => { manifest.tensorLayout.find((entry) => entry.name === "v.blk.0.attn_out.weight")!.shardOffset = "4"; }],
    ["cross-owner", (manifest) => { manifest.tensorLayout.find((entry) => entry.name === "v.blk.0.attn_out.bias")!.shard = 3; }],
  ];
  for (const [label, change] of cases) {
    const manifest = completeVisionManifest();
    change(manifest);
    const manifestBytes = new TextEncoder().encode(JSON.stringify(manifest));
    const layerIndexBytes = new TextEncoder().encode(JSON.stringify({
      format: "webml-qwen-vision-layer-index", version: 1,
      groups: [{ layer: "bootstrap", shards: [0, 1] }, ...Array.from(
        { length: 24 }, (_, layer) => ({ layer: String(layer), shards: [layer + 2] }),
      )],
    }));
    let authenticated: Qwen35IntegrityValidatedVisionPackage;
    try {
      authenticated = createIntegrityValidatedQwen35VisionPackage({
        manifestBytes,
        layerIndexBytes,
        pins: {
          packageBaseUrl: BASE_URL, expectedPackageBaseUrl: BASE_URL,
          expectedManifestSha256: sha256(manifestBytes), expectedLayerIndexSha256: sha256(layerIndexBytes),
        },
      });
    } catch {
      continue;
    }
    assert.throws(() => createQwen35VisionProgram(authenticated), /vision directory/i, label);
  }
});
