import { GgmlType } from "./gguf.js";
import { diagnosticError } from "./diagnostics.js";
import {
  validateModelPackageManifest,
  type ModelPackageManifest,
  type TensorLayoutEntry,
} from "./manifest.js";
import {
  assertIntegrityValidatedQwen35VisionPackage,
  type Qwen35IntegrityValidatedVisionPackage,
} from "./qwen35-vision-package-loader.js";

/**
 * Resolves authenticated vision shards into the fixed Qwen3.5 execution ABI.
 * Operation-order fields are kernel requirements from the pinned oracle, not scheduling suggestions.
 */

const LAYER_COUNT = 24;
const HIDDEN_SIZE = 1_024;
const HEAD_COUNT = 16;
const FEED_FORWARD_SIZE = 4_096;
const PROJECTED_HIDDEN_SIZE = 2_560;
const PACKAGE_ALIGNMENT = 256;

export interface Qwen35VisionTensorSegment {
  readonly shard: number;
  readonly shardOffset: number;
  readonly tensorOffset: number;
  readonly byteLength: number;
}

/** A resolved tensor still points into immutable package shards; it owns no bytes. */
export interface Qwen35VisionTensor {
  readonly name: string;
  readonly shape: readonly number[];
  readonly precision: "f32" | "bf16";
  readonly storageType: "f32" | "raw";
  readonly segments: readonly Qwen35VisionTensorSegment[];
}

export interface Qwen35VisionLayerProgram {
  readonly layer: number;
  /** Each transformer layer is isolated in one shard for streamed execution. */
  readonly shard: number;
  readonly normalization: Readonly<{
    preAttention: Readonly<{ weight: Qwen35VisionTensor; bias: Qwen35VisionTensor }>;
    preMlp: Readonly<{ weight: Qwen35VisionTensor; bias: Qwen35VisionTensor }>;
  }>;
  readonly attention: Readonly<{
    qkv: Readonly<{ weight: Qwen35VisionTensor; bias: Qwen35VisionTensor }>;
    output: Readonly<{ weight: Qwen35VisionTensor; bias: Qwen35VisionTensor }>;
  }>;
  readonly mlp: Readonly<{
    up: Readonly<{ weight: Qwen35VisionTensor; bias: Qwen35VisionTensor }>;
    down: Readonly<{ weight: Qwen35VisionTensor; bias: Qwen35VisionTensor }>;
  }>;
  readonly activation: "gelu-pytorch-tanh";
  readonly operationSequence: readonly [
    "save-attention-residual",
    "layernorm-1",
    "qkv-linear",
    "split-q-k-v",
    "apply-2d-rope-to-q-k",
    "per-image-non-causal-attention",
    "attention-output-linear",
    "add-attention-residual",
    "save-mlp-residual",
    "layernorm-2",
    "mlp-up-linear",
    "gelu-pytorch-tanh",
    "mlp-down-linear",
    "add-mlp-residual",
  ];
}

export interface Qwen35VisionProgram {
  readonly architecture: Readonly<{
    layerCount: 24;
    hiddenSize: 1_024;
    headCount: 16;
    headDimension: 64;
    feedForwardSize: 4_096;
    projectedHiddenSize: 2_560;
    patchSize: 16;
    temporalPatchSize: 2;
    mergeSize: 2;
    layerNormEpsilon: 0.000001;
    learnedPosition: Readonly<{
      tableShape: readonly [48, 48, 1_024];
      interpolation: "bilinear";
      alignCorners: true;
      tapsPerPatch: 4;
      patchOrder: "temporal-then-spatial-merge-block-major";
    }>;
    rotary: Readonly<{
      coordinateAxes: readonly ["height", "width"];
      base: 10_000;
      frequencyCountPerAxis: 16;
      rotatedDimensionsPerHead: 64;
      frequencyLayout: "height-16-then-width-16-duplicated-for-rotate-half";
      qkOnly: true;
      arithmetic: "f32-then-cast-to-input";
      applicationOrder: "qkv-split-then-qk-rotate-half";
      positionOrder: "temporal-then-spatial-merge-block-major";
    }>;
    attention: Readonly<{
      segmentation: "one-non-causal-segment-per-image-frame";
      scaling: 0.125;
      dropout: 0;
    }>;
    tensorOrientation: Readonly<{
      linear: Readonly<{
        manifestShape: readonly ["input-width", "output-rows"];
        contiguousDimension: "input-width";
      }>;
      patchConv3d: Readonly<{
        manifestShape: readonly ["kernel-width", "kernel-height", "input-channel", "output-channel"];
        temporalSliceOrder: readonly ["v.patch_embd.weight", "v.patch_embd.weight.1"];
        accumulation: "sum-slice-0-then-slice-1-then-add-one-bias";
        kernel: readonly [2, 16, 16];
        stride: readonly [2, 16, 16];
      }>;
    }>;
  }>;
  readonly bootstrap: Readonly<{
    patchEmbedding: Readonly<{
      temporalWeights: readonly [Qwen35VisionTensor, Qwen35VisionTensor];
      bias: Qwen35VisionTensor;
    }>;
    positionEmbedding: Qwen35VisionTensor;
    postLayerNorm: Readonly<{ weight: Qwen35VisionTensor; bias: Qwen35VisionTensor }>;
    merger: Readonly<{
      input: Readonly<{ weight: Qwen35VisionTensor; bias: Qwen35VisionTensor }>;
      output: Readonly<{ weight: Qwen35VisionTensor; bias: Qwen35VisionTensor }>;
      normalization: "layernorm-before-spatial-shuffle";
      layerNormEpsilon: 0.000001;
      activation: "gelu-exact";
      operationSequence: readonly [
        "layernorm-before-spatial-shuffle",
        "reshape-2x2-merge-blocks",
        "input-linear",
        "gelu-exact",
        "output-linear",
      ];
    }>;
  }>;
  readonly layers: readonly Qwen35VisionLayerProgram[];
  readonly executionSequence: readonly [
    "patch-conv3d",
    "interpolate-learned-position",
    "add-learned-position",
    "prepare-2d-rotary",
    "prepare-per-image-attention-segments",
    "layers-0-through-23",
    "merge-2x2-patches",
    "project-to-language-hidden-size",
  ];
}

interface TensorContract {
  readonly name: string;
  readonly shape: readonly number[];
  readonly ggmlType: typeof GgmlType.F32 | typeof GgmlType.BF16;
  readonly storageType: "f32" | "raw";
  readonly owner: "bootstrap" | number;
  readonly segmentCount: 1 | 2;
}

function fail(): never {
  throw new Error("Vision directory does not match the fixed Qwen3.5 4B architecture");
}

function contract(
  name: string,
  shape: readonly number[],
  ggmlType: TensorContract["ggmlType"],
  owner: TensorContract["owner"],
  segmentCount: TensorContract["segmentCount"] = 1,
): TensorContract {
  return Object.freeze({
    name,
    shape: Object.freeze([...shape]),
    ggmlType,
    storageType: ggmlType === GgmlType.F32 ? "f32" : "raw",
    owner,
    segmentCount,
  });
}

function bootstrapContracts(): readonly TensorContract[] {
  return Object.freeze([
    contract("mm.0.bias", [FEED_FORWARD_SIZE], GgmlType.F32, "bootstrap"),
    contract("mm.0.weight", [FEED_FORWARD_SIZE, FEED_FORWARD_SIZE], GgmlType.BF16, "bootstrap"),
    contract("mm.2.bias", [PROJECTED_HIDDEN_SIZE], GgmlType.F32, "bootstrap"),
    contract("mm.2.weight", [FEED_FORWARD_SIZE, PROJECTED_HIDDEN_SIZE], GgmlType.BF16, "bootstrap"),
    contract("v.patch_embd.bias", [HIDDEN_SIZE], GgmlType.F32, "bootstrap"),
    contract("v.patch_embd.weight", [16, 16, 3, HIDDEN_SIZE], GgmlType.F32, "bootstrap"),
    contract("v.patch_embd.weight.1", [16, 16, 3, HIDDEN_SIZE], GgmlType.F32, "bootstrap"),
    contract("v.position_embd.weight", [HIDDEN_SIZE, 2_304], GgmlType.F32, "bootstrap", 2),
    contract("v.post_ln.bias", [HIDDEN_SIZE], GgmlType.F32, "bootstrap"),
    contract("v.post_ln.weight", [HIDDEN_SIZE], GgmlType.F32, "bootstrap"),
  ]);
}

function layerContracts(layer: number): readonly TensorContract[] {
  const prefix = `v.blk.${layer}`;
  return Object.freeze([
    contract(`${prefix}.attn_out.bias`, [HIDDEN_SIZE], GgmlType.F32, layer),
    contract(`${prefix}.attn_out.weight`, [HIDDEN_SIZE, HIDDEN_SIZE], GgmlType.BF16, layer),
    contract(`${prefix}.attn_qkv.bias`, [HIDDEN_SIZE * 3], GgmlType.F32, layer),
    contract(`${prefix}.attn_qkv.weight`, [HIDDEN_SIZE, HIDDEN_SIZE * 3], GgmlType.BF16, layer),
    contract(`${prefix}.ffn_down.bias`, [HIDDEN_SIZE], GgmlType.F32, layer),
    contract(`${prefix}.ffn_down.weight`, [FEED_FORWARD_SIZE, HIDDEN_SIZE], GgmlType.BF16, layer),
    contract(`${prefix}.ffn_up.bias`, [FEED_FORWARD_SIZE], GgmlType.F32, layer),
    contract(`${prefix}.ffn_up.weight`, [HIDDEN_SIZE, FEED_FORWARD_SIZE], GgmlType.BF16, layer),
    contract(`${prefix}.ln1.bias`, [HIDDEN_SIZE], GgmlType.F32, layer),
    contract(`${prefix}.ln1.weight`, [HIDDEN_SIZE], GgmlType.F32, layer),
    contract(`${prefix}.ln2.bias`, [HIDDEN_SIZE], GgmlType.F32, layer),
    contract(`${prefix}.ln2.weight`, [HIDDEN_SIZE], GgmlType.F32, layer),
  ]);
}

function contracts(): readonly TensorContract[] {
  return Object.freeze([
    ...bootstrapContracts(),
    ...Array.from({ length: LAYER_COUNT }, (_, layer) => layerContracts(layer)).flat(),
  ]);
}

const CONTRACTS = contracts();
const CONTRACT_BY_NAME = new Map(CONTRACTS.map((entry) => [entry.name, entry]));

// A structurally identical program is not sufficient: tensor offsets are safe
// to stage only when this exact descriptor came from the package it will load.
const INTEGRITY_VALIDATED_PROGRAMS = new WeakMap<Qwen35VisionProgram, Qwen35IntegrityValidatedVisionPackage>();

/** Verifies that a program was derived from this exact integrity-validated package. */
export function assertIntegrityValidatedQwen35VisionProgram(
  program: unknown,
  package_: Qwen35IntegrityValidatedVisionPackage,
): Qwen35VisionProgram {
  if (typeof program !== "object" || program === null || !INTEGRITY_VALIDATED_PROGRAMS.has(program as Qwen35VisionProgram)) {
    throw diagnosticError("vision-program-integrity-unvalidated", "Vision program was not created by the integrity-validated package program builder");
  }
  if (INTEGRITY_VALIDATED_PROGRAMS.get(program as Qwen35VisionProgram) !== package_) {
    throw diagnosticError("vision-program-package-mismatch", "Vision program belongs to a different integrity-validated package");
  }
  return program as Qwen35VisionProgram;
}

function sameShape(actual: readonly string[], expected: readonly number[]): boolean {
  return actual.length === expected.length && actual.every((value, index) => value === String(expected[index]));
}

function safeInteger(value: string): number {
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed < 0 || String(parsed) !== value) fail();
  return parsed;
}

function expectedShard(owner: TensorContract["owner"]): readonly number[] {
  return owner === "bootstrap" ? [0, 1] : [owner + 2];
}

function freeze<T>(value: T): T {
  if (typeof value !== "object" || value === null || Object.isFrozen(value)) return value;
  for (const child of Object.values(value)) freeze(child);
  return Object.freeze(value);
}

function validateManifest(manifest: ModelPackageManifest): Map<string, readonly TensorLayoutEntry[]> {
  try {
    validateModelPackageManifest(manifest);
  } catch {
    fail();
  }
  if (
    manifest.packageKind !== "vision" ||
    manifest.runtime.abi !== "qwen35-webgpu-vision-v1" ||
    manifest.shards.length !== LAYER_COUNT + 2 ||
    manifest.excludedTensors.length !== 0 ||
    manifest.processorSettings?.patchSize !== 16 ||
    manifest.processorSettings.temporalPatchSize !== 2 ||
    manifest.processorSettings.mergeSize !== 2
  ) fail();

  const grouped = new Map<string, TensorLayoutEntry[]>();
  for (const entry of manifest.tensorLayout) {
    const expected = CONTRACT_BY_NAME.get(entry.name);
    if (
      expected === undefined ||
      entry.ggmlType !== expected.ggmlType ||
      entry.storageType !== expected.storageType ||
      !sameShape(entry.shape, expected.shape)
    ) fail();
    const shardOffset = safeInteger(entry.shardOffset);
    const tensorOffset = safeInteger(entry.tensorOffset);
    const length = safeInteger(entry.length);
    const owners = expectedShard(expected.owner);
    // Converter segments are 256-byte aligned so future direct WebGPU uploads
    // never require a copy solely to repair an otherwise-valid tensor range.
    if (
      !owners.includes(entry.shard) ||
      shardOffset % PACKAGE_ALIGNMENT !== 0 ||
      tensorOffset % PACKAGE_ALIGNMENT !== 0 ||
      length % PACKAGE_ALIGNMENT !== 0
    ) fail();
    grouped.set(entry.name, [...(grouped.get(entry.name) ?? []), entry]);
  }

  if (grouped.size !== CONTRACTS.length) fail();
  for (const expected of CONTRACTS) {
    const entries = grouped.get(expected.name);
    if (entries === undefined || entries.length !== expected.segmentCount) fail();
    const owners = expectedShard(expected.owner);
    if (entries.some((entry) => !owners.includes(entry.shard))) fail();
    entries.sort((left, right) => safeInteger(left.tensorOffset) - safeInteger(right.tensorOffset));
    if (entries.some((entry, index) => index > 0 && entry.shard === entries[index - 1]!.shard && entry.shardOffset < entries[index - 1]!.shardOffset)) fail();
  }
  const position = grouped.get("v.position_embd.weight")!;
  if (position[0]!.shard !== 0 || position[1]!.shard !== 1) fail();
  return grouped;
}

function toTensor(name: string, grouped: ReadonlyMap<string, readonly TensorLayoutEntry[]>): Qwen35VisionTensor {
  const expected = CONTRACT_BY_NAME.get(name);
  const entries = grouped.get(name);
  if (expected === undefined || entries === undefined) fail();
  return freeze({
    name,
    shape: Object.freeze([...expected.shape]),
    precision: expected.ggmlType === GgmlType.F32 ? "f32" : "bf16",
    storageType: expected.storageType,
    segments: Object.freeze(entries.map((entry) => Object.freeze({
      shard: entry.shard,
      shardOffset: safeInteger(entry.shardOffset),
      tensorOffset: safeInteger(entry.tensorOffset),
      byteLength: safeInteger(entry.length),
    }))),
  });
}

function layerProgram(layer: number, grouped: ReadonlyMap<string, readonly TensorLayoutEntry[]>): Qwen35VisionLayerProgram {
  const prefix = `v.blk.${layer}`;
  const tensor = (suffix: string) => toTensor(`${prefix}.${suffix}`, grouped);
  return freeze({
    layer,
    shard: layer + 2,
    normalization: {
      preAttention: { weight: tensor("ln1.weight"), bias: tensor("ln1.bias") },
      preMlp: { weight: tensor("ln2.weight"), bias: tensor("ln2.bias") },
    },
    attention: {
      qkv: { weight: tensor("attn_qkv.weight"), bias: tensor("attn_qkv.bias") },
      output: { weight: tensor("attn_out.weight"), bias: tensor("attn_out.bias") },
    },
    mlp: {
      up: { weight: tensor("ffn_up.weight"), bias: tensor("ffn_up.bias") },
      down: { weight: tensor("ffn_down.weight"), bias: tensor("ffn_down.bias") },
    },
    activation: "gelu-pytorch-tanh" as const,
    operationSequence: [
      "save-attention-residual",
      "layernorm-1",
      "qkv-linear",
      "split-q-k-v",
      "apply-2d-rope-to-q-k",
      "per-image-non-causal-attention",
      "attention-output-linear",
      "add-attention-residual",
      "save-mlp-residual",
      "layernorm-2",
      "mlp-up-linear",
      "gelu-pytorch-tanh",
      "mlp-down-linear",
      "add-mlp-residual",
    ] as const,
  });
}

/**
 * Resolves only the fixed Qwen3.5 vision directory after package authentication.
 * It deliberately contains no GPU resources or generic tensor operations.
 */
export function createQwen35VisionProgram(package_: Qwen35IntegrityValidatedVisionPackage): Qwen35VisionProgram {
  const authenticatedPackage = assertIntegrityValidatedQwen35VisionPackage(package_);
  const manifest = authenticatedPackage.manifest;
  const grouped = validateManifest(manifest);
  const tensor = (name: string) => toTensor(name, grouped);
  const program: Qwen35VisionProgram = freeze({
    architecture: {
      layerCount: 24,
      hiddenSize: 1_024,
      headCount: 16,
      headDimension: 64,
      feedForwardSize: 4_096,
      projectedHiddenSize: 2_560,
      patchSize: 16,
      temporalPatchSize: 2,
      mergeSize: 2,
      layerNormEpsilon: 0.000001,
      learnedPosition: {
        tableShape: [48, 48, 1_024],
        interpolation: "bilinear",
        alignCorners: true,
        tapsPerPatch: 4,
        patchOrder: "temporal-then-spatial-merge-block-major",
      },
      rotary: {
        coordinateAxes: ["height", "width"],
        base: 10_000,
        frequencyCountPerAxis: 16,
        rotatedDimensionsPerHead: 64,
        frequencyLayout: "height-16-then-width-16-duplicated-for-rotate-half",
        qkOnly: true,
        arithmetic: "f32-then-cast-to-input",
        applicationOrder: "qkv-split-then-qk-rotate-half",
        positionOrder: "temporal-then-spatial-merge-block-major",
      },
      attention: {
        segmentation: "one-non-causal-segment-per-image-frame",
        scaling: 0.125,
        dropout: 0,
      },
      tensorOrientation: {
        linear: {
          manifestShape: ["input-width", "output-rows"],
          contiguousDimension: "input-width",
        },
        patchConv3d: {
          manifestShape: ["kernel-width", "kernel-height", "input-channel", "output-channel"],
          temporalSliceOrder: ["v.patch_embd.weight", "v.patch_embd.weight.1"],
          accumulation: "sum-slice-0-then-slice-1-then-add-one-bias",
          kernel: [2, 16, 16],
          stride: [2, 16, 16],
        },
      },
    },
    bootstrap: {
      patchEmbedding: {
        temporalWeights: [tensor("v.patch_embd.weight"), tensor("v.patch_embd.weight.1")],
        bias: tensor("v.patch_embd.bias"),
      },
      positionEmbedding: tensor("v.position_embd.weight"),
      postLayerNorm: { weight: tensor("v.post_ln.weight"), bias: tensor("v.post_ln.bias") },
      merger: {
        input: { weight: tensor("mm.0.weight"), bias: tensor("mm.0.bias") },
        output: { weight: tensor("mm.2.weight"), bias: tensor("mm.2.bias") },
        normalization: "layernorm-before-spatial-shuffle",
        layerNormEpsilon: 0.000001,
        activation: "gelu-exact",
        operationSequence: [
          "layernorm-before-spatial-shuffle",
          "reshape-2x2-merge-blocks",
          "input-linear",
          "gelu-exact",
          "output-linear",
        ],
      },
    },
    layers: Array.from({ length: LAYER_COUNT }, (_, layer) => layerProgram(layer, grouped)),
    executionSequence: [
      "patch-conv3d",
      "interpolate-learned-position",
      "add-learned-position",
      "prepare-2d-rotary",
      "prepare-per-image-attention-segments",
      "layers-0-through-23",
      "merge-2x2-patches",
      "project-to-language-hidden-size",
    ],
  });
  INTEGRITY_VALIDATED_PROGRAMS.set(program, authenticatedPackage);
  return program;
}
