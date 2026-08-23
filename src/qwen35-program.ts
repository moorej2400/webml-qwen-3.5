import { GgmlType, type GgmlType as GgmlTypeValue } from "./gguf.js";
import type { GemvLayout } from "./mixed-gemv.js";
import { isMtpTensorName } from "./tensor-policy.js";
import { QWEN35_NO_SELECTED_TOKEN } from "./qwen35-logits-reduction.js";
import {
  assertQwen35Config,
  type Qwen35Config,
} from "./qwen35-config.js";

export interface Qwen35TensorDirectoryEntry {
  readonly name: string;
  /** GGML order: contiguous row width is shape[0]. */
  readonly shape: readonly number[];
  readonly ggmlType: GgmlTypeValue;
  readonly storageType: GemvLayout;
}

interface TensorContract {
  readonly name: string;
  readonly shape: readonly number[];
}

export interface LinearAttentionTensorBindings {
  readonly gate: string;
  readonly qkv: string;
  readonly a: string;
  readonly alpha: string;
  readonly beta: string;
  readonly convolution: string;
  readonly timeStepBias: string;
  readonly norm: string;
  readonly output: string;
}

export interface FullAttentionTensorBindings {
  readonly query: string;
  readonly key: string;
  readonly value: string;
  readonly queryNorm: string;
  readonly keyNorm: string;
  readonly output: string;
}

export type Qwen35Invocation =
  | Readonly<{
      kind: "embedding";
      weight: "token_embd.weight";
      embeddingLength: 2_560;
      vocabularySize: 248_320;
      directPackedRowRead: true;
    }>
  | Readonly<{
      kind: "rms-norm";
      layer: number | "final";
      site: "input" | "post-attention" | "final";
      weight: string;
      epsilon: number;
      fp32Accumulation: true;
    }>
  | Readonly<{
      kind: "gated-deltanet";
      layer: number;
      tensors: LinearAttentionTensorBindings;
      kernels: readonly [
        "deltanet-conv",
        "deltanet-parameters",
        "deltanet-recurrent",
        "deltanet-gated-norm",
      ];
      stateLayout: "fp32-recurrent-state";
      runnable: true;
    }>
  | Readonly<{
      kind: "full-attention";
      layer: number;
      tensors: FullAttentionTensorBindings;
      outputGate: "query-projection-second-half";
      kernels: readonly [
        "full-attention-prepare",
        "full-attention-online",
      ];
      stateLayout: "fp16-kv-pages";
      runnable: true;
    }>
  | Readonly<{
      kind: "residual-add";
      layer: number;
      site: "attention" | "mlp";
    }>
  | Readonly<{
      kind: "gemv";
      layer: number;
      projection: "ffn-gate" | "ffn-up" | "ffn-down";
      weight: string;
      rows: number;
      columns: number;
    }>
  | Readonly<{
      kind: "swiglu";
      layer: number;
      elements: 9_216;
    }>
  | Readonly<{
      kind: "tiled-tied-logits";
      weight: "token_embd.weight";
      tiedWeightOwner: "embedding";
      modelRows: 248_320;
      decodableRows: 248_070;
      columns: 2_560;
      logicalTileRows: 1_024;
      /** Physical row views may split a tile; reductions remain per logical tile. */
      mathematicalTileCount: 243;
      finalTileRows: 262;
    }>
  | Readonly<{
      kind: "greedy-logits-reduction";
      kernels: readonly ["logits-tile-top-1", "indexed-top-1"];
      mathematicalTileCount: 243;
      candidatesPerTile: 1;
      candidateCount: 243;
      candidateCapacity: 256;
      selectedTokenReadback: Readonly<{
        resource: "selected-token";
        scalarType: "u32";
        elementCount: 1;
        byteOffset: 0;
        noSelectionSentinel: typeof QWEN35_NO_SELECTED_TOKEN;
      }>;
      runnable: true;
    }>;

export interface Qwen35TensorBinding {
  readonly tensor: Qwen35TensorDirectoryEntry;
  readonly consumers: readonly string[];
}

/**
 * Lookup-only facade; a ReadonlyMap type would still expose mutable Map
 * methods at runtime when callers cast or inspect the returned object.
 */
export interface Qwen35TensorBindings
  extends Iterable<readonly [string, Qwen35TensorBinding]> {
  readonly size: number;
  get(name: string): Qwen35TensorBinding | undefined;
  has(name: string): boolean;
  entries(): MapIterator<[string, Qwen35TensorBinding]>;
}

export interface Qwen35Program {
  readonly model: "qwen35-4b";
  readonly invocations: readonly Qwen35Invocation[];
  readonly tensorBindings: Qwen35TensorBindings;
  /** The fixed program has a concrete model-weight scheduler and driver. */
  readonly runnable: true;
}

// Both ABIs represent the same pinned GGML types. ABI v2 is also supported on
// Safari through portable packed kernels; layout support is not tied to an
// optional WebGPU subgroup feature.
const SUPPORTED_LAYOUTS = new Map<GgmlTypeValue, ReadonlySet<GemvLayout>>([
  [GgmlType.F32, new Set(["f32"])],
  [GgmlType.Q8_0, new Set(["q8-0-36"])],
  [GgmlType.Q3_K, new Set([
    "q3-k-112", "q3-k-nibble-148", "q3-k-fused-f32-192",
  ])],
  [GgmlType.Q4_K, new Set(["q4-k-144", "q4-k-fused-f32-192"])],
  [GgmlType.Q5_K, new Set(["q5-k-176", "q5-k-fused-f32-224"])],
  [GgmlType.Q6_K, new Set(["q6-k-212", "q6-k-fused-f32-256"])],
]);

function contract(name: string, ...shape: number[]): TensorContract {
  return Object.freeze({ name, shape: Object.freeze(shape) });
}

function layerContracts(layer: number, full: boolean): readonly TensorContract[] {
  const prefix = `blk.${layer}`;
  const shared = [
    contract(`${prefix}.ffn_down.weight`, 9_216, 2_560),
    contract(`${prefix}.ffn_gate.weight`, 2_560, 9_216),
    contract(`${prefix}.ffn_up.weight`, 2_560, 9_216),
    contract(`${prefix}.post_attention_norm.weight`, 2_560),
  ];
  if (full) {
    // attn_q contains query and output-gate halves because the GGUF has no
    // separate full-attention gate tensor.
    return Object.freeze([
      contract(`${prefix}.attn_k.weight`, 2_560, 1_024),
      contract(`${prefix}.attn_k_norm.weight`, 256),
      contract(`${prefix}.attn_norm.weight`, 2_560),
      contract(`${prefix}.attn_output.weight`, 4_096, 2_560),
      contract(`${prefix}.attn_q.weight`, 2_560, 8_192),
      contract(`${prefix}.attn_q_norm.weight`, 256),
      contract(`${prefix}.attn_v.weight`, 2_560, 1_024),
      ...shared,
    ]);
  }
  return Object.freeze([
    contract(`${prefix}.attn_gate.weight`, 2_560, 4_096),
    contract(`${prefix}.attn_norm.weight`, 2_560),
    contract(`${prefix}.attn_qkv.weight`, 2_560, 8_192),
    contract(`${prefix}.ssm_a`, 32),
    contract(`${prefix}.ssm_alpha.weight`, 2_560, 32),
    contract(`${prefix}.ssm_beta.weight`, 2_560, 32),
    contract(`${prefix}.ssm_conv1d.weight`, 4, 8_192),
    contract(`${prefix}.ssm_dt.bias`, 32),
    contract(`${prefix}.ssm_norm.weight`, 128),
    contract(`${prefix}.ssm_out.weight`, 4_096, 2_560),
    ...shared,
  ]);
}

function requiredContracts(config: Qwen35Config): readonly TensorContract[] {
  const result = [
    contract("output_norm.weight", 2_560),
    contract("token_embd.weight", 2_560, 248_320),
  ];
  for (let layer = 0; layer < config.baseBlockCount; layer += 1) {
    result.push(...layerContracts(layer, isFullAttentionLayer(config, layer)));
  }
  return Object.freeze(result);
}

function isFullAttentionLayer(config: Qwen35Config, layer: number): boolean {
  return config.fullAttentionLayers.some((candidate) => candidate === layer);
}

function sameShape(actual: readonly number[], expected: readonly number[]): boolean {
  return (
    actual.length === expected.length &&
    expected.every((dimension, index) => actual[index] === dimension)
  );
}

function validateDirectory(
  config: Qwen35Config,
  tensors: readonly Qwen35TensorDirectoryEntry[],
): Map<string, Qwen35TensorDirectoryEntry> {
  const actual = new Map<string, Qwen35TensorDirectoryEntry>();
  for (const tensor of tensors) {
    if (isMtpTensorName(tensor.name)) {
      throw new Error(`MTP tensor ${tensor.name} must remain excluded`);
    }
    if (actual.has(tensor.name)) {
      throw new Error(`Duplicate Qwen3.5 tensor ${tensor.name}`);
    }
    const supportedLayouts = SUPPORTED_LAYOUTS.get(tensor.ggmlType);
    if (supportedLayouts === undefined || !supportedLayouts.has(tensor.storageType)) {
      throw new Error(`Unsupported type/layout for Qwen3.5 tensor ${tensor.name}`);
    }
    const snapshot = Object.freeze({
      name: tensor.name,
      shape: Object.freeze([...tensor.shape]),
      ggmlType: tensor.ggmlType,
      storageType: tensor.storageType,
    });
    actual.set(snapshot.name, snapshot);
  }
  const required = requiredContracts(config);
  const expectedNames = new Set(required.map((item) => item.name));
  for (const item of required) {
    const tensor = actual.get(item.name);
    if (tensor === undefined) {
      throw new Error(`Missing required Qwen3.5 tensor ${item.name}`);
    }
    if (!sameShape(tensor.shape, item.shape)) {
      throw new Error(`Shape mismatch for Qwen3.5 tensor ${item.name}`);
    }
  }
  for (const name of actual.keys()) {
    if (!expectedNames.has(name)) {
      throw new Error(`Unexpected base-model tensor ${name}`);
    }
  }
  return actual;
}

function immutableTensorBindings(
  entries: Iterable<readonly [string, Qwen35TensorBinding]>,
): Qwen35TensorBindings {
  const bindings = new Map(entries);
  return Object.freeze({
    size: bindings.size,
    get: (name: string) => bindings.get(name),
    has: (name: string) => bindings.has(name),
    entries: () => bindings.entries(),
    [Symbol.iterator]: () => bindings[Symbol.iterator](),
  });
}

function linearBindings(layer: number): LinearAttentionTensorBindings {
  const prefix = `blk.${layer}`;
  return Object.freeze({
    gate: `${prefix}.attn_gate.weight`,
    qkv: `${prefix}.attn_qkv.weight`,
    a: `${prefix}.ssm_a`,
    alpha: `${prefix}.ssm_alpha.weight`,
    beta: `${prefix}.ssm_beta.weight`,
    convolution: `${prefix}.ssm_conv1d.weight`,
    timeStepBias: `${prefix}.ssm_dt.bias`,
    norm: `${prefix}.ssm_norm.weight`,
    output: `${prefix}.ssm_out.weight`,
  });
}

function fullBindings(layer: number): FullAttentionTensorBindings {
  const prefix = `blk.${layer}`;
  return Object.freeze({
    query: `${prefix}.attn_q.weight`,
    key: `${prefix}.attn_k.weight`,
    value: `${prefix}.attn_v.weight`,
    queryNorm: `${prefix}.attn_q_norm.weight`,
    keyNorm: `${prefix}.attn_k_norm.weight`,
    output: `${prefix}.attn_output.weight`,
  });
}

/**
 * Builds a fixed Qwen3.5 4B invocation list. Attention operators have direct
 * kernels, state contracts, and a concrete model-weight scheduler.
 */
export function buildQwen35Program(input: {
  readonly config: Qwen35Config;
  readonly tensors: readonly Qwen35TensorDirectoryEntry[];
}): Qwen35Program {
  assertQwen35Config(input.config);
  const directory = validateDirectory(input.config, input.tensors);
  const invocations: Qwen35Invocation[] = [
    Object.freeze({
      kind: "embedding",
      weight: "token_embd.weight",
      embeddingLength: 2_560,
      vocabularySize: 248_320,
      directPackedRowRead: true,
    }),
  ];
  const consumers = new Map<string, string[]>();
  const consume = (name: string, consumer: string): void => {
    const names = consumers.get(name) ?? [];
    names.push(consumer);
    consumers.set(name, names);
  };
  consume("token_embd.weight", "embedding");

  for (let layer = 0; layer < input.config.baseBlockCount; layer += 1) {
    const prefix = `blk.${layer}`;
    const full = isFullAttentionLayer(input.config, layer);
    invocations.push(
      Object.freeze({
        kind: "rms-norm",
        layer,
        site: "input",
        weight: `${prefix}.attn_norm.weight`,
        epsilon: input.config.rmsNormEpsilon,
        fp32Accumulation: true,
      }),
      full
        ? Object.freeze({
            kind: "full-attention",
            layer,
            tensors: fullBindings(layer),
            outputGate: "query-projection-second-half",
            kernels: Object.freeze([
              "full-attention-prepare",
              "full-attention-online",
            ] as const),
            stateLayout: "fp16-kv-pages",
            runnable: true,
          })
        : Object.freeze({
            kind: "gated-deltanet",
            layer,
            tensors: linearBindings(layer),
            kernels: Object.freeze([
              "deltanet-conv",
              "deltanet-parameters",
              "deltanet-recurrent",
              "deltanet-gated-norm",
            ] as const),
            stateLayout: "fp32-recurrent-state",
            runnable: true,
          }),
      Object.freeze({ kind: "residual-add", layer, site: "attention" }),
      Object.freeze({
        kind: "rms-norm",
        layer,
        site: "post-attention",
        weight: `${prefix}.post_attention_norm.weight`,
        epsilon: input.config.rmsNormEpsilon,
        fp32Accumulation: true,
      }),
      Object.freeze({
        kind: "gemv",
        layer,
        projection: "ffn-gate",
        weight: `${prefix}.ffn_gate.weight`,
        rows: 9_216,
        columns: 2_560,
      }),
      Object.freeze({
        kind: "gemv",
        layer,
        projection: "ffn-up",
        weight: `${prefix}.ffn_up.weight`,
        rows: 9_216,
        columns: 2_560,
      }),
      Object.freeze({ kind: "swiglu", layer, elements: 9_216 }),
      Object.freeze({
        kind: "gemv",
        layer,
        projection: "ffn-down",
        weight: `${prefix}.ffn_down.weight`,
        rows: 2_560,
        columns: 9_216,
      }),
      Object.freeze({ kind: "residual-add", layer, site: "mlp" }),
    );
    for (const tensor of layerContracts(layer, full)) {
      consume(tensor.name, `layer-${layer}`);
    }
  }
  invocations.push(
    Object.freeze({
      kind: "rms-norm",
      layer: "final",
      site: "final",
      weight: "output_norm.weight",
      epsilon: input.config.rmsNormEpsilon,
      fp32Accumulation: true,
    }),
    // The embedding owns this allocation; logits only scan row shards through
    // the existing packed GEMV path.
    Object.freeze({
      kind: "tiled-tied-logits",
      weight: "token_embd.weight",
      tiedWeightOwner: "embedding",
      modelRows: 248_320,
      decodableRows: 248_070,
      columns: 2_560,
      logicalTileRows: 1_024,
      mathematicalTileCount: 243,
      finalTileRows: 262,
    }),
    Object.freeze({
      kind: "greedy-logits-reduction",
      kernels: Object.freeze([
        "logits-tile-top-1",
        "indexed-top-1",
      ] as const),
      mathematicalTileCount: 243,
      candidatesPerTile: 1,
      candidateCount: 243,
      candidateCapacity: 256,
      selectedTokenReadback: Object.freeze({
        resource: "selected-token",
        scalarType: "u32",
        elementCount: 1,
        byteOffset: 0,
        noSelectionSentinel: QWEN35_NO_SELECTED_TOKEN,
      }),
      runnable: true,
    }),
  );
  consume("output_norm.weight", "final-norm");
  consume("token_embd.weight", "tiled-logits");

  const tensorBindings = new Map<string, Qwen35TensorBinding>();
  for (const [name, tensor] of directory) {
    tensorBindings.set(
      name,
      Object.freeze({
        tensor,
        consumers: Object.freeze([...(consumers.get(name) ?? [])]),
      }),
    );
  }
  return Object.freeze({
    model: "qwen35-4b",
    invocations: Object.freeze(invocations),
    tensorBindings: immutableTensorBindings(tensorBindings),
    runnable: true,
  });
}
