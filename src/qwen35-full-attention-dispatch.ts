import { diagnosticError } from "./diagnostics.js";
import { GgmlType } from "./gguf.js";
import {
  QWEN35_HYBRID_KERNELS,
  planQwen35HybridDispatch,
  type Qwen35HybridOperation,
} from "./hybrid-kernels.js";
import {
  QWEN35_HYBRID_STATE_PAGE_TOKENS,
  type Qwen35HybridLayerResources,
} from "./hybrid-state.js";
import {
  QWEN_PRIMITIVE_KERNELS,
  planPrimitiveDispatch,
  type QwenPrimitiveOperation,
} from "./qwen-primitives.js";
import type {
  Qwen35ActivationResourceKind,
  Qwen35ActivationResourceView,
  Qwen35ActivationWorkspace,
} from "./qwen35-activation-workspace.js";
import {
  planQwen35PackedGemvDispatches,
  type Qwen35ForwardBufferSlice,
  type Qwen35ForwardDeviceLimits,
  type Qwen35ForwardDispatchPlan,
} from "./qwen35-forward-dispatch.js";
import type { Qwen35Invocation, Qwen35Program } from "./qwen35-program.js";
import type {
  Qwen35TensorWeightView,
  Qwen35WeightDirectoryView,
} from "./qwen35-weight-directory.js";
import type {
  Qwen35BufferBinding,
  Qwen35DispatchRequest,
  Qwen35KernelSource,
} from "./qwen35-webgpu-executor.js";

const HIDDEN = 2_560;
const QUERY_GATE = 8_192;
const KEY_VALUE = 1_024;
const ATTENTION_OUTPUT = 4_096;
const FFN = 9_216;
const KV_ROW_BYTES = 2_048;
const MAX_KV_PAGE_COUNT = 16_384 / QWEN35_HYBRID_STATE_PAGE_TOKENS;
const NON_ONLINE_FIXED_UNIFORM_COUNT = 6;

export type Qwen35FullAttentionLayerStage =
  | "input-rms"
  | "query-projection"
  | "key-projection"
  | "value-projection"
  | "full-attention-prepare"
  | "full-attention-online"
  | "attention-output-projection"
  | "attention-residual"
  | "post-attention-rms"
  | "ffn-gate-projection"
  | "ffn-up-projection"
  | "swiglu"
  | "ffn-down-projection"
  | "mlp-residual";

export interface Qwen35FullAttentionLayerCommand extends Qwen35DispatchRequest {
  readonly stage: Qwen35FullAttentionLayerStage;
  /** Caller writes these bit-exact u32 words into this command's uniform slot. */
  readonly uniformWords: readonly number[];
  readonly mutatesPersistentState: boolean;
}

export interface Qwen35FullAttentionLayerDispatchPlan {
  readonly layer: number;
  readonly position: number;
  readonly commands: readonly Qwen35FullAttentionLayerCommand[];
  readonly uniformCount: number;
  readonly stateSemantics: Readonly<{
    readonly mutatingStages: readonly ["full-attention-prepare"];
    readonly failureAfterSubmission: "persistent-state-indeterminate-dispose-required";
    readonly advancePositionAfter: "successful-queue-retirement";
  }>;
}

export interface PlanQwen35FullAttentionLayerDispatchInput {
  readonly program: Qwen35Program;
  readonly invocation: Extract<Qwen35Invocation, { kind: "full-attention" }>;
  readonly weights: Qwen35WeightDirectoryView;
  readonly workspace: Pick<Qwen35ActivationWorkspace, "get">;
  readonly state: Qwen35HybridLayerResources;
  readonly position: number;
  readonly capacity: number;
  readonly mropePositions: readonly [number, number, number];
  readonly limits: Qwen35ForwardDeviceLimits;
  readonly uniforms: readonly Qwen35ForwardBufferSlice[];
}

export interface PlanQwen35FullAttentionLayerGeometryInput {
  readonly program: Qwen35Program;
  readonly invocation: Extract<Qwen35Invocation, { kind: "full-attention" }>;
  readonly weights: Qwen35WeightDirectoryView;
}

export interface Qwen35FullAttentionLayerGeometry {
  readonly layer: number;
  /** Maximum fixed slots; online attention consumes one slot per resident page. */
  readonly fixedUniformCount: 70;
  readonly physicalGemvPieceCount: number;
  readonly uniformCount: number;
}

interface BufferRange {
  readonly buffer: object;
  readonly offset: number;
  readonly size: number;
}

interface LayerSequence {
  readonly inputNormWeight: string;
  readonly postAttentionNormWeight: string;
  readonly ffnGateWeight: string;
  readonly ffnUpWeight: string;
  readonly ffnDownWeight: string;
  readonly epsilon: number;
}

interface FullAttentionGeometryData {
  readonly sequence: LayerSequence;
  readonly matrixWeights: Readonly<{
    readonly query: Qwen35TensorWeightView;
    readonly key: Qwen35TensorWeightView;
    readonly value: Qwen35TensorWeightView;
    readonly output: Qwen35TensorWeightView;
    readonly ffnGate: Qwen35TensorWeightView;
    readonly ffnUp: Qwen35TensorWeightView;
    readonly ffnDown: Qwen35TensorWeightView;
  }>;
  readonly directWeights: Readonly<{
    readonly inputNorm: Qwen35TensorWeightView;
    readonly queryNorm: Qwen35TensorWeightView;
    readonly keyNorm: Qwen35TensorWeightView;
    readonly postNorm: Qwen35TensorWeightView;
  }>;
  readonly geometry: Qwen35FullAttentionLayerGeometry;
}

function fail(code: string, message: string): never {
  throw diagnosticError(code, message);
}

function isRunnableProgram(program: Qwen35Program): boolean {
  const contract = program as unknown as {
    readonly runnable?: unknown;
    readonly blockedBy?: unknown;
  };
  // A blocker field, even with an undefined value, is not the driver's exact runnable contract.
  return contract.runnable === true && !("blockedBy" in contract);
}

function sameValues(left: readonly string[], right: readonly string[]): boolean {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

function requireLayerSequence(
  program: Qwen35Program,
  invocation: PlanQwen35FullAttentionLayerDispatchInput["invocation"],
): LayerSequence {
  if (
    program.model !== "qwen35-4b" ||
    !isRunnableProgram(program) ||
    !Number.isSafeInteger(invocation.layer) ||
    invocation.layer < 0 ||
    program.invocations.indexOf(invocation) < 1 ||
    invocation.stateLayout !== "fp16-kv-pages" ||
    invocation.outputGate !== "query-projection-second-half" ||
    invocation.runnable !== true ||
    !sameValues(invocation.kernels, [
      "full-attention-prepare",
      "full-attention-online",
    ])
  ) {
    fail("full-attention-program-invalid", "The Qwen3.5 full-attention program invocation is invalid");
  }
  const prefix = `blk.${invocation.layer}`;
  const expectedTensors = {
    query: `${prefix}.attn_q.weight`,
    key: `${prefix}.attn_k.weight`,
    value: `${prefix}.attn_v.weight`,
    queryNorm: `${prefix}.attn_q_norm.weight`,
    keyNorm: `${prefix}.attn_k_norm.weight`,
    output: `${prefix}.attn_output.weight`,
  };
  if (Object.entries(expectedTensors).some(
    ([role, name]) => invocation.tensors[role as keyof typeof expectedTensors] !== name,
  )) {
    fail("full-attention-program-invalid", "The Qwen3.5 full-attention tensor names are invalid");
  }

  const index = program.invocations.indexOf(invocation);
  const inputNorm = program.invocations[index - 1];
  const attentionResidual = program.invocations[index + 1];
  const postNorm = program.invocations[index + 2];
  const ffnGate = program.invocations[index + 3];
  const ffnUp = program.invocations[index + 4];
  const swiglu = program.invocations[index + 5];
  const ffnDown = program.invocations[index + 6];
  const mlpResidual = program.invocations[index + 7];
  if (
    inputNorm?.kind !== "rms-norm" ||
    inputNorm.layer !== invocation.layer ||
    inputNorm.site !== "input" ||
    inputNorm.weight !== `${prefix}.attn_norm.weight` ||
    inputNorm.fp32Accumulation !== true ||
    attentionResidual?.kind !== "residual-add" ||
    attentionResidual.layer !== invocation.layer ||
    attentionResidual.site !== "attention" ||
    postNorm?.kind !== "rms-norm" ||
    postNorm.layer !== invocation.layer ||
    postNorm.site !== "post-attention" ||
    postNorm.weight !== `${prefix}.post_attention_norm.weight` ||
    postNorm.fp32Accumulation !== true ||
    ffnGate?.kind !== "gemv" || ffnGate.layer !== invocation.layer ||
    ffnGate.projection !== "ffn-gate" || ffnGate.weight !== `${prefix}.ffn_gate.weight` ||
    ffnGate.rows !== FFN || ffnGate.columns !== HIDDEN ||
    ffnUp?.kind !== "gemv" || ffnUp.layer !== invocation.layer ||
    ffnUp.projection !== "ffn-up" || ffnUp.weight !== `${prefix}.ffn_up.weight` ||
    ffnUp.rows !== FFN || ffnUp.columns !== HIDDEN ||
    swiglu?.kind !== "swiglu" || swiglu.layer !== invocation.layer || swiglu.elements !== FFN ||
    ffnDown?.kind !== "gemv" || ffnDown.layer !== invocation.layer ||
    ffnDown.projection !== "ffn-down" || ffnDown.weight !== `${prefix}.ffn_down.weight` ||
    ffnDown.rows !== HIDDEN || ffnDown.columns !== FFN ||
    mlpResidual?.kind !== "residual-add" || mlpResidual.layer !== invocation.layer ||
    mlpResidual.site !== "mlp" ||
    !Number.isFinite(inputNorm.epsilon) || inputNorm.epsilon < 0 ||
    postNorm.epsilon !== inputNorm.epsilon
  ) {
    fail("full-attention-program-invalid", "The Qwen3.5 full-attention layer sequence is invalid");
  }
  return Object.freeze({
    inputNormWeight: inputNorm.weight,
    postAttentionNormWeight: postNorm.weight,
    ffnGateWeight: ffnGate.weight,
    ffnUpWeight: ffnUp.weight,
    ffnDownWeight: ffnDown.weight,
    epsilon: inputNorm.epsilon,
  });
}

function sameShape(actual: readonly number[], expected: readonly number[]): boolean {
  return actual.length === expected.length && expected.every((value, index) => actual[index] === value);
}

function requireWeight(
  weights: Qwen35WeightDirectoryView,
  name: string,
  shape: readonly number[],
): Qwen35TensorWeightView {
  let tensor: Qwen35TensorWeightView | undefined;
  // The directory callback is caller-owned, so its thrown text cannot cross this boundary.
  try {
    tensor = weights.get(name);
  } catch {
    fail("full-attention-weight-invalid", "A Qwen3.5 full-attention weight is invalid");
  }
  if (tensor === undefined || tensor.name !== name || !sameShape(tensor.shape, shape)) {
    fail("full-attention-weight-invalid", "A Qwen3.5 full-attention weight is invalid");
  }
  return tensor;
}

function requirePhysicalMatrixRows(tensor: Qwen35TensorWeightView): void {
  const layout = {
    f32: { values: 1, bytes: 4, ggmlType: GgmlType.F32 },
    "q8-0-36": { values: 32, bytes: 36, ggmlType: GgmlType.Q8_0 },
    "q3-k-112": { values: 256, bytes: 112, ggmlType: GgmlType.Q3_K },
    "q4-k-144": { values: 256, bytes: 144, ggmlType: GgmlType.Q4_K },
    "q5-k-176": { values: 256, bytes: 176, ggmlType: GgmlType.Q5_K },
    "q6-k-212": { values: 256, bytes: 212, ggmlType: GgmlType.Q6_K },
  }[tensor.storageType];
  const columns = tensor.shape[0];
  const rows = tensor.shape[1];
  if (
    layout === undefined || layout.ggmlType !== tensor.ggmlType ||
    !Number.isSafeInteger(columns) || columns! <= 0 || columns! % layout.values !== 0 ||
    !Number.isSafeInteger(rows) || rows! <= 0 || tensor.rowCount !== rows ||
    tensor.rowBytes !== (columns! / layout.values) * layout.bytes ||
    tensor.logicalBytes !== BigInt(tensor.rowBytes) * BigInt(rows!) ||
    tensor.physicalRows.length < 1
  ) {
    fail("full-attention-weight-views-invalid", "A Qwen3.5 full-attention matrix layout is invalid");
  }
  let nextRow = 0;
  let nextByte = 0;
  const rangesByBuffer = new Map<object, { readonly start: number; readonly end: number }[]>();
  for (const view of tensor.physicalRows) {
    const end = view.bufferByteOffset + view.byteLength;
    const ranges = rangesByBuffer.get(view.buffer) ?? [];
    if (
      view.firstRow !== nextRow || !Number.isSafeInteger(view.rowCount) || view.rowCount <= 0 ||
      view.tensorByteOffset !== nextByte || !Number.isSafeInteger(view.bufferByteOffset) ||
      view.bufferByteOffset < 0 || view.byteLength !== view.rowCount * tensor.rowBytes ||
      !Number.isSafeInteger(end) ||
      ranges.some((range) => view.bufferByteOffset < range.end && range.start < end)
    ) {
      fail("full-attention-weight-views-invalid", "Qwen3.5 full-attention matrix rows are invalid");
    }
    ranges.push({ start: view.bufferByteOffset, end });
    rangesByBuffer.set(view.buffer, ranges);
    nextRow += view.rowCount;
    nextByte += view.byteLength;
  }
  if (nextRow !== rows || nextByte !== Number(tensor.logicalBytes)) {
    fail("full-attention-weight-views-invalid", "Qwen3.5 full-attention matrix rows are incomplete");
  }
}

function requireDirectF32Tensor(
  weights: Qwen35WeightDirectoryView,
  name: string,
  shape: readonly number[],
): Qwen35TensorWeightView {
  const tensor = requireWeight(weights, name, shape);
  const elements = shape.reduce((value, next) => value * next, 1);
  const bytes = elements * 4;
  const rowCount = shape.slice(1).reduce((value, next) => value * next, 1);
  const view = tensor.physicalRows[0];
  if (
    tensor.ggmlType !== GgmlType.F32 || tensor.storageType !== "f32" ||
    tensor.rowBytes !== shape[0]! * 4 || tensor.rowCount !== rowCount ||
    tensor.logicalBytes !== BigInt(bytes) || tensor.physicalRows.length !== 1 ||
    view === undefined || view.firstRow !== 0 || view.rowCount !== rowCount ||
    view.tensorByteOffset !== 0 || view.byteLength !== bytes ||
    !Number.isSafeInteger(view.bufferByteOffset) || view.bufferByteOffset < 0
  ) {
    fail("full-attention-direct-f32-invalid", "A direct-read Qwen3.5 full-attention tensor must use one F32 range");
  }
  return tensor;
}

function fullAttentionGeometryData(
  input: PlanQwen35FullAttentionLayerGeometryInput,
): FullAttentionGeometryData {
  const sequence = requireLayerSequence(input.program, input.invocation);
  const names = input.invocation.tensors;
  const matrixWeights = Object.freeze({
    query: requireWeight(input.weights, names.query, [HIDDEN, QUERY_GATE]),
    key: requireWeight(input.weights, names.key, [HIDDEN, KEY_VALUE]),
    value: requireWeight(input.weights, names.value, [HIDDEN, KEY_VALUE]),
    output: requireWeight(input.weights, names.output, [ATTENTION_OUTPUT, HIDDEN]),
    ffnGate: requireWeight(input.weights, sequence.ffnGateWeight, [HIDDEN, FFN]),
    ffnUp: requireWeight(input.weights, sequence.ffnUpWeight, [HIDDEN, FFN]),
    ffnDown: requireWeight(input.weights, sequence.ffnDownWeight, [FFN, HIDDEN]),
  });
  for (const tensor of Object.values(matrixWeights)) requirePhysicalMatrixRows(tensor);
  const directWeights = Object.freeze({
    inputNorm: requireDirectF32Tensor(input.weights, sequence.inputNormWeight, [HIDDEN]),
    queryNorm: requireDirectF32Tensor(input.weights, names.queryNorm, [256]),
    keyNorm: requireDirectF32Tensor(input.weights, names.keyNorm, [256]),
    postNorm: requireDirectF32Tensor(input.weights, sequence.postAttentionNormWeight, [HIDDEN]),
  });
  const physicalGemvPieceCount = Object.values(matrixWeights).reduce(
    (sum, tensor) => sum + tensor.physicalRows.length,
    0,
  );
  const geometry = Object.freeze({
    layer: input.invocation.layer,
    fixedUniformCount: (NON_ONLINE_FIXED_UNIFORM_COUNT + MAX_KV_PAGE_COUNT) as 70,
    physicalGemvPieceCount,
    uniformCount:
      physicalGemvPieceCount + NON_ONLINE_FIXED_UNIFORM_COUNT +
      MAX_KV_PAGE_COUNT,
  });
  return Object.freeze({ sequence, matrixWeights, directWeights, geometry });
}

/** Derives exact uniform capacity without activation, state, or uniform buffers. */
export function planQwen35FullAttentionLayerGeometry(
  input: PlanQwen35FullAttentionLayerGeometryInput,
): Qwen35FullAttentionLayerGeometry {
  return fullAttentionGeometryData(input).geometry;
}

function makeBinding(
  index: number,
  kind: "storage" | "uniform",
  slice: Qwen35ForwardBufferSlice,
  requiredBytes: number,
  limits: Qwen35ForwardDeviceLimits,
): Qwen35BufferBinding {
  const alignment = kind === "storage"
    ? limits.minStorageBufferOffsetAlignment
    : limits.minUniformBufferOffsetAlignment;
  const maximum = kind === "storage"
    ? limits.maxStorageBufferBindingSize
    : limits.maxUniformBufferBindingSize;
  if (
    !Number.isSafeInteger(index) || index < 0 ||
    !Number.isSafeInteger(slice.offset) || slice.offset < 0 ||
    !Number.isSafeInteger(slice.byteLength) || slice.byteLength < requiredBytes ||
    !Number.isSafeInteger(requiredBytes) || requiredBytes < 4 || requiredBytes % 4 !== 0 ||
    !Number.isSafeInteger(alignment) || alignment < 1 || slice.offset % alignment !== 0 ||
    !Number.isSafeInteger(maximum) || maximum < requiredBytes ||
    !Number.isSafeInteger(slice.offset + requiredBytes)
  ) {
    fail("full-attention-binding-invalid", "A Qwen3.5 full-attention binding is invalid");
  }
  return Object.freeze({ binding: index, kind, buffer: slice.buffer, offset: slice.offset, size: requiredBytes });
}

function requireDirectF32Weight(
  tensor: Qwen35TensorWeightView,
  limits: Qwen35ForwardDeviceLimits,
): { readonly tensor: Qwen35TensorWeightView; readonly binding: Qwen35BufferBinding } {
  const bytes = Number(tensor.logicalBytes);
  const view = tensor.physicalRows[0];
  if (view === undefined) fail("full-attention-direct-f32-invalid", "A direct-read Qwen3.5 full-attention tensor is unavailable");
  return Object.freeze({
    tensor,
    binding: makeBinding(0, "storage", {
      buffer: view.buffer,
      offset: view.bufferByteOffset,
      byteLength: view.byteLength,
    }, bytes, limits),
  });
}

function workspaceSlice(
  workspace: PlanQwen35FullAttentionLayerDispatchInput["workspace"],
  kind: Qwen35ActivationResourceKind,
  requiredBytes: number,
): Qwen35ForwardBufferSlice {
  let view: Qwen35ActivationResourceView | undefined;
  // The workspace callback is caller-owned, so its thrown text cannot cross this boundary.
  try {
    view = workspace.get(kind);
  } catch {
    fail("full-attention-workspace-invalid", "A Qwen3.5 full-attention activation view is invalid");
  }
  if (
    view === undefined || view.kind !== kind || view.scalarType !== "f32" ||
    view.bytes !== BigInt(view.byteLength) || view.elementCount * 4 !== view.byteLength ||
    view.binding.offset !== 0 || view.binding.size !== view.byteLength ||
    view.byteLength < requiredBytes
  ) {
    fail("full-attention-workspace-invalid", "A Qwen3.5 full-attention activation view is invalid");
  }
  return Object.freeze({ buffer: view.binding.buffer, offset: 0, byteLength: view.byteLength });
}

interface Qwen35AttentionStatePage {
  readonly tokenStart: number;
  readonly tokenCapacity: number;
  readonly slice: Qwen35ForwardBufferSlice;
}

function statePages(
  resource: Extract<Qwen35HybridLayerResources, { kind: "full-attention" }>["key"],
  logicalCapacity: number,
  limits: Qwen35ForwardDeviceLimits,
): readonly Qwen35AttentionStatePage[] {
  if (
    !Number.isSafeInteger(logicalCapacity) || logicalCapacity < 1 ||
    logicalCapacity > 16_384 || resource.bytes !== resource.byteLength ||
    resource.byteLength <= 0n || resource.byteLength % BigInt(KV_ROW_BYTES) !== 0n
  ) {
    fail("full-attention-state-invalid", "A Qwen3.5 full-attention state view is invalid");
  }
  let expectedOffset = 0n;
  const pages = resource.shards.map((shard) => {
    if (
      shard.logicalByteOffset !== expectedOffset ||
      shard.logicalByteLength <= 0n ||
      shard.logicalByteLength % BigInt(KV_ROW_BYTES) !== 0n ||
      shard.allocatedByteLength < shard.logicalByteLength ||
      shard.logicalByteLength > BigInt(limits.maxStorageBufferBindingSize) ||
      shard.logicalByteOffset > BigInt(Number.MAX_SAFE_INTEGER) ||
      shard.logicalByteLength > BigInt(Number.MAX_SAFE_INTEGER)
    ) {
      fail("full-attention-state-invalid", "A Qwen3.5 full-attention state page is invalid");
    }
    const tokenStart = Number(shard.logicalByteOffset / BigInt(KV_ROW_BYTES));
    const tokenCapacity = Number(shard.logicalByteLength / BigInt(KV_ROW_BYTES));
    const slice = Object.freeze({
      buffer: shard.buffer,
      offset: 0,
      byteLength: Number(shard.logicalByteLength),
    });
    makeBinding(0, "storage", slice, slice.byteLength, limits);
    expectedOffset += shard.logicalByteLength;
    return Object.freeze({ tokenStart, tokenCapacity, slice });
  });
  if (
    pages.length < 1 || pages.length > logicalCapacity ||
    expectedOffset !== resource.byteLength ||
    Number(expectedOffset / BigInt(KV_ROW_BYTES)) > logicalCapacity
  ) {
    fail("full-attention-state-invalid", "A Qwen3.5 full-attention state page directory is invalid");
  }
  return Object.freeze(pages);
}

function rangeOverlaps(left: BufferRange, right: BufferRange): boolean {
  return left.buffer === right.buffer &&
    left.offset < right.offset + right.size && right.offset < left.offset + left.size;
}

function requireDisjointRanges(ranges: readonly BufferRange[], code: string): void {
  for (let index = 0; index < ranges.length; index += 1) {
    for (let prior = 0; prior < index; prior += 1) {
      if (rangeOverlaps(ranges[index]!, ranges[prior]!)) {
        fail(code, "Qwen3.5 full-attention buffer ranges overlap");
      }
    }
  }
}

function kernelSource(kernel: { readonly id: string; readonly source: string }): Qwen35KernelSource {
  return Object.freeze({ id: kernel.id, source: kernel.source, entryPoint: "main" });
}

function primitiveKernel(operation: QwenPrimitiveOperation): Qwen35KernelSource {
  const kernel = QWEN_PRIMITIVE_KERNELS.find((candidate) => candidate.operation === operation);
  if (kernel === undefined) fail("full-attention-kernel-missing", "A Qwen3.5 primitive kernel is missing");
  return kernelSource(kernel);
}

function hybridKernel(operation: Qwen35HybridOperation): Qwen35KernelSource {
  const kernel = QWEN35_HYBRID_KERNELS.find((candidate) => candidate.key.operation === operation);
  if (kernel === undefined) fail("full-attention-kernel-missing", "A Qwen3.5 hybrid kernel is missing");
  return kernelSource(kernel);
}

function frozenCommand(input: {
  readonly stage: Qwen35FullAttentionLayerStage;
  readonly kernel: Qwen35KernelSource;
  readonly bindings: readonly Qwen35BufferBinding[];
  readonly workgroups: Qwen35DispatchRequest["workgroups"];
  readonly uniformWords?: readonly number[];
  readonly mutatesPersistentState?: boolean;
}): Qwen35FullAttentionLayerCommand {
  return Object.freeze({
    stage: input.stage,
    kernel: Object.freeze({ ...input.kernel }),
    bindings: Object.freeze(input.bindings.map((binding) => Object.freeze({ ...binding }))),
    workgroups: Object.freeze({ ...input.workgroups }),
    uniformWords: Object.freeze([...(input.uniformWords ?? [])]),
    mutatesPersistentState: input.mutatesPersistentState ?? false,
  });
}

function f32Word(value: number): number {
  const bytes = new ArrayBuffer(4);
  const view = new DataView(bytes);
  view.setFloat32(0, value, true);
  return view.getUint32(0, true);
}

/** Builds one fixed decode-layer schedule; it never submits or advances state. */
export function planQwen35FullAttentionLayerDispatch(
  input: PlanQwen35FullAttentionLayerDispatchInput,
): Qwen35FullAttentionLayerDispatchPlan {
  const geometryData = fullAttentionGeometryData(input);
  const { sequence, matrixWeights } = geometryData;
  if (input.uniforms.length > geometryData.geometry.uniformCount) {
    fail("full-attention-uniform-count-invalid", "Qwen3.5 full-attention uniform count is invalid");
  }
  if (
    input.state.kind !== "full-attention" || input.state.layer !== input.invocation.layer ||
    input.state.key.layer !== input.invocation.layer || input.state.key.kind !== "key" ||
    input.state.value.layer !== input.invocation.layer || input.state.value.kind !== "value"
  ) {
    fail("full-attention-state-invalid", "The Qwen3.5 layer requires full-attention state");
  }

  let preparePlan: ReturnType<typeof planQwen35HybridDispatch>;
  try {
    if (
      !Number.isSafeInteger(input.position) || input.position < 0 ||
      !Number.isSafeInteger(input.capacity) || input.capacity < 1 ||
      input.capacity > 16_384 || input.position >= input.capacity
    ) {
      throw new Error("invalid logical context address");
    }
  } catch {
    fail("full-attention-context-invalid", "The Qwen3.5 full-attention context address is invalid");
  }

  const keyPages = statePages(input.state.key, input.capacity, input.limits);
  const valuePages = statePages(input.state.value, input.capacity, input.limits);
  if (
    keyPages.length !== valuePages.length ||
    keyPages.some((page, index) => {
      const value = valuePages[index];
      return value === undefined || page.tokenStart !== value.tokenStart ||
        page.tokenCapacity !== value.tokenCapacity;
    })
  ) {
    fail("full-attention-state-invalid", "Qwen3.5 K/V state pages do not match");
  }
  const activePages = keyPages.filter(({ tokenStart }) => tokenStart <= input.position);
  const activeValuePages = valuePages.slice(0, activePages.length);
  const targetPageIndex = activePages.findIndex(
    ({ tokenStart, tokenCapacity }) =>
      input.position >= tokenStart && input.position < tokenStart + tokenCapacity,
  );
  if (
    targetPageIndex < 0 || activePages.length > MAX_KV_PAGE_COUNT ||
    activePages.length !== targetPageIndex + 1
  ) {
    fail("full-attention-state-invalid", "The current token has no resident Qwen3.5 K/V page");
  }
  const targetKeyPage = activePages[targetPageIndex]!;
  const targetValuePage = activeValuePages[targetPageIndex]!;
  const localPosition = input.position - targetKeyPage.tokenStart;
  try {
    preparePlan = planQwen35HybridDispatch({
      operation: "full-attention-prepare",
      maxComputeWorkgroupsPerDimension: input.limits.maxComputeWorkgroupsPerDimension,
      position: localPosition,
      capacity: targetKeyPage.tokenCapacity,
      positions: input.mropePositions,
    });
  } catch {
    fail("full-attention-context-invalid", "The Qwen3.5 full-attention page context address is invalid");
  }
  const requiredUniformCount = geometryData.geometry.physicalGemvPieceCount +
    NON_ONLINE_FIXED_UNIFORM_COUNT + activePages.length;
  if (input.uniforms.length < requiredUniformCount) {
    fail("full-attention-uniform-count-invalid", "Qwen3.5 full-attention uniforms are incomplete");
  }

  const direct = {
    inputNorm: requireDirectF32Weight(geometryData.directWeights.inputNorm, input.limits),
    queryNorm: requireDirectF32Weight(geometryData.directWeights.queryNorm, input.limits),
    keyNorm: requireDirectF32Weight(geometryData.directWeights.keyNorm, input.limits),
    postNorm: requireDirectF32Weight(geometryData.directWeights.postNorm, input.limits),
  };
  const activations = {
    hidden: workspaceSlice(input.workspace, "packed-embedding-output", HIDDEN * 4),
    hiddenSecondary: workspaceSlice(input.workspace, "hidden-secondary", HIDDEN * 4),
    normalized: workspaceSlice(input.workspace, "normalized-hidden", HIDDEN * 4),
    queryGate: workspaceSlice(input.workspace, "attention-projection-primary", QUERY_GATE * 4),
    preparedQueryGate: workspaceSlice(input.workspace, "attention-projection-secondary", QUERY_GATE * 4),
    key: workspaceSlice(input.workspace, "full-attention-key", KEY_VALUE * 4),
    value: workspaceSlice(input.workspace, "full-attention-value", KEY_VALUE * 4),
    attention: workspaceSlice(input.workspace, "attention-inner-primary", ATTENTION_OUTPUT * 4),
    ffnGate: workspaceSlice(input.workspace, "ffn-gate", FFN * 4),
    ffnUp: workspaceSlice(input.workspace, "ffn-up", FFN * 4),
    ffnProduct: workspaceSlice(input.workspace, "ffn-product", FFN * 4),
  };
  const mutableRanges: BufferRange[] = Object.values(activations).map((slice) => ({
    buffer: slice.buffer, offset: slice.offset, size: slice.byteLength,
  }));
  mutableRanges.push(
    ...activePages.map(({ slice }) => ({
      buffer: slice.buffer, offset: slice.offset, size: slice.byteLength,
    })),
    ...activeValuePages.map(({ slice }) => ({
      buffer: slice.buffer, offset: slice.offset, size: slice.byteLength,
    })),
  );
  requireDisjointRanges(mutableRanges, "full-attention-buffer-alias-invalid");

  const requiredWeights = [
    ...Object.values(matrixWeights),
    ...Object.values(direct).map(({ tensor }) => tensor),
  ];
  const weightRanges = requiredWeights.flatMap((tensor) => tensor.physicalRows.map((view) => ({
    buffer: view.buffer, offset: view.bufferByteOffset, size: view.byteLength,
  })));
  requireDisjointRanges(weightRanges, "full-attention-weight-alias-invalid");
  for (const mutable of mutableRanges) {
    if (weightRanges.some((weight) => rangeOverlaps(mutable, weight))) {
      fail("full-attention-buffer-alias-invalid", "Qwen3.5 state or activation aliases model weights");
    }
  }

  const commands: Qwen35FullAttentionLayerCommand[] = [];
  const usedUniforms: Qwen35BufferBinding[] = [];
  let uniformCursor = 0;
  const nextUniform = (bindingIndex: number, requiredBytes: number): Qwen35ForwardBufferSlice => {
    const slice = input.uniforms[uniformCursor];
    if (slice === undefined) fail("full-attention-uniform-count-invalid", "Qwen3.5 full-attention uniforms are incomplete");
    usedUniforms.push(makeBinding(bindingIndex, "uniform", slice, requiredBytes, input.limits));
    uniformCursor += 1;
    return slice;
  };
  const storage = (index: number, slice: Qwen35ForwardBufferSlice, bytes: number) =>
    makeBinding(index, "storage", slice, bytes, input.limits);
  const uniformBinding = (index: number, slice: Qwen35ForwardBufferSlice, bytes: number) =>
    makeBinding(index, "uniform", slice, bytes, input.limits);
  const addPrimitive = (
    stage: Qwen35FullAttentionLayerStage,
    operation: "rms-norm" | "residual-add" | "swiglu",
    slices: readonly Qwen35ForwardBufferSlice[],
    elements: number,
    words: readonly number[],
    weight?: Qwen35BufferBinding,
  ): void => {
    const slot = nextUniform(3, 16);
    const plan = planPrimitiveDispatch({ operation, elementCount: elements });
    if (plan.workgroups.x > input.limits.maxComputeWorkgroupsPerDimension) {
      fail("full-attention-dispatch-invalid", "A Qwen3.5 primitive exceeds device limits");
    }
    const bindings = operation === "rms-norm"
      ? [
          storage(0, slices[0]!, elements * 4),
          Object.freeze({ ...weight!, binding: 1 }),
          storage(2, slices[1]!, elements * 4),
          uniformBinding(3, slot, 16),
        ]
      : [
          storage(0, slices[0]!, elements * 4),
          storage(1, slices[1]!, elements * 4),
          storage(2, slices[2]!, elements * 4),
          uniformBinding(3, slot, 16),
        ];
    requireOutputDisjoint(bindings[2]!, bindings.filter((_, index) => index !== 2));
    commands.push(frozenCommand({ stage, kernel: primitiveKernel(operation), bindings, workgroups: plan.workgroups, uniformWords: words }));
  };
  const addGemv = (
    stage: Qwen35FullAttentionLayerStage,
    tensor: Qwen35TensorWeightView,
    activation: Qwen35ForwardBufferSlice,
    output: Qwen35ForwardBufferSlice,
  ): void => {
    const slots = tensor.physicalRows.map(() => nextUniform(3, 20));
    const plans = planQwen35PackedGemvDispatches({
      weights: input.weights, tensorName: tensor.name, activation, output, uniforms: slots, limits: input.limits,
    });
    commands.push(...plans.map((plan) => gemvCommand(stage, plan)));
  };

  addPrimitive("input-rms", "rms-norm", [activations.hidden, activations.normalized], HIDDEN,
    [HIDDEN, HIDDEN, f32Word(sequence.epsilon), 0], direct.inputNorm.binding);
  addGemv("query-projection", matrixWeights.query, activations.normalized, activations.queryGate);
  addGemv("key-projection", matrixWeights.key, activations.normalized, activations.key);
  addGemv("value-projection", matrixWeights.value, activations.normalized, activations.value);

  const prepareUniform = nextUniform(8, 32);
  const prepareBindings = [
    storage(0, activations.queryGate, QUERY_GATE * 4),
    storage(1, activations.key, KEY_VALUE * 4),
    storage(2, activations.value, KEY_VALUE * 4),
    Object.freeze({ ...direct.queryNorm.binding, binding: 3 }),
    Object.freeze({ ...direct.keyNorm.binding, binding: 4 }),
    storage(5, activations.preparedQueryGate, QUERY_GATE * 4),
    storage(6, targetKeyPage.slice, targetKeyPage.slice.byteLength),
    storage(7, targetValuePage.slice, targetValuePage.slice.byteLength),
    uniformBinding(8, prepareUniform, 32),
  ];
  requireOutputDisjoint(prepareBindings[5]!, prepareBindings.slice(0, 5));
  commands.push(frozenCommand({
    stage: "full-attention-prepare",
    kernel: hybridKernel("full-attention-prepare"),
    bindings: prepareBindings,
    workgroups: preparePlan.workgroups,
    uniformWords: [
      localPosition,
      targetKeyPage.tokenCapacity,
      ...input.mropePositions,
      0,
      0,
      0,
    ],
    mutatesPersistentState: true,
  }));

  // Each dispatch binds one stable KV page. The shader carries the online
  // softmax accumulator through activation storage, so no score matrix or
  // monolithic 16K K/V binding is required.
  for (const [pageIndex, keyPage] of activePages.entries()) {
    const valuePage = activeValuePages[pageIndex]!;
    const tokenCount = pageIndex === targetPageIndex
      ? localPosition + 1
      : keyPage.tokenCapacity;
    let onlinePlan: ReturnType<typeof planQwen35HybridDispatch>;
    try {
      onlinePlan = planQwen35HybridDispatch({
        operation: "full-attention-online",
        maxComputeWorkgroupsPerDimension: input.limits.maxComputeWorkgroupsPerDimension,
        position: tokenCount - 1,
        capacity: keyPage.tokenCapacity,
        tokenCount,
      });
    } catch {
      fail("full-attention-context-invalid", "The Qwen3.5 full-attention page range is invalid");
    }
    const onlineUniform = nextUniform(5, 16);
    const onlineBindings = [
      storage(0, activations.preparedQueryGate, QUERY_GATE * 4),
      storage(1, keyPage.slice, keyPage.slice.byteLength),
      storage(2, valuePage.slice, valuePage.slice.byteLength),
      storage(3, activations.attention, ATTENTION_OUTPUT * 4),
      storage(4, activations.key, 16 * 2 * 4),
      uniformBinding(5, onlineUniform, 16),
    ];
    requireOutputDisjoint(
      onlineBindings[3]!,
      onlineBindings.filter((_, index) => index !== 3),
    );
    commands.push(frozenCommand({
      stage: "full-attention-online",
      kernel: hybridKernel("full-attention-online"),
      bindings: onlineBindings,
      workgroups: onlinePlan.workgroups,
      uniformWords: [tokenCount, pageIndex, activePages.length, 0],
    }));
  }

  addGemv("attention-output-projection", matrixWeights.output, activations.attention, activations.normalized);
  addPrimitive("attention-residual", "residual-add", [activations.normalized, activations.hidden, activations.hiddenSecondary], HIDDEN, [HIDDEN, 0, 0, 0]);
  addPrimitive("post-attention-rms", "rms-norm", [activations.hiddenSecondary, activations.normalized], HIDDEN,
    [HIDDEN, HIDDEN, f32Word(sequence.epsilon), 0], direct.postNorm.binding);
  addGemv("ffn-gate-projection", matrixWeights.ffnGate, activations.normalized, activations.ffnGate);
  addGemv("ffn-up-projection", matrixWeights.ffnUp, activations.normalized, activations.ffnUp);
  addPrimitive("swiglu", "swiglu", [activations.ffnGate, activations.ffnUp, activations.ffnProduct], FFN, [FFN, 0, 0, 0]);
  addGemv("ffn-down-projection", matrixWeights.ffnDown, activations.ffnProduct, activations.normalized);
  addPrimitive("mlp-residual", "residual-add", [activations.normalized, activations.hiddenSecondary, activations.hidden], HIDDEN, [HIDDEN, 0, 0, 0]);

  if (uniformCursor !== requiredUniformCount) {
    fail("full-attention-uniform-count-invalid", "Qwen3.5 full-attention uniform count is invalid");
  }
  requireDisjointRanges(usedUniforms, "full-attention-uniform-alias-invalid");
  for (const uniform of usedUniforms) {
    if (mutableRanges.some((range) => rangeOverlaps(uniform, range)) ||
        weightRanges.some((range) => rangeOverlaps(uniform, range))) {
      fail("full-attention-buffer-alias-invalid", "Qwen3.5 uniforms alias a storage range");
    }
  }

  return Object.freeze({
    layer: input.invocation.layer,
    position: input.position,
    commands: Object.freeze(commands),
    uniformCount: uniformCursor,
    stateSemantics: Object.freeze({
      mutatingStages: Object.freeze(["full-attention-prepare"] as const),
      failureAfterSubmission: "persistent-state-indeterminate-dispose-required",
      advancePositionAfter: "successful-queue-retirement",
    }),
  });
}

function requireOutputDisjoint(output: Qwen35BufferBinding, reads: readonly Qwen35BufferBinding[]): void {
  if (reads.some((read) => rangeOverlaps(output, read))) {
    fail("full-attention-buffer-alias-invalid", "A Qwen3.5 full-attention output aliases an input");
  }
}

function gemvCommand(
  stage: Qwen35FullAttentionLayerStage,
  plan: Qwen35ForwardDispatchPlan,
): Qwen35FullAttentionLayerCommand {
  return frozenCommand({ stage, kernel: plan.kernel, bindings: plan.bindings, workgroups: plan.workgroups, uniformWords: plan.uniformWords });
}
