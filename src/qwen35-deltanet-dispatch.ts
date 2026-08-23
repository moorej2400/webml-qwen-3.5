import { diagnosticError } from "./diagnostics.js";
import { GgmlType } from "./gguf.js";
import {
  QWEN35_HYBRID_KERNELS,
  planQwen35HybridDispatch,
  type Qwen35HybridOperation,
} from "./hybrid-kernels.js";
import type { Qwen35HybridLayerResources } from "./hybrid-state.js";
import {
  QWEN_PRIMITIVE_KERNELS,
  planPrimitiveDispatch,
  type QwenPrimitiveOperation,
} from "./qwen-primitives.js";
import type {
  Qwen35ActivationResourceKind,
  Qwen35ActivationResourceView,
  Qwen35ActivationWorkspace,
  Qwen35DeltaNetParameterLiveness,
} from "./qwen35-activation-workspace.js";
import {
  planQwen35PackedGemvDispatches,
  planQwen35TwinF32GemvDispatch,
  planQwen35TwinQ3GemvDispatches,
  type Qwen35ForwardBufferSlice,
  type Qwen35ForwardDeviceLimits,
  type Qwen35ForwardDispatchPlan,
} from "./qwen35-forward-dispatch.js";
import type {
  Qwen35Invocation,
  Qwen35Program,
} from "./qwen35-program.js";
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
const INNER = 4_096;
const QKV = 8_192;
const FFN = 9_216;
const HEADS = 32;
const PARAMETER_BYTES = HEADS * 4;
const CONV_BYTES = 8_192 * 4 * 4;
const RECURRENT_BYTES = 32 * 128 * 128 * 4;

export type Qwen35DeltaNetLayerStage =
  | "input-rms"
  | "attention-gate-projection"
  | "attention-qkv-projection"
  | "deltanet-alpha-projection"
  | "deltanet-beta-projection"
  | "deltanet-alpha-beta-projection"
  | "deltanet-parameters"
  | "deltanet-conv"
  | "deltanet-recurrent"
  | "deltanet-gated-norm"
  | "deltanet-recurrent-gated-norm"
  | "deltanet-output-projection"
  | "attention-residual"
  | "post-attention-rms"
  | "ffn-gate-projection"
  | "ffn-up-projection"
  | "swiglu"
  | "ffn-down-projection"
  | "mlp-residual";

export interface Qwen35DeltaNetLayerCommand extends Qwen35DispatchRequest {
  readonly stage: Qwen35DeltaNetLayerStage;
  /** Caller writes these bit-exact u32 words into this command's uniform slot. */
  readonly uniformWords: readonly number[];
  readonly mutatesPersistentState: boolean;
}

export interface Qwen35DeltaNetLayerDispatchPlan {
  readonly layer: number;
  readonly commands: readonly Qwen35DeltaNetLayerCommand[];
  readonly uniformCount: number;
  /**
   * Conv and recurrent kernels mutate state in place. Once submitted, any
   * failure makes that state indeterminate until the owning session disposes it.
   */
  readonly stateSemantics: Readonly<{
    readonly mutatingStages: readonly ["deltanet-conv", "deltanet-recurrent-gated-norm"];
    readonly failureAfterSubmission: "persistent-state-indeterminate-dispose-required";
    readonly advancePositionAfter: "successful-queue-retirement";
  }>;
}

export interface Qwen35DeltaNetLayerGeometry {
  readonly fixedUniformCount: 4;
  readonly physicalGemvPieceCount: number;
  readonly uniformCount: number;
}

export interface PlanQwen35DeltaNetLayerDispatchInput {
  readonly program: Qwen35Program;
  readonly invocation: Extract<Qwen35Invocation, { kind: "gated-deltanet" }>;
  readonly weights: Qwen35WeightDirectoryView;
  readonly workspace: Pick<Qwen35ActivationWorkspace, "get">;
  readonly deltanetParameterLiveness: Qwen35DeltaNetParameterLiveness;
  readonly state: Qwen35HybridLayerResources;
  readonly limits: Qwen35ForwardDeviceLimits;
  readonly uniforms: readonly Qwen35ForwardBufferSlice[];
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
  invocation: PlanQwen35DeltaNetLayerDispatchInput["invocation"],
): LayerSequence {
  if (
    program.model !== "qwen35-4b" ||
    !isRunnableProgram(program) ||
    !Number.isSafeInteger(invocation.layer) ||
    invocation.layer < 0 ||
    program.invocations.indexOf(invocation) < 1 ||
    invocation.stateLayout !== "fp32-recurrent-state" ||
    invocation.runnable !== true ||
    !sameValues(invocation.kernels, [
      "deltanet-conv",
      "deltanet-parameters",
      "deltanet-recurrent",
      "deltanet-gated-norm",
    ])
  ) {
    fail(
      "deltanet-program-invalid",
      "The Qwen3.5 DeltaNet program invocation is invalid",
    );
  }
  const prefix = `blk.${invocation.layer}`;
  const expectedTensors = {
    gate: `${prefix}.attn_gate.weight`,
    qkv: `${prefix}.attn_qkv.weight`,
    a: `${prefix}.ssm_a`,
    alpha: `${prefix}.ssm_alpha.weight`,
    beta: `${prefix}.ssm_beta.weight`,
    convolution: `${prefix}.ssm_conv1d.weight`,
    timeStepBias: `${prefix}.ssm_dt.bias`,
    norm: `${prefix}.ssm_norm.weight`,
    output: `${prefix}.ssm_out.weight`,
  };
  if (Object.entries(expectedTensors).some(
    ([role, name]) => invocation.tensors[role as keyof typeof expectedTensors] !== name,
  )) {
    fail(
      "deltanet-program-invalid",
      "The Qwen3.5 DeltaNet tensor names are invalid",
    );
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
    ffnGate?.kind !== "gemv" ||
    ffnGate.layer !== invocation.layer ||
    ffnGate.projection !== "ffn-gate" ||
    ffnGate.weight !== `${prefix}.ffn_gate.weight` ||
    ffnGate.rows !== FFN ||
    ffnGate.columns !== HIDDEN ||
    ffnUp?.kind !== "gemv" ||
    ffnUp.layer !== invocation.layer ||
    ffnUp.projection !== "ffn-up" ||
    ffnUp.weight !== `${prefix}.ffn_up.weight` ||
    ffnUp.rows !== FFN ||
    ffnUp.columns !== HIDDEN ||
    swiglu?.kind !== "swiglu" ||
    swiglu.layer !== invocation.layer ||
    swiglu.elements !== FFN ||
    ffnDown?.kind !== "gemv" ||
    ffnDown.layer !== invocation.layer ||
    ffnDown.projection !== "ffn-down" ||
    ffnDown.weight !== `${prefix}.ffn_down.weight` ||
    ffnDown.rows !== HIDDEN ||
    ffnDown.columns !== FFN ||
    mlpResidual?.kind !== "residual-add" ||
    mlpResidual.layer !== invocation.layer ||
    mlpResidual.site !== "mlp" ||
    !Number.isFinite(inputNorm.epsilon) ||
    inputNorm.epsilon < 0 ||
    postNorm.epsilon !== inputNorm.epsilon
  ) {
    fail(
      "deltanet-program-invalid",
      "The Qwen3.5 DeltaNet layer sequence is invalid",
    );
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
  return actual.length === expected.length &&
    expected.every((value, index) => actual[index] === value);
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
    fail("deltanet-weight-invalid", "A Qwen3.5 DeltaNet weight is invalid");
  }
  if (tensor === undefined || tensor.name !== name || !sameShape(tensor.shape, shape)) {
    fail("deltanet-weight-invalid", "A Qwen3.5 DeltaNet weight is invalid");
  }
  return tensor;
}

function requireDirectF32Weight(
  weights: Qwen35WeightDirectoryView,
  name: string,
  shape: readonly number[],
  limits: Qwen35ForwardDeviceLimits,
): { readonly tensor: Qwen35TensorWeightView; readonly binding: Qwen35BufferBinding } {
  const tensor = requireWeight(weights, name, shape);
  const elementCount = shape.reduce((value, next) => value * next, 1);
  const bytes = elementCount * 4;
  const rowCount = shape.slice(1).reduce((value, next) => value * next, 1);
  const view = tensor.physicalRows[0];
  if (
    tensor.ggmlType !== GgmlType.F32 ||
    tensor.storageType !== "f32" ||
    tensor.rowBytes !== shape[0]! * 4 ||
    tensor.rowCount !== rowCount ||
    tensor.logicalBytes !== BigInt(bytes) ||
    tensor.physicalRows.length !== 1 ||
    view === undefined ||
    view.firstRow !== 0 ||
    view.rowCount !== rowCount ||
    view.tensorByteOffset !== 0 ||
    view.byteLength !== bytes
  ) {
    fail(
      "deltanet-direct-f32-invalid",
      "A direct-read Qwen3.5 DeltaNet tensor must use one F32 range",
    );
  }
  return Object.freeze({
    tensor,
    binding: makeBinding(0, "storage", {
      buffer: view.buffer,
      offset: view.bufferByteOffset,
      byteLength: view.byteLength,
    }, bytes, limits),
  });
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
    fail("deltanet-binding-invalid", "A Qwen3.5 DeltaNet binding is invalid");
  }
  return Object.freeze({
    binding: index,
    kind,
    buffer: slice.buffer,
    offset: slice.offset,
    size: requiredBytes,
  });
}

function rangeOverlaps(left: BufferRange, right: BufferRange): boolean {
  return left.buffer === right.buffer &&
    left.offset < right.offset + right.size &&
    right.offset < left.offset + left.size;
}

function requireDisjointRanges(ranges: readonly BufferRange[], code: string): void {
  for (let index = 0; index < ranges.length; index += 1) {
    for (let prior = 0; prior < index; prior += 1) {
      if (rangeOverlaps(ranges[index]!, ranges[prior]!)) {
        fail(code, "Qwen3.5 DeltaNet buffer ranges overlap");
      }
    }
  }
}

function workspaceSlice(
  workspace: PlanQwen35DeltaNetLayerDispatchInput["workspace"],
  kind: Qwen35ActivationResourceKind,
  requiredBytes: number,
  scalarType: "f16" | "f32" = "f32",
): Qwen35ForwardBufferSlice {
  let view: Qwen35ActivationResourceView | undefined;
  // The workspace callback is caller-owned, so its thrown text cannot cross this boundary.
  try {
    view = workspace.get(kind);
  } catch {
    fail("deltanet-workspace-invalid", "A Qwen3.5 activation view is invalid");
  }
  if (
    view === undefined ||
    view.kind !== kind ||
    view.scalarType !== scalarType ||
    view.bytes !== BigInt(view.byteLength) ||
    view.elementCount * (scalarType === "f16" ? 2 : 4) !== view.byteLength ||
    view.binding.offset !== 0 ||
    view.binding.size !== view.byteLength ||
    view.byteLength < requiredBytes
  ) {
    fail("deltanet-workspace-invalid", "A Qwen3.5 activation view is invalid");
  }
  return Object.freeze({
    buffer: view.binding.buffer,
    offset: view.binding.offset,
    byteLength: view.byteLength,
  });
}

function requireLiveness(liveness: Qwen35DeltaNetParameterLiveness): void {
  const expected = {
    rawAlpha: ["deltanet-alpha", "read"],
    rawBeta: ["deltanet-beta", "read"],
    transformedBeta: ["full-attention-key", "write"],
    decay: ["full-attention-value", "write"],
  } as const;
  if (
    liveness.activeLayerKind !== "gated-deltanet" ||
    !sameValues(liveness.borrowedFullAttentionResources, [
      "full-attention-key",
      "full-attention-value",
    ]) ||
    Object.entries(expected).some(([role, [resource, access]]) => {
      const range = liveness.ranges[role as keyof typeof expected];
      return range.resource !== resource || range.access !== access ||
        range.byteOffset !== 0 || range.byteLength !== PARAMETER_BYTES;
    })
  ) {
    fail(
      "deltanet-liveness-invalid",
      "The Qwen3.5 DeltaNet parameter liveness map is invalid",
    );
  }
}

function stateSlice(
  resource: Extract<Qwen35HybridLayerResources, { kind: "gated-deltanet" }>["conv"],
  expectedBytes: number,
  limits: Qwen35ForwardDeviceLimits,
): Qwen35ForwardBufferSlice {
  const shard = resource.shards[0];
  if (
    resource.bytes !== BigInt(expectedBytes) ||
    resource.byteLength !== BigInt(expectedBytes) ||
    resource.shards.length !== 1 ||
    shard === undefined ||
    shard.logicalByteOffset !== 0n ||
    shard.logicalByteLength !== BigInt(expectedBytes) ||
    shard.allocatedByteLength < BigInt(expectedBytes)
  ) {
    fail("deltanet-state-invalid", "A Qwen3.5 DeltaNet state view is invalid");
  }
  const slice = Object.freeze({ buffer: shard.buffer, offset: 0, byteLength: expectedBytes });
  makeBinding(0, "storage", slice, expectedBytes, limits);
  return slice;
}

function kernelSource(kernel: { readonly id: string; readonly source: string }): Qwen35KernelSource {
  return Object.freeze({ id: kernel.id, source: kernel.source, entryPoint: "main" });
}

function primitiveKernel(operation: QwenPrimitiveOperation): Qwen35KernelSource {
  const kernel = QWEN_PRIMITIVE_KERNELS.find((candidate) => candidate.operation === operation);
  if (kernel === undefined) fail("deltanet-kernel-missing", "A Qwen3.5 primitive kernel is missing");
  return kernelSource(kernel);
}

function hybridKernel(operation: Qwen35HybridOperation): Qwen35KernelSource {
  const kernel = QWEN35_HYBRID_KERNELS.find((candidate) => candidate.key.operation === operation);
  if (kernel === undefined) fail("deltanet-kernel-missing", "A Qwen3.5 hybrid kernel is missing");
  return kernelSource(kernel);
}

function frozenCommand(input: {
  readonly stage: Qwen35DeltaNetLayerStage;
  readonly kernel: Qwen35KernelSource;
  readonly bindings: readonly Qwen35BufferBinding[];
  readonly workgroups: Qwen35DispatchRequest["workgroups"];
  readonly uniformWords?: readonly number[];
  readonly mutatesPersistentState?: boolean;
}): Qwen35DeltaNetLayerCommand {
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
  new DataView(bytes).setFloat32(0, value, true);
  return new DataView(bytes).getUint32(0, true);
}

function canFuseTwinQ3(
  first: Qwen35TensorWeightView,
  second: Qwen35TensorWeightView,
): boolean {
  return first.ggmlType === GgmlType.Q3_K && second.ggmlType === GgmlType.Q3_K &&
    first.storageType === "q3-k-fused-f32-192" && second.storageType === "q3-k-fused-f32-192" &&
    first.shape.length === 2 && second.shape.length === 2 &&
    first.shape[0] === second.shape[0] && first.shape[1] === second.shape[1] &&
    first.shape[0]! % 512 === 0 && first.rowBytes === second.rowBytes &&
    first.physicalRows.length === 1 && second.physicalRows.length === 1;
}

function canFuseTwinF32(
  first: Qwen35TensorWeightView,
  second: Qwen35TensorWeightView,
): boolean {
  return first.ggmlType === GgmlType.F32 && second.ggmlType === GgmlType.F32 &&
    first.storageType === "f32" && second.storageType === "f32" &&
    first.shape.length === 2 && second.shape.length === 2 &&
    first.shape[0] === second.shape[0] && first.shape[1] === second.shape[1] &&
    first.shape[0]! % 32 === 0 &&
    first.physicalRows.length === 1 && second.physicalRows.length === 1;
}

function deltaNetGeometryData(input: {
  readonly program: Qwen35Program;
  readonly invocation: Extract<Qwen35Invocation, { kind: "gated-deltanet" }>;
  readonly weights: Qwen35WeightDirectoryView;
}): {
  readonly sequence: LayerSequence;
  readonly matrixWeights: Readonly<{
    readonly attentionGate: Qwen35TensorWeightView;
    readonly qkv: Qwen35TensorWeightView;
    readonly alpha: Qwen35TensorWeightView;
    readonly beta: Qwen35TensorWeightView;
    readonly output: Qwen35TensorWeightView;
    readonly ffnGate: Qwen35TensorWeightView;
    readonly ffnUp: Qwen35TensorWeightView;
    readonly ffnDown: Qwen35TensorWeightView;
  }>;
  readonly geometry: Qwen35DeltaNetLayerGeometry;
} {
  const sequence = requireLayerSequence(input.program, input.invocation);
  const tensorNames = input.invocation.tensors;
  const matrixWeights = Object.freeze({
    attentionGate: requireWeight(input.weights, tensorNames.gate, [HIDDEN, INNER]),
    qkv: requireWeight(input.weights, tensorNames.qkv, [HIDDEN, QKV]),
    alpha: requireWeight(input.weights, tensorNames.alpha, [HIDDEN, HEADS]),
    beta: requireWeight(input.weights, tensorNames.beta, [HIDDEN, HEADS]),
    output: requireWeight(input.weights, tensorNames.output, [INNER, HIDDEN]),
    ffnGate: requireWeight(input.weights, sequence.ffnGateWeight, [HIDDEN, FFN]),
    ffnUp: requireWeight(input.weights, sequence.ffnUpWeight, [HIDDEN, FFN]),
    ffnDown: requireWeight(input.weights, sequence.ffnDownWeight, [FFN, HIDDEN]),
  });
  const rawPhysicalGemvPieceCount = Object.values(matrixWeights).reduce(
    (count, tensor) => count + tensor.physicalRows.length,
    0,
  );
  const physicalGemvPieceCount = rawPhysicalGemvPieceCount -
    (canFuseTwinQ3(matrixWeights.ffnGate, matrixWeights.ffnUp) ? 2 : 0) -
    (canFuseTwinF32(matrixWeights.alpha, matrixWeights.beta) ? 1 : 0);
  if (
    !Number.isSafeInteger(rawPhysicalGemvPieceCount) ||
    rawPhysicalGemvPieceCount < Object.keys(matrixWeights).length
  ) {
    fail(
      "deltanet-weight-invalid",
      "A Qwen3.5 DeltaNet matrix has invalid physical row coverage",
    );
  }
  const geometry = Object.freeze({
    fixedUniformCount: 4 as const,
    physicalGemvPieceCount,
    uniformCount: physicalGemvPieceCount + 4,
  });
  return Object.freeze({ sequence, matrixWeights, geometry });
}

/** Derives exact uniform capacity from physical weights without GPU buffers. */
export function planQwen35DeltaNetLayerGeometry(input: {
  readonly program: Qwen35Program;
  readonly invocation: Extract<Qwen35Invocation, { kind: "gated-deltanet" }>;
  readonly weights: Qwen35WeightDirectoryView;
}): Qwen35DeltaNetLayerGeometry {
  return deltaNetGeometryData(input).geometry;
}

/** Builds the fixed decode schedule; it does not submit or advance state. */
export function planQwen35DeltaNetLayerDispatch(
  input: PlanQwen35DeltaNetLayerDispatchInput,
): Qwen35DeltaNetLayerDispatchPlan {
  const geometryData = deltaNetGeometryData(input);
  const { sequence, matrixWeights } = geometryData;
  if (input.uniforms.length !== geometryData.geometry.uniformCount) {
    fail(
      "deltanet-uniform-count-invalid",
      "Qwen3.5 DeltaNet uniform count is invalid",
    );
  }
  requireLiveness(input.deltanetParameterLiveness);
  if (
    input.state.kind !== "gated-deltanet" ||
    input.state.layer !== input.invocation.layer ||
    input.state.conv.layer !== input.invocation.layer ||
    input.state.conv.kind !== "conv" ||
    input.state.recurrent.layer !== input.invocation.layer ||
    input.state.recurrent.kind !== "recurrent"
  ) {
    fail("deltanet-state-invalid", "The Qwen3.5 layer requires DeltaNet state");
  }

  const tensorNames = input.invocation.tensors;
  const direct = {
    inputNorm: requireDirectF32Weight(
      input.weights, sequence.inputNormWeight, [HIDDEN], input.limits,
    ),
    a: requireDirectF32Weight(input.weights, tensorNames.a, [HEADS], input.limits),
    convolution: requireDirectF32Weight(
      input.weights, tensorNames.convolution, [4, QKV], input.limits,
    ),
    timeStepBias: requireDirectF32Weight(
      input.weights, tensorNames.timeStepBias, [HEADS], input.limits,
    ),
    norm: requireDirectF32Weight(input.weights, tensorNames.norm, [128], input.limits),
    postNorm: requireDirectF32Weight(
      input.weights, sequence.postAttentionNormWeight, [HIDDEN], input.limits,
    ),
  };

  const activations = {
    hidden: workspaceSlice(input.workspace, "packed-embedding-output", HIDDEN * 4),
    hiddenSecondary: workspaceSlice(input.workspace, "hidden-secondary", HIDDEN * 4),
    normalized: workspaceSlice(input.workspace, "normalized-hidden", HIDDEN * 4),
    projectionPrimary: workspaceSlice(input.workspace, "attention-projection-primary", QKV * 4),
    projectionSecondary: workspaceSlice(input.workspace, "attention-projection-secondary", QKV * 4),
    innerPrimary: workspaceSlice(input.workspace, "attention-inner-primary", INNER * 4),
    innerSecondary: workspaceSlice(input.workspace, "attention-inner-secondary", INNER * 4),
    betaOutput: workspaceSlice(input.workspace, "full-attention-key", PARAMETER_BYTES),
    decay: workspaceSlice(input.workspace, "full-attention-value", PARAMETER_BYTES),
    rawAlpha: workspaceSlice(input.workspace, "deltanet-alpha", PARAMETER_BYTES),
    rawBeta: workspaceSlice(input.workspace, "deltanet-beta", PARAMETER_BYTES),
    ffnGate: workspaceSlice(input.workspace, "ffn-gate", FFN * 4),
    ffnUp: workspaceSlice(input.workspace, "ffn-up", FFN * 4),
    ffnProduct: workspaceSlice(input.workspace, "ffn-product", FFN * 4),
    packedGemvInput: workspaceSlice(
      input.workspace,
      "packed-gemv-input-f16",
      FFN * 2,
      "f16",
    ),
  };
  const convState = stateSlice(input.state.conv, CONV_BYTES, input.limits);
  const recurrentState = stateSlice(input.state.recurrent, RECURRENT_BYTES, input.limits);

  const mutableRanges = Object.values(activations).map(({ buffer, offset, byteLength }) => ({
    buffer, offset, size: byteLength,
  }));
  mutableRanges.push(
    { buffer: convState.buffer, offset: 0, size: convState.byteLength },
    { buffer: recurrentState.buffer, offset: 0, size: recurrentState.byteLength },
  );
  requireDisjointRanges(mutableRanges, "deltanet-buffer-alias-invalid");

  const requiredWeights = [
    ...Object.values(matrixWeights),
    ...Object.values(direct).map(({ tensor }) => tensor),
  ];
  const weightRanges = requiredWeights.flatMap((tensor) =>
    tensor.physicalRows.map((view) => ({
      buffer: view.buffer,
      offset: view.bufferByteOffset,
      size: view.byteLength,
    })));
  // Shared model arenas may hold several tensors, but their owned byte ranges
  // cannot overlap without making the fixed layer program read ambiguous data.
  requireDisjointRanges(weightRanges, "deltanet-weight-alias-invalid");
  for (const mutable of mutableRanges) {
    if (weightRanges.some((weight) => rangeOverlaps(mutable, weight))) {
      fail("deltanet-buffer-alias-invalid", "Qwen3.5 state or activation aliases model weights");
    }
  }

  const commands: Qwen35DeltaNetLayerCommand[] = [];
  const usedUniforms: Qwen35BufferBinding[] = [];
  let uniformCursor = 0;
  const nextUniform = (requiredBytes: number): Qwen35ForwardBufferSlice => {
    const slice = input.uniforms[uniformCursor];
    if (slice === undefined) fail("deltanet-uniform-count-invalid", "Qwen3.5 DeltaNet uniforms are incomplete");
    const binding = makeBinding(3, "uniform", slice, requiredBytes, input.limits);
    usedUniforms.push(binding);
    uniformCursor += 1;
    return slice;
  };
  const storage = (
    index: number,
    slice: Qwen35ForwardBufferSlice,
    bytes: number,
  ): Qwen35BufferBinding => makeBinding(index, "storage", slice, bytes, input.limits);
  const uniformBinding = (
    index: number,
    slice: Qwen35ForwardBufferSlice,
    bytes: number,
  ): Qwen35BufferBinding => makeBinding(index, "uniform", slice, bytes, input.limits);
  const addPrimitive = (
    stage: Qwen35DeltaNetLayerStage,
    operation: "rms-norm" | "residual-add" | "swiglu",
    bindingSlices: readonly Qwen35ForwardBufferSlice[],
    elementCount: number,
    uniformWords: readonly number[],
    weightBinding?: Qwen35BufferBinding,
  ): void => {
    const slot = nextUniform(16);
    const plan = planPrimitiveDispatch({
      operation,
      elementCount,
      ...(operation === "rms-norm" ? { width: elementCount } : {}),
    });
    if (plan.workgroups.x > input.limits.maxComputeWorkgroupsPerDimension) {
      fail("deltanet-dispatch-invalid", "A Qwen3.5 primitive exceeds device limits");
    }
    const bindings = operation === "rms-norm"
      ? [
          storage(0, bindingSlices[0]!, elementCount * 4),
          Object.freeze({ ...weightBinding!, binding: 1 }),
          storage(2, bindingSlices[1]!, elementCount * 4),
          uniformBinding(3, slot, 16),
        ]
      : [
          storage(0, bindingSlices[0]!, elementCount * 4),
          storage(1, bindingSlices[1]!, elementCount * 4),
          storage(2, bindingSlices[2]!, elementCount * 4),
          uniformBinding(3, slot, 16),
        ];
    requireOutputDisjoint(bindings[2]!, bindings.filter((_, index) => index !== 2));
    commands.push(frozenCommand({
      stage,
      kernel: primitiveKernel(operation),
      bindings,
      workgroups: plan.workgroups,
      uniformWords,
    }));
  };
  const addGemv = (
    stage: Qwen35DeltaNetLayerStage,
    tensor: Qwen35TensorWeightView,
    activation: Qwen35ForwardBufferSlice,
    output: Qwen35ForwardBufferSlice,
  ): void => {
    const slots = tensor.physicalRows.map(() => nextUniform(20));
    const plans = planQwen35PackedGemvDispatches({
      weights: input.weights,
      tensorName: tensor.name,
      activation,
      packedActivation: activations.packedGemvInput,
      output,
      uniforms: slots,
      limits: input.limits,
    });
    commands.push(...plans.map((plan) => gemvCommand(stage, plan)));
  };
  const addResidualRms = (): void => {
    const slot = nextUniform(16);
    const plan = planPrimitiveDispatch({
      operation: "residual-rms-norm",
      elementCount: HIDDEN,
      width: HIDDEN,
    });
    const bindings = [
      storage(0, activations.normalized, HIDDEN * 4),
      storage(1, activations.hidden, HIDDEN * 4),
      Object.freeze({ ...direct.postNorm.binding, binding: 2 }),
      storage(3, activations.hiddenSecondary, HIDDEN * 4),
      uniformBinding(4, slot, 16),
    ];
    requireOutputDisjoint(bindings[3]!, bindings.slice(0, 3));
    commands.push(frozenCommand({
      stage: "post-attention-rms",
      kernel: primitiveKernel("residual-rms-norm"),
      bindings,
      workgroups: plan.workgroups,
      uniformWords: [HIDDEN, HIDDEN, f32Word(sequence.epsilon), 0],
    }));
  };
  const addTwinFfnGemv = (): boolean => {
    const eligible = canFuseTwinQ3(matrixWeights.ffnGate, matrixWeights.ffnUp);
    const gateSlots = matrixWeights.ffnGate.physicalRows.map(() => nextUniform(20));
    const upSlots = eligible
      ? []
      : matrixWeights.ffnUp.physicalRows.map(() => nextUniform(20));
    const fused = eligible
      ? planQwen35TwinQ3GemvDispatches({
          weights: input.weights,
          firstTensorName: matrixWeights.ffnGate.name,
          secondTensorName: matrixWeights.ffnUp.name,
          activation: activations.normalized,
          packedActivation: activations.packedGemvInput,
          output: activations.ffnProduct,
          uniform: gateSlots[0]!,
          limits: input.limits,
        })
      : null;
    if (eligible && fused === null) {
      fail("deltanet-dispatch-invalid", "The fused Qwen3.5 FFN projection is unavailable");
    }
    if (fused !== null) {
      commands.push(...fused.map((plan) => gemvCommand(
        plan.kernel.id.includes("swiglu") ? "swiglu" : "ffn-gate-projection",
        plan,
      )));
      return true;
    }
    const gatePlans = planQwen35PackedGemvDispatches({
      weights: input.weights, tensorName: matrixWeights.ffnGate.name,
      activation: activations.normalized, packedActivation: activations.packedGemvInput,
      output: activations.ffnGate, uniforms: gateSlots, limits: input.limits,
    });
    const upPlans = planQwen35PackedGemvDispatches({
      weights: input.weights, tensorName: matrixWeights.ffnUp.name,
      activation: activations.normalized, packedActivation: activations.packedGemvInput,
      output: activations.ffnUp, uniforms: upSlots, limits: input.limits,
    });
    commands.push(...gatePlans.map((plan) => gemvCommand("ffn-gate-projection", plan)));
    commands.push(...upPlans.map((plan) => gemvCommand("ffn-up-projection", plan)));
    return false;
  };
  const addAttentionGemvs = (): void => {
    const gateSlots = matrixWeights.attentionGate.physicalRows.map(() => nextUniform(20));
    const qkvSlots = matrixWeights.qkv.physicalRows.map(() => nextUniform(20));
    const gatePlans = planQwen35PackedGemvDispatches({
      weights: input.weights, tensorName: matrixWeights.attentionGate.name,
      activation: activations.normalized, packedActivation: activations.packedGemvInput,
      output: activations.innerPrimary, uniforms: gateSlots, limits: input.limits,
    });
    const qkvPlans = planQwen35PackedGemvDispatches({
      weights: input.weights, tensorName: matrixWeights.qkv.name,
      activation: activations.normalized, packedActivation: activations.packedGemvInput,
      output: activations.projectionPrimary, uniforms: qkvSlots, limits: input.limits,
    });
    commands.push(...gatePlans.map((plan) => gemvCommand("attention-gate-projection", plan)));
    commands.push(...qkvPlans.map((plan) => gemvCommand("attention-qkv-projection", plan)));
  };

  addPrimitive(
    "input-rms",
    "rms-norm",
    [activations.hidden, activations.normalized],
    HIDDEN,
    [HIDDEN, HIDDEN, f32Word(sequence.epsilon), 0],
    direct.inputNorm.binding,
  );
  addAttentionGemvs();
  if (canFuseTwinF32(matrixWeights.alpha, matrixWeights.beta)) {
    const slot = nextUniform(16);
    const fused = planQwen35TwinF32GemvDispatch({
      weights: input.weights,
      firstTensorName: matrixWeights.alpha.name,
      secondTensorName: matrixWeights.beta.name,
      activation: activations.normalized,
      firstOutput: activations.rawAlpha,
      secondOutput: activations.rawBeta,
      uniform: slot,
      limits: input.limits,
    });
    if (fused === null) {
      fail("deltanet-dispatch-invalid", "The fused DeltaNet parameter projection is unavailable");
    }
    commands.push(gemvCommand("deltanet-alpha-beta-projection", fused));
  } else {
    addGemv("deltanet-alpha-projection", matrixWeights.alpha, activations.normalized, activations.rawAlpha);
    addGemv("deltanet-beta-projection", matrixWeights.beta, activations.normalized, activations.rawBeta);
  }

  const parameterPlan = planQwen35HybridDispatch({
    operation: "deltanet-parameters",
    maxComputeWorkgroupsPerDimension: input.limits.maxComputeWorkgroupsPerDimension,
  });
  const parameterBindings = [
    storage(0, activations.rawBeta, PARAMETER_BYTES),
    storage(1, activations.rawAlpha, PARAMETER_BYTES),
    Object.freeze({ ...direct.timeStepBias.binding, binding: 2 }),
    Object.freeze({ ...direct.a.binding, binding: 3 }),
    storage(4, activations.betaOutput, PARAMETER_BYTES),
    storage(5, activations.decay, PARAMETER_BYTES),
  ];
  requireOutputDisjoint(parameterBindings[4]!, parameterBindings.slice(0, 4));
  requireOutputDisjoint(parameterBindings[5]!, parameterBindings.slice(0, 5));
  commands.push(frozenCommand({
    stage: "deltanet-parameters",
    kernel: hybridKernel("deltanet-parameters"),
    bindings: parameterBindings,
    workgroups: parameterPlan.workgroups,
  }));

  const convPlan = planQwen35HybridDispatch({
    operation: "deltanet-conv",
    maxComputeWorkgroupsPerDimension: input.limits.maxComputeWorkgroupsPerDimension,
  });
  const convBindings = [
    storage(0, activations.projectionPrimary, QKV * 4),
    Object.freeze({ ...direct.convolution.binding, binding: 1 }),
    storage(2, convState, CONV_BYTES),
    storage(3, activations.projectionSecondary, QKV * 4),
  ];
  requireOutputDisjoint(convBindings[3]!, convBindings.slice(0, 3));
  commands.push(frozenCommand({
    stage: "deltanet-conv",
    kernel: hybridKernel("deltanet-conv"),
    bindings: convBindings,
    workgroups: convPlan.workgroups,
    mutatesPersistentState: true,
  }));

  const recurrentPlan = planQwen35HybridDispatch({
    operation: "deltanet-recurrent-gated-norm",
    maxComputeWorkgroupsPerDimension: input.limits.maxComputeWorkgroupsPerDimension,
  });
  const recurrentBindings = [
    storage(0, activations.projectionSecondary, QKV * 4),
    storage(1, activations.betaOutput, PARAMETER_BYTES),
    storage(2, activations.decay, PARAMETER_BYTES),
    storage(3, recurrentState, RECURRENT_BYTES),
    storage(4, activations.innerPrimary, INNER * 4),
    Object.freeze({ ...direct.norm.binding, binding: 5 }),
    storage(6, activations.projectionPrimary, INNER * 4),
  ];
  requireOutputDisjoint(recurrentBindings[6]!, recurrentBindings.slice(0, 6));
  commands.push(frozenCommand({
    stage: "deltanet-recurrent-gated-norm",
    kernel: hybridKernel("deltanet-recurrent-gated-norm"),
    bindings: recurrentBindings,
    workgroups: recurrentPlan.workgroups,
    mutatesPersistentState: true,
  }));

  addGemv("deltanet-output-projection", matrixWeights.output, activations.projectionPrimary, activations.normalized);
  addResidualRms();
  const fusedSwiGlu = addTwinFfnGemv();
  if (!fusedSwiGlu) {
    addPrimitive(
      "swiglu",
      "swiglu",
      [activations.ffnGate, activations.ffnUp, activations.ffnProduct],
      FFN,
      [FFN, 0, 0, 0],
    );
  }
  addGemv("ffn-down-projection", matrixWeights.ffnDown, activations.ffnProduct, activations.normalized);
  addPrimitive(
    "mlp-residual",
    "residual-add",
    [activations.normalized, activations.hiddenSecondary, activations.hidden],
    HIDDEN,
    [HIDDEN, 0, 0, 0],
  );

  if (uniformCursor !== input.uniforms.length) {
    fail("deltanet-uniform-count-invalid", "Qwen3.5 DeltaNet uniform count is invalid");
  }
  requireDisjointRanges(usedUniforms, "deltanet-uniform-alias-invalid");
  for (const uniform of usedUniforms) {
    if (
      mutableRanges.some((range) => rangeOverlaps(uniform, range)) ||
      weightRanges.some((range) => rangeOverlaps(uniform, range))
    ) {
      fail("deltanet-buffer-alias-invalid", "Qwen3.5 uniforms alias a storage range");
    }
  }

  return Object.freeze({
    layer: input.invocation.layer,
    commands: Object.freeze(commands),
    uniformCount: uniformCursor,
    stateSemantics: Object.freeze({
      mutatingStages: Object.freeze([
        "deltanet-conv",
        "deltanet-recurrent-gated-norm",
      ] as const),
      failureAfterSubmission: "persistent-state-indeterminate-dispose-required",
      advancePositionAfter: "successful-queue-retirement",
    }),
  });
}

function requireOutputDisjoint(
  output: Qwen35BufferBinding,
  reads: readonly Qwen35BufferBinding[],
): void {
  if (reads.some((read) => rangeOverlaps(output, read))) {
    fail("deltanet-buffer-alias-invalid", "A Qwen3.5 DeltaNet output aliases an input");
  }
}

function gemvCommand(
  stage: Qwen35DeltaNetLayerStage,
  plan: Qwen35ForwardDispatchPlan,
): Qwen35DeltaNetLayerCommand {
  return frozenCommand({
    stage,
    kernel: plan.kernel,
    bindings: plan.bindings,
    workgroups: plan.workgroups,
    uniformWords: plan.uniformWords,
  });
}
