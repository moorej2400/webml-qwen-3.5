import { diagnosticError } from "./diagnostics.js";
import { GgmlType } from "./gguf.js";
import {
  QWEN_PRIMITIVE_KERNELS,
  planPrimitiveDispatch,
} from "./qwen-primitives.js";
import type {
  Qwen35ActivationResourceKind,
  Qwen35ActivationResourceView,
  Qwen35ActivationWorkspace,
} from "./qwen35-activation-workspace.js";
import type {
  Qwen35ForwardBufferSlice,
  Qwen35ForwardDeviceLimits,
} from "./qwen35-forward-dispatch.js";
import { QWEN35_NO_SELECTED_TOKEN } from "./qwen35-logits-reduction.js";
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
const HIDDEN_BYTES = HIDDEN * 4;

export interface Qwen35FinalNormCommand extends Qwen35DispatchRequest {
  readonly stage: "final-rms";
  readonly uniformWords: readonly [number, number, number, number];
}

export interface Qwen35FinalNormDispatchPlan {
  readonly uniformCount: 1;
  readonly command: Qwen35FinalNormCommand;
}

export interface PlanQwen35FinalNormDispatchInput {
  readonly program: Qwen35Program;
  readonly invocation: Extract<Qwen35Invocation, { kind: "rms-norm" }>;
  readonly weights: Qwen35WeightDirectoryView;
  readonly workspace: Pick<Qwen35ActivationWorkspace, "get">;
  readonly limits: Qwen35ForwardDeviceLimits;
  readonly uniform: Qwen35ForwardBufferSlice;
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

function binding(
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
    !Number.isSafeInteger(slice.offset) ||
    slice.offset < 0 ||
    !Number.isSafeInteger(slice.byteLength) ||
    slice.byteLength < requiredBytes ||
    !Number.isSafeInteger(alignment) ||
    alignment < 1 ||
    slice.offset % alignment !== 0 ||
    !Number.isSafeInteger(maximum) ||
    requiredBytes > maximum ||
    !Number.isSafeInteger(slice.offset + requiredBytes)
  ) {
    fail(
      "final-norm-binding-invalid",
      "A Qwen3.5 final normalization binding is invalid",
    );
  }
  return Object.freeze({
    binding: index,
    kind,
    buffer: slice.buffer,
    offset: slice.offset,
    size: requiredBytes,
  });
}

function workspaceSlice(
  workspace: PlanQwen35FinalNormDispatchInput["workspace"],
  kind: "packed-embedding-output" | "normalized-hidden",
): Qwen35ForwardBufferSlice {
  let view: Qwen35ActivationResourceView | undefined;
  // The workspace callback is caller-owned, so its thrown text cannot cross this boundary.
  try {
    view = workspace.get(kind as Qwen35ActivationResourceKind);
  } catch {
    fail(
      "final-norm-workspace-invalid",
      "The Qwen3.5 final normalization workspace is incomplete",
    );
  }
  if (
    view === undefined ||
    view.kind !== kind ||
    view.scalarType !== "f32" ||
    view.elementCount !== HIDDEN ||
    view.bytes !== BigInt(HIDDEN_BYTES) ||
    view.byteLength !== HIDDEN_BYTES ||
    view.binding.offset !== 0 ||
    view.binding.size !== HIDDEN_BYTES
  ) {
    fail(
      "final-norm-workspace-invalid",
      "A Qwen3.5 final normalization workspace view is invalid",
    );
  }
  return Object.freeze({
    buffer: view.binding.buffer,
    offset: view.binding.offset,
    byteLength: view.binding.size,
  });
}

function overlaps(left: Qwen35BufferBinding, right: Qwen35BufferBinding): boolean {
  return left.buffer === right.buffer &&
    left.offset < right.offset + right.size &&
    right.offset < left.offset + left.size;
}

function f32Word(value: number): number {
  const bytes = new ArrayBuffer(4);
  const view = new DataView(bytes);
  view.setFloat32(0, value, true);
  return view.getUint32(0, true);
}

function finalKernel(): Qwen35KernelSource {
  const kernel = QWEN_PRIMITIVE_KERNELS.find(
    (candidate) => candidate.operation === "rms-norm",
  );
  if (kernel === undefined) {
    fail("final-norm-kernel-missing", "The Qwen3.5 RMS kernel is unavailable");
  }
  return Object.freeze({ id: kernel.id, source: kernel.source, entryPoint: "main" });
}

/** Builds the final output normalization before tiled tied logits. */
export function planQwen35FinalNormDispatch(
  input: PlanQwen35FinalNormDispatchInput,
): Qwen35FinalNormDispatchPlan {
  const index = input.program.invocations.indexOf(input.invocation);
  const logits = input.program.invocations[index + 1];
  const reduction = input.program.invocations[index + 2];
  const readback = reduction?.kind === "greedy-logits-reduction"
    ? reduction.selectedTokenReadback
    : undefined;
  const reductionKernels = reduction?.kind === "greedy-logits-reduction"
    ? reduction.kernels
    : undefined;
  // The driver binds this fixed tail without ABI renegotiation, so every field is fail-closed.
  if (
    input.program.model !== "qwen35-4b" ||
    !isRunnableProgram(input.program) ||
    index < 0 ||
    (input.invocation as { readonly kind?: unknown }).kind !== "rms-norm" ||
    input.invocation.layer !== "final" ||
    input.invocation.site !== "final" ||
    input.invocation.weight !== "output_norm.weight" ||
    input.invocation.fp32Accumulation !== true ||
    !Number.isFinite(input.invocation.epsilon) ||
    input.invocation.epsilon < 0 ||
    logits?.kind !== "tiled-tied-logits" ||
    logits.weight !== "token_embd.weight" ||
    logits.tiedWeightOwner !== "embedding" ||
    logits.modelRows !== 248_320 ||
    logits.decodableRows !== 248_070 ||
    logits.columns !== HIDDEN ||
    logits.logicalTileRows !== 1_024 ||
    logits.mathematicalTileCount !== 243 ||
    logits.finalTileRows !== 262 ||
    reduction?.kind !== "greedy-logits-reduction" ||
    !Array.isArray(reductionKernels) ||
    reductionKernels.length !== 2 ||
    reductionKernels[0] !== "logits-tile-top-1" ||
    reductionKernels[1] !== "indexed-top-1" ||
    reduction.mathematicalTileCount !== 243 ||
    reduction.candidatesPerTile !== 1 ||
    reduction.candidateCount !== 243 ||
    reduction.candidateCapacity !== 256 ||
    readback?.resource !== "selected-token" ||
    readback.scalarType !== "u32" ||
    readback.elementCount !== 1 ||
    readback.byteOffset !== 0 ||
    readback.noSelectionSentinel !== QWEN35_NO_SELECTED_TOKEN ||
    reduction.runnable !== true ||
    index + 3 !== input.program.invocations.length
  ) {
    fail(
      "final-norm-program-invalid",
      "The Qwen3.5 final normalization program is invalid",
    );
  }

  let weight: Qwen35TensorWeightView | undefined;
  // The directory callback is caller-owned, so its thrown text cannot cross this boundary.
  try {
    weight = input.weights.get("output_norm.weight");
  } catch {
    fail(
      "final-norm-weight-invalid",
      "The Qwen3.5 final normalization weight must use one F32 range",
    );
  }
  const view = weight?.physicalRows[0];
  if (
    weight === undefined ||
    weight.name !== "output_norm.weight" ||
    weight.shape.length !== 1 ||
    weight.shape[0] !== HIDDEN ||
    weight.ggmlType !== GgmlType.F32 ||
    weight.storageType !== "f32" ||
    weight.rowBytes !== HIDDEN_BYTES ||
    weight.rowCount !== 1 ||
    weight.logicalBytes !== BigInt(HIDDEN_BYTES) ||
    weight.physicalRows.length !== 1 ||
    view === undefined ||
    view.firstRow !== 0 ||
    view.rowCount !== 1 ||
    view.tensorByteOffset !== 0 ||
    view.byteLength !== HIDDEN_BYTES
  ) {
    fail(
      "final-norm-weight-invalid",
      "The Qwen3.5 final normalization weight must use one F32 range",
    );
  }

  const hidden = binding(
    0,
    "storage",
    workspaceSlice(input.workspace, "packed-embedding-output"),
    HIDDEN_BYTES,
    input.limits,
  );
  const weightBinding = binding(1, "storage", {
    buffer: view.buffer,
    offset: view.bufferByteOffset,
    byteLength: view.byteLength,
  }, HIDDEN_BYTES, input.limits);
  const output = binding(
    2,
    "storage",
    workspaceSlice(input.workspace, "normalized-hidden"),
    HIDDEN_BYTES,
    input.limits,
  );
  const uniform = binding(3, "uniform", input.uniform, 16, input.limits);
  const ranges = [hidden, weightBinding, output, uniform];
  for (let current = 0; current < ranges.length; current += 1) {
    for (let prior = 0; prior < current; prior += 1) {
      if (overlaps(ranges[current]!, ranges[prior]!)) {
        fail(
          "final-norm-buffer-alias-invalid",
          "Qwen3.5 final normalization buffers overlap",
        );
      }
    }
  }

  const primitive = planPrimitiveDispatch({
    operation: "rms-norm",
    elementCount: HIDDEN,
  });
  if (primitive.workgroups.x > input.limits.maxComputeWorkgroupsPerDimension) {
    fail(
      "final-norm-dispatch-invalid",
      "The Qwen3.5 final normalization dispatch exceeds device limits",
    );
  }
  const uniformWords = Object.freeze([
    HIDDEN,
    HIDDEN,
    f32Word(input.invocation.epsilon),
    0,
  ] as const);
  const command = Object.freeze({
    stage: "final-rms" as const,
    kernel: finalKernel(),
    bindings: Object.freeze(ranges),
    workgroups: Object.freeze({ ...primitive.workgroups }),
    uniformWords,
  });
  return Object.freeze({ uniformCount: 1 as const, command });
}
