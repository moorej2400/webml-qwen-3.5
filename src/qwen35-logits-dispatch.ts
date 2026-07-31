import { diagnosticError } from "./diagnostics.js";
import type {
  Qwen35ActivationResourceKind,
  Qwen35ActivationResourceView,
} from "./qwen35-activation-workspace.js";
import {
  planQwen35TiedLogitsGeometry,
  planQwen35TiedLogitsDispatches,
  type Qwen35ForwardBufferSlice,
  type Qwen35ForwardDeviceLimits,
  type Qwen35TiedLogitsDispatchPlan,
} from "./qwen35-forward-dispatch.js";
import {
  QWEN35_LOGITS_REDUCTION_KERNELS,
  planQwen35FinalTokenSelection,
  planQwen35LogitsTileWinner,
  type Qwen35LogitsReductionOperation,
} from "./qwen35-logits-reduction.js";
import type { Qwen35WeightDirectoryView } from "./qwen35-weight-directory.js";
import type {
  Qwen35BufferBinding,
  Qwen35DispatchRequest,
  Qwen35KernelSource,
} from "./qwen35-webgpu-executor.js";

const QWEN35_HIDDEN_BYTES = 2_560 * 4;
const QWEN35_LOGITS_TILE_ROWS = 1_024;
const QWEN35_DECODABLE_ROWS = 248_070;
const QWEN35_MATHEMATICAL_TILE_COUNT = 243;
const QWEN35_REDUCTION_COMMAND_COUNT = QWEN35_MATHEMATICAL_TILE_COUNT + 1;
const QWEN35_CANDIDATE_CAPACITY = 256;
const GPU_BUFFER_USAGE_COPY_SRC = 0x0004;
const GPU_BUFFER_USAGE_STORAGE = 0x0080;

export interface Qwen35LogitsWorkspaceViews {
  get(kind: Qwen35ActivationResourceKind): Qwen35ActivationResourceView;
}

export interface Qwen35LogitsDispatchCommand extends Qwen35DispatchRequest {
  readonly kind:
    | "logits-gemv-piece"
    | "logits-tile-top-1"
    | "indexed-top-1";
  readonly tileIndex: number | null;
  readonly uniformWords: readonly number[];
}

export interface Qwen35SelectedTokenReadback {
  readonly buffer: object;
  readonly offset: number;
  readonly byteLength: 4;
  readonly scalarType: "u32";
}

export interface Qwen35TiledLogitsCommands {
  readonly commands: readonly Qwen35LogitsDispatchCommand[];
  readonly candidateCount: 243;
  readonly uniformCount: number;
  readonly selectedTokenReadback: Qwen35SelectedTokenReadback;
}

const REDUCTION_KERNEL_SOURCES = new Map<
  Qwen35LogitsReductionOperation,
  Qwen35KernelSource
>(QWEN35_LOGITS_REDUCTION_KERNELS.map((kernel) => [
  kernel.operation,
  Object.freeze({
    id: kernel.id,
    source: kernel.source,
    entryPoint: kernel.entryPoint,
  }),
]));

function overlaps(left: Qwen35BufferBinding, right: Qwen35BufferBinding): boolean {
  return left.buffer === right.buffer &&
    left.offset < right.offset + right.size &&
    right.offset < left.offset + left.size;
}

function requireNoOverlap(
  left: readonly Qwen35BufferBinding[],
  right: readonly Qwen35BufferBinding[],
  allowSameEntry = false,
): void {
  for (let leftIndex = 0; leftIndex < left.length; leftIndex += 1) {
    for (let rightIndex = 0; rightIndex < right.length; rightIndex += 1) {
      if (
        allowSameEntry &&
        left === right &&
        leftIndex === rightIndex
      ) {
        continue;
      }
      if (overlaps(left[leftIndex]!, right[rightIndex]!)) {
        throw diagnosticError(
          "logits-dispatch-buffer-alias-invalid",
          "Qwen3.5 logits command buffers have an unsafe alias",
        );
      }
    }
  }
}

function requirePositiveLimit(value: number): boolean {
  return Number.isSafeInteger(value) && value > 0;
}

function bindSlice(
  bindingIndex: number,
  kind: "storage" | "uniform",
  slice: Qwen35ForwardBufferSlice,
  requiredBytes: number,
  limits: Qwen35ForwardDeviceLimits,
  errorCode: string,
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
    !requirePositiveLimit(alignment) ||
    slice.offset % alignment !== 0 ||
    !requirePositiveLimit(maximum) ||
    !Number.isSafeInteger(requiredBytes) ||
    requiredBytes <= 0 ||
    requiredBytes % 4 !== 0 ||
    requiredBytes > maximum ||
    !Number.isSafeInteger(slice.offset + requiredBytes)
  ) {
    throw diagnosticError(
      errorCode,
      "A Qwen3.5 logits command binding is invalid",
    );
  }
  return Object.freeze({
    binding: bindingIndex,
    kind,
    buffer: slice.buffer,
    offset: slice.offset,
    size: requiredBytes,
  });
}

function requireWorkspaceResource(
  workspace: Qwen35LogitsWorkspaceViews,
  kind: "logits-tile" | "top-k-scores" | "top-k-indices" | "selected-token",
  scalarType: "f32" | "u32",
  elementCount: number,
  requiredUsage: number,
  limits: Qwen35ForwardDeviceLimits,
): { readonly view: Qwen35ActivationResourceView; readonly binding: Qwen35BufferBinding } {
  let view: Qwen35ActivationResourceView;
  try {
    view = workspace.get(kind);
  } catch {
    throw diagnosticError(
      "logits-dispatch-workspace-invalid",
      "The Qwen3.5 logits workspace is incomplete",
    );
  }
  const bytes = elementCount * 4;
  if (
    view.kind !== kind ||
    view.scalarType !== scalarType ||
    view.elementCount !== elementCount ||
    view.bytes !== BigInt(bytes) ||
    view.byteLength !== bytes ||
    (view.usage & requiredUsage) !== requiredUsage ||
    view.binding.size !== bytes
  ) {
    throw diagnosticError(
      "logits-dispatch-workspace-invalid",
      "A Qwen3.5 logits workspace resource has an invalid shape or type",
    );
  }
  let bound: Qwen35BufferBinding;
  try {
    bound = bindSlice(
      0,
      "storage",
      {
        buffer: view.binding.buffer,
        offset: view.binding.offset,
        byteLength: view.binding.size,
      },
      bytes,
      limits,
      "logits-dispatch-workspace-invalid",
    );
  } catch {
    throw diagnosticError(
      "logits-dispatch-workspace-invalid",
      "A Qwen3.5 logits workspace resource cannot be bound",
    );
  }
  return Object.freeze({ view, binding: bound });
}

function requireDistinctUniforms(uniforms: readonly Qwen35BufferBinding[]): void {
  for (let index = 0; index < uniforms.length; index += 1) {
    for (let prior = 0; prior < index; prior += 1) {
      if (overlaps(uniforms[index]!, uniforms[prior]!)) {
        throw diagnosticError(
          "logits-dispatch-uniform-alias-invalid",
          "Qwen3.5 logits commands require distinct uniform ranges",
        );
      }
    }
  }
}

/** Validates the physical-piece groups before reduction commands are inserted. */
export function validateQwen35TiedLogitsDispatchGroups(
  pieces: readonly Qwen35TiedLogitsDispatchPlan[],
): void {
  let expectedTile = 0;
  let expectedOutputOffset = 0;
  for (const piece of pieces) {
    const expectedStart = expectedTile * QWEN35_LOGITS_TILE_ROWS;
    const expectedRows = Math.min(
      QWEN35_LOGITS_TILE_ROWS,
      QWEN35_DECODABLE_ROWS - expectedStart,
    );
    const nextOutputOffset = expectedOutputOffset + piece.pieceRows;
    const expectedCompletion = nextOutputOffset === expectedRows;
    if (
      expectedTile >= QWEN35_MATHEMATICAL_TILE_COUNT ||
      piece.tileIndex !== expectedTile ||
      piece.vocabularyStart !== expectedStart ||
      piece.tileRows !== expectedRows ||
      piece.pieceOutputOffset !== expectedOutputOffset ||
      !Number.isSafeInteger(piece.pieceRows) ||
      piece.pieceRows < 1 ||
      nextOutputOffset > expectedRows ||
      piece.completesTile !== expectedCompletion ||
      piece.uniformWords.length !== 5 ||
      piece.uniformWords[0] !== piece.pieceRows ||
      piece.uniformWords[4] !== piece.pieceOutputOffset
    ) {
      throw diagnosticError(
        "logits-dispatch-group-invalid",
        "Qwen3.5 logits physical pieces do not form complete logical tiles",
      );
    }
    if (expectedCompletion) {
      expectedTile += 1;
      expectedOutputOffset = 0;
    } else {
      expectedOutputOffset = nextOutputOffset;
    }
  }
  if (
    expectedTile !== QWEN35_MATHEMATICAL_TILE_COUNT ||
    expectedOutputOffset !== 0
  ) {
    throw diagnosticError(
      "logits-dispatch-group-invalid",
      "Qwen3.5 logits physical pieces do not cover every logical tile",
    );
  }
}

function reductionKernel(
  operation: Qwen35LogitsReductionOperation,
): Qwen35KernelSource {
  const kernel = REDUCTION_KERNEL_SOURCES.get(operation);
  if (kernel === undefined) {
    throw diagnosticError(
      "logits-dispatch-kernel-invalid",
      "A required Qwen3.5 logits reduction kernel is unavailable",
    );
  }
  return kernel;
}

function command(input: {
  readonly kind: Qwen35LogitsDispatchCommand["kind"];
  readonly tileIndex: number | null;
  readonly kernel: Qwen35KernelSource;
  readonly bindings: readonly Qwen35BufferBinding[];
  readonly uniformWords: readonly number[];
  readonly workgroups: Qwen35DispatchRequest["workgroups"];
}): Qwen35LogitsDispatchCommand {
  return Object.freeze({
    kind: input.kind,
    tileIndex: input.tileIndex,
    kernel: input.kernel,
    bindings: Object.freeze([...input.bindings]),
    uniformWords: Object.freeze([...input.uniformWords]),
    workgroups: Object.freeze({ ...input.workgroups }),
  });
}

/** Returns the exact arena slot count without requiring any uniform buffers. */
export function planQwen35TiledLogitsUniformCount(input: {
  readonly weights: Qwen35WeightDirectoryView;
  readonly limits: Qwen35ForwardDeviceLimits;
}): number {
  return planQwen35TiedLogitsGeometry(input).uniformCount;
}

/** Assembles the complete GPU-only greedy logits tail for one decode token. */
export function assembleQwen35TiledLogitsCommands(input: {
  readonly weights: Qwen35WeightDirectoryView;
  readonly normalizedHidden: Qwen35ForwardBufferSlice;
  readonly workspace: Qwen35LogitsWorkspaceViews;
  readonly limits: Qwen35ForwardDeviceLimits;
  readonly uniforms: readonly Qwen35ForwardBufferSlice[];
}): Qwen35TiledLogitsCommands {
  const logitsTile = requireWorkspaceResource(
    input.workspace,
    "logits-tile",
    "f32",
    QWEN35_LOGITS_TILE_ROWS,
    GPU_BUFFER_USAGE_STORAGE,
    input.limits,
  );
  const candidateScores = requireWorkspaceResource(
    input.workspace,
    "top-k-scores",
    "f32",
    QWEN35_CANDIDATE_CAPACITY,
    GPU_BUFFER_USAGE_STORAGE,
    input.limits,
  );
  const candidateIndices = requireWorkspaceResource(
    input.workspace,
    "top-k-indices",
    "u32",
    QWEN35_CANDIDATE_CAPACITY,
    GPU_BUFFER_USAGE_STORAGE,
    input.limits,
  );
  const selectedToken = requireWorkspaceResource(
    input.workspace,
    "selected-token",
    "u32",
    1,
    GPU_BUFFER_USAGE_STORAGE | GPU_BUFFER_USAGE_COPY_SRC,
    input.limits,
  );
  const normalizedHidden = bindSlice(
    1,
    "storage",
    input.normalizedHidden,
    QWEN35_HIDDEN_BYTES,
    input.limits,
    "logits-dispatch-hidden-invalid",
  );

  const geometry = planQwen35TiedLogitsGeometry({
    weights: input.weights,
    limits: input.limits,
  });
  if (
    geometry.reductionDispatchCount !== QWEN35_REDUCTION_COMMAND_COUNT ||
    input.uniforms.length !== geometry.uniformCount
  ) {
    throw diagnosticError(
      "logits-dispatch-uniform-count-invalid",
      "Qwen3.5 logits command uniform count is invalid",
    );
  }
  let pieces: readonly Qwen35TiedLogitsDispatchPlan[];
  try {
    pieces = planQwen35TiedLogitsDispatches({
      weights: input.weights,
      activation: input.normalizedHidden,
      output: {
        buffer: logitsTile.binding.buffer,
        offset: logitsTile.binding.offset,
        byteLength: logitsTile.binding.size,
      },
      uniforms: input.uniforms.slice(0, geometry.physicalPieceCount),
      limits: input.limits,
    });
  } catch (error) {
    const code = (error as { readonly code?: unknown }).code;
    if (code === "forward-uniform-count-invalid") {
      throw diagnosticError(
        "logits-dispatch-uniform-count-invalid",
        "Qwen3.5 logits command uniform count is invalid",
      );
    }
    if (code === "forward-uniform-alias-invalid") {
      throw diagnosticError(
        "logits-dispatch-uniform-alias-invalid",
        "Qwen3.5 logits commands require distinct uniform ranges",
      );
    }
    throw error;
  }
  validateQwen35TiedLogitsDispatchGroups(pieces);
  if (pieces.length !== geometry.physicalPieceCount) {
    throw diagnosticError(
      "logits-dispatch-uniform-count-invalid",
      "Qwen3.5 logits command uniform count is invalid",
    );
  }

  const uniformBindings = input.uniforms.map((uniform, index) => bindSlice(
    3,
    "uniform",
    uniform,
    index < pieces.length ? 20 : 16,
    input.limits,
    "logits-dispatch-uniform-invalid",
  ));
  requireDistinctUniforms(uniformBindings);

  const workspaceBindings = [
    logitsTile.binding,
    candidateScores.binding,
    candidateIndices.binding,
    selectedToken.binding,
  ];
  requireNoOverlap(workspaceBindings, workspaceBindings, true);
  const weightBindings = pieces.map((piece) => piece.bindings[0]!);
  requireNoOverlap(workspaceBindings, [normalizedHidden]);
  requireNoOverlap(workspaceBindings, weightBindings);
  requireNoOverlap(workspaceBindings, uniformBindings);
  requireNoOverlap(uniformBindings, [normalizedHidden]);
  requireNoOverlap(uniformBindings, weightBindings);

  const commands: Qwen35LogitsDispatchCommand[] = [];
  let reductionUniformIndex = pieces.length;
  for (const piece of pieces) {
    commands.push(Object.freeze({
      ...piece,
      kind: "logits-gemv-piece" as const,
      tileIndex: piece.tileIndex,
    }));
    if (!piece.completesTile) continue;

    const plan = planQwen35LogitsTileWinner({
      vocabularyStart: piece.vocabularyStart,
      vocabularyRows: piece.tileRows,
      candidateSlot: piece.tileIndex,
    });
    commands.push(command({
      kind: "logits-tile-top-1",
      tileIndex: piece.tileIndex,
      kernel: reductionKernel(plan.operation),
      bindings: [
        Object.freeze({ ...logitsTile.binding, binding: 0 }),
        Object.freeze({ ...candidateScores.binding, binding: 1 }),
        Object.freeze({ ...candidateIndices.binding, binding: 2 }),
        uniformBindings[reductionUniformIndex++]!,
      ],
      uniformWords: plan.uniformWords,
      workgroups: plan.workgroups,
    }));
  }
  const finalPlan = planQwen35FinalTokenSelection({
    candidateCount: QWEN35_MATHEMATICAL_TILE_COUNT,
  });
  commands.push(command({
    kind: "indexed-top-1",
    tileIndex: null,
    kernel: reductionKernel(finalPlan.operation),
    bindings: [
      Object.freeze({ ...candidateScores.binding, binding: 0 }),
      Object.freeze({ ...candidateIndices.binding, binding: 1 }),
      Object.freeze({ ...selectedToken.binding, binding: 2 }),
      uniformBindings[reductionUniformIndex++]!,
    ],
    uniformWords: finalPlan.uniformWords,
    workgroups: finalPlan.workgroups,
  }));
  if (reductionUniformIndex !== uniformBindings.length) {
    throw diagnosticError(
      "logits-dispatch-uniform-count-invalid",
      "Qwen3.5 logits command uniform count is invalid",
    );
  }

  return Object.freeze({
    commands: Object.freeze(commands),
    candidateCount: QWEN35_MATHEMATICAL_TILE_COUNT,
    uniformCount: geometry.uniformCount,
    selectedTokenReadback: Object.freeze({
      buffer: selectedToken.binding.buffer,
      offset: selectedToken.binding.offset,
      byteLength: 4,
      scalarType: "u32",
    }),
  });
}
