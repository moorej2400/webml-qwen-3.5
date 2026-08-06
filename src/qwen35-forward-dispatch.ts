import { diagnosticError } from "./diagnostics.js";
import {
  LANGUAGE_GEMV_KERNELS,
  planGemvDispatch,
  type GemvLayout,
  type LanguageGemvKernel,
} from "./mixed-gemv.js";
import {
  PACKED_EMBEDDING_KERNELS,
  planPackedEmbeddingRow,
  type PackedEmbeddingKernel,
} from "./qwen-embedding.js";
import type {
  Qwen35PhysicalRowView,
  Qwen35TensorWeightView,
  Qwen35WeightDirectoryView,
} from "./qwen35-weight-directory.js";
import type { Qwen35StagedPackedRows } from "./qwen35-disk-backed-tied-embedding.js";
import type {
  Qwen35BufferBinding,
  Qwen35DispatchRequest,
  Qwen35KernelSource,
} from "./qwen35-webgpu-executor.js";

export type Qwen35ForwardDeviceLimits = Readonly<{
  minStorageBufferOffsetAlignment: number;
  minUniformBufferOffsetAlignment: number;
  maxStorageBufferBindingSize: number;
  maxUniformBufferBindingSize: number;
  maxComputeWorkgroupsPerDimension: number;
}>;

export interface Qwen35ForwardBufferSlice {
  readonly buffer: object;
  readonly offset: number;
  /** Available bytes from offset; the planner binds only the required prefix. */
  readonly byteLength: number;
}

export interface Qwen35ForwardDispatchPlan extends Qwen35DispatchRequest {
  /** Caller writes these u32 values into this request's uniform binding. */
  readonly uniformWords: readonly number[];
}

export interface Qwen35TiedLogitsDispatchPlan
  extends Qwen35ForwardDispatchPlan {
  readonly tileIndex: number;
  /** Logical vocabulary row represented by tile element zero. */
  readonly vocabularyStart: number;
  readonly tileRows: number;
  readonly pieceOutputOffset: number;
  readonly pieceRows: number;
  /** Top-k runs once after the piece with this marker completes. */
  readonly completesTile: boolean;
}

export interface Qwen35TiedLogitsGeometry {
  readonly modelRows: 248_320;
  readonly decodableRows: 248_070;
  readonly logicalTileRows: 1_024;
  readonly mathematicalTileCount: 243;
  readonly finalTileRows: 262;
  readonly physicalPieceCount: number;
  readonly reductionDispatchCount: 244;
  readonly uniformCount: number;
}

const MAX_LOGITS_TILE_ROWS = 1_024;
const QWEN35_HIDDEN_SIZE = 2_560;
const QWEN35_VOCABULARY_SIZE = 248_320;
const QWEN35_DECODABLE_LOGIT_ROWS = 248_070;
const QWEN35_MATHEMATICAL_LOGITS_TILES = 243;
const QWEN35_FINAL_LOGITS_TILE_ROWS = 262;
const QWEN35_LOGITS_REDUCTION_DISPATCHES = 244;

const VISUAL_EMBEDDING_KERNEL: Qwen35KernelSource = Object.freeze({
  id: "qwen35-visual-embedding-f32",
  entryPoint: "main",
  source: /* wgsl */ `
struct Params { output_elements: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<storage, read> source: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x < params.output_elements) { output[id.x] = source[id.x]; }
}`,
});

const EMBEDDING_KERNEL_SOURCES = new Map<GemvLayout, Qwen35KernelSource>(
  PACKED_EMBEDDING_KERNELS.map((kernel) => [
    kernel.storageType,
    Object.freeze({
      id: kernel.id,
      source: kernel.source,
      entryPoint: "main",
    }),
  ]),
);

const GEMV_KERNEL_SOURCES = new Map<GemvLayout, Qwen35KernelSource>(
  LANGUAGE_GEMV_KERNELS.map((kernel) => [
    kernel.layout,
    Object.freeze({
      id: kernel.id,
      source: kernel.source,
      entryPoint: "packed_gemv",
    }),
  ]),
);

function layoutOf(tensor: Qwen35TensorWeightView): GemvLayout {
  const embedding = PACKED_EMBEDDING_KERNELS.find(
    (kernel) => kernel.storageType === tensor.storageType,
  );
  if (embedding === undefined || embedding.ggmlType !== tensor.ggmlType) {
    throw diagnosticError(
      "forward-weight-layout-invalid",
      "Qwen3.5 forward weight type and packed layout do not match",
    );
  }
  return embedding.storageType;
}

function requireTensor(
  weights: Qwen35WeightDirectoryView,
  name: string,
): Qwen35TensorWeightView {
  const tensor = weights.get(name);
  if (tensor === undefined) {
    throw diagnosticError(
      "forward-weight-missing",
      "A required Qwen3.5 forward weight is unavailable",
    );
  }
  return tensor;
}

function matrixShape(tensor: Qwen35TensorWeightView): {
  readonly columns: number;
  readonly rows: number;
} {
  if (
    tensor.shape.length !== 2 ||
    !Number.isSafeInteger(tensor.shape[0]) ||
    tensor.shape[0]! <= 0 ||
    !Number.isSafeInteger(tensor.shape[1]) ||
    tensor.shape[1]! <= 0 ||
    tensor.rowCount !== tensor.shape[1]
  ) {
    throw diagnosticError(
      "forward-weight-shape-invalid",
      "Qwen3.5 forward weight must be a positive packed matrix",
    );
  }
  return { columns: tensor.shape[0]!, rows: tensor.shape[1]! };
}

function requireTiedEmbeddingShape(shape: {
  readonly columns: number;
  readonly rows: number;
}): void {
  if (
    shape.columns !== QWEN35_HIDDEN_SIZE ||
    shape.rows !== QWEN35_VOCABULARY_SIZE
  ) {
    throw diagnosticError(
      "forward-embedding-shape-invalid",
      "The tied Qwen3.5 embedding matrix has an unexpected model shape",
    );
  }
}

function expectedRowBytes(
  tensor: Qwen35TensorWeightView,
  columns: number,
): number {
  const layout = layoutOf(tensor);
  const kernel = LANGUAGE_GEMV_KERNELS.find(
    (candidate) => candidate.layout === layout,
  );
  if (
    kernel === undefined ||
    columns % kernel.abi.valuesPerBlock !== 0
  ) {
    throw diagnosticError(
      "forward-weight-shape-invalid",
      "Qwen3.5 forward matrix row splits a packed block",
    );
  }
  return (columns / kernel.abi.valuesPerBlock) * kernel.abi.bytesPerBlock;
}

function physicalRows(
  tensor: Qwen35TensorWeightView,
  rows: number,
  rowBytes: number,
): readonly Qwen35PhysicalRowView[] {
  if (
    tensor.rowBytes !== rowBytes ||
    tensor.logicalBytes !== BigInt(rows) * BigInt(rowBytes) ||
    tensor.physicalRows.length === 0
  ) {
    throw diagnosticError(
      "forward-weight-views-invalid",
      "Qwen3.5 physical weight rows do not match the packed matrix",
    );
  }
  let nextRow = 0;
  let nextByte = 0;
  const rangesByBuffer = new Map<object, { start: number; end: number }[]>();
  for (const view of tensor.physicalRows) {
    if (
      !Number.isSafeInteger(view.firstRow) ||
      view.firstRow !== nextRow ||
      !Number.isSafeInteger(view.rowCount) ||
      view.rowCount <= 0 ||
      !Number.isSafeInteger(view.tensorByteOffset) ||
      view.tensorByteOffset !== nextByte ||
      !Number.isSafeInteger(view.bufferByteOffset) ||
      view.bufferByteOffset < 0 ||
      !Number.isSafeInteger(view.byteLength) ||
      view.byteLength !== view.rowCount * rowBytes
    ) {
      throw diagnosticError(
        "forward-weight-views-invalid",
        "Qwen3.5 physical weight rows are not contiguous complete rows",
      );
    }
    const range = {
      start: view.bufferByteOffset,
      end: view.bufferByteOffset + view.byteLength,
    };
    const priorRanges = rangesByBuffer.get(view.buffer) ?? [];
    if (
      !Number.isSafeInteger(range.end) ||
      priorRanges.some(
        (prior) => range.start < prior.end && prior.start < range.end,
      )
    ) {
      throw diagnosticError(
        "forward-weight-views-invalid",
        "Qwen3.5 physical weight ranges overlap in one GPU buffer",
      );
    }
    priorRanges.push(range);
    rangesByBuffer.set(view.buffer, priorRanges);
    nextRow += view.rowCount;
    nextByte += view.byteLength;
  }
  if (nextRow !== rows || nextByte !== Number(tensor.logicalBytes)) {
    throw diagnosticError(
      "forward-weight-views-invalid",
      "Qwen3.5 physical weight rows do not cover the packed matrix",
    );
  }
  return tensor.physicalRows;
}

function positiveLimit(value: number): boolean {
  return Number.isSafeInteger(value) && value > 0;
}

function binding(
  bindingIndex: number,
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
    !positiveLimit(alignment) ||
    slice.offset % alignment !== 0 ||
    !positiveLimit(maximum) ||
    !Number.isSafeInteger(requiredBytes) ||
    requiredBytes <= 0 ||
    requiredBytes % 4 !== 0 ||
    requiredBytes > maximum ||
    !Number.isSafeInteger(slice.offset + requiredBytes)
  ) {
    throw diagnosticError(
      "forward-binding-invalid",
      "A Qwen3.5 forward buffer binding is invalid for this device",
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

function physicalBinding(
  bindingIndex: number,
  view: Qwen35PhysicalRowView,
  limits: Qwen35ForwardDeviceLimits,
): Qwen35BufferBinding {
  return binding(
    bindingIndex,
    "storage",
    {
      buffer: view.buffer,
      offset: view.bufferByteOffset,
      byteLength: view.byteLength,
    },
    view.byteLength,
    limits,
  );
}

function physicalPackedRangeBinding(
  bindingIndex: number,
  view: Qwen35PhysicalRowView,
  relativeByteOffset: number,
  payloadBytes: number,
  limits: Qwen35ForwardDeviceLimits,
): { readonly binding: Qwen35BufferBinding; readonly wordOffset: number } {
  const alignment = limits.minStorageBufferOffsetAlignment;
  if (
    !positiveLimit(alignment) ||
    view.bufferByteOffset % alignment !== 0 ||
    !Number.isSafeInteger(relativeByteOffset) ||
    relativeByteOffset < 0 ||
    !Number.isSafeInteger(payloadBytes) ||
    payloadBytes <= 0 ||
    relativeByteOffset + payloadBytes > view.byteLength
  ) {
    throw diagnosticError(
      "forward-binding-invalid",
      "A packed Qwen3.5 weight range cannot be bound on this device",
    );
  }
  const alignedRelativeOffset =
    Math.floor(relativeByteOffset / alignment) * alignment;
  const prefixBytes = relativeByteOffset - alignedRelativeOffset;
  const requiredBytes = prefixBytes + payloadBytes;
  if (prefixBytes % 4 !== 0 || !Number.isSafeInteger(requiredBytes)) {
    throw diagnosticError(
      "forward-binding-invalid",
      "A packed Qwen3.5 weight range cannot use a word offset",
    );
  }
  return Object.freeze({
    binding: binding(
      bindingIndex,
      "storage",
      {
        buffer: view.buffer,
        offset: view.bufferByteOffset + alignedRelativeOffset,
        byteLength: view.byteLength - alignedRelativeOffset,
      },
      requiredBytes,
      limits,
    ),
    wordOffset: prefixBytes / 4,
  });
}

function bindingsOverlap(
  left: Qwen35BufferBinding,
  right: Qwen35BufferBinding,
): boolean {
  if (left.buffer !== right.buffer) {
    return false;
  }
  return (
    left.offset < right.offset + right.size &&
    right.offset < left.offset + left.size
  );
}

function requireWritableOutputDisjoint(
  output: Qwen35BufferBinding,
  reads: readonly Qwen35BufferBinding[],
): void {
  if (reads.some((read) => bindingsOverlap(output, read))) {
    throw diagnosticError(
      "forward-buffer-alias-invalid",
      "A Qwen3.5 writable output overlaps a read binding",
    );
  }
}

function requireDistinctUniformRanges(
  uniforms: readonly Qwen35BufferBinding[],
): void {
  for (let index = 0; index < uniforms.length; index += 1) {
    for (let prior = 0; prior < index; prior += 1) {
      if (bindingsOverlap(uniforms[index]!, uniforms[prior]!)) {
        throw diagnosticError(
          "forward-uniform-alias-invalid",
          "Batched Qwen3.5 dispatches require distinct uniform ranges",
        );
      }
    }
  }
}

function dispatchPlan(input: {
  readonly kernel: Qwen35KernelSource;
  readonly bindings: readonly Qwen35BufferBinding[];
  readonly uniformWords: readonly number[];
  readonly workgroups: { readonly x: number; readonly y: number; readonly z: number };
}): Qwen35ForwardDispatchPlan {
  return Object.freeze({
    kernel: input.kernel,
    bindings: Object.freeze([...input.bindings]),
    uniformWords: Object.freeze([...input.uniformWords]),
    workgroups: Object.freeze({ ...input.workgroups }),
  });
}

function logitsPieces(
  views: readonly Qwen35PhysicalRowView[],
  vocabularyRows: number,
  rowBytes: number,
  limits: Qwen35ForwardDeviceLimits,
): readonly {
  readonly view: Qwen35PhysicalRowView;
  readonly firstLocalRow: number;
  readonly rowCount: number;
  readonly tileIndex: number;
  readonly vocabularyStart: number;
  readonly tileRows: number;
  readonly pieceOutputOffset: number;
  readonly completesTile: boolean;
}[] {
  const alignment = limits.minStorageBufferOffsetAlignment;
  const maximum = limits.maxStorageBufferBindingSize;
  if (!positiveLimit(alignment) || !positiveLimit(maximum)) {
    throw diagnosticError(
      "forward-binding-invalid",
      "A Qwen3.5 forward buffer binding is invalid for this device",
    );
  }
  const pieces: {
    readonly view: Qwen35PhysicalRowView;
    readonly firstLocalRow: number;
    readonly rowCount: number;
    readonly tileIndex: number;
    readonly vocabularyStart: number;
    readonly tileRows: number;
    readonly pieceOutputOffset: number;
    readonly completesTile: boolean;
  }[] = [];
  let viewIndex = 0;
  for (
    let vocabularyStart = 0, tileIndex = 0;
    vocabularyStart < vocabularyRows;
    vocabularyStart += MAX_LOGITS_TILE_ROWS, tileIndex += 1
  ) {
    const tileRows = Math.min(
      MAX_LOGITS_TILE_ROWS,
      vocabularyRows - vocabularyStart,
    );
    const tileEnd = vocabularyStart + tileRows;
    let pieceStart = vocabularyStart;
    while (pieceStart < tileEnd) {
      while (
        viewIndex < views.length &&
        pieceStart >= views[viewIndex]!.firstRow + views[viewIndex]!.rowCount
      ) {
        viewIndex += 1;
      }
      const view = views[viewIndex];
      if (
        view === undefined ||
        pieceStart < view.firstRow ||
        view.bufferByteOffset % alignment !== 0
      ) {
        throw diagnosticError(
          "forward-weight-views-invalid",
          "A Qwen3.5 logits tile has no valid physical weight range",
        );
      }
      const firstLocalRow = pieceStart - view.firstRow;
      const relativeByteOffset = firstLocalRow * rowBytes;
      const alignedRelativeOffset =
        Math.floor(relativeByteOffset / alignment) * alignment;
      const prefixBytes = relativeByteOffset - alignedRelativeOffset;
      const maximumPayloadBytes = maximum - prefixBytes;
      const maximumPieceRows = Math.floor(maximumPayloadBytes / rowBytes);
      const physicalEnd = Math.min(
        tileEnd,
        view.firstRow + view.rowCount,
      );
      const rowCount = Math.min(
        physicalEnd - pieceStart,
        maximumPieceRows,
      );
      if (rowCount < 1) {
        throw diagnosticError(
          "forward-binding-invalid",
          "A Qwen3.5 logits row exceeds the device storage binding limit",
        );
      }
      const nextPieceStart = pieceStart + rowCount;
      pieces.push(Object.freeze({
        view,
        firstLocalRow,
        rowCount,
        tileIndex,
        vocabularyStart,
        tileRows,
        pieceOutputOffset: pieceStart - vocabularyStart,
        completesTile: nextPieceStart === tileEnd,
      }));
      pieceStart = nextPieceStart;
    }
  }
  return Object.freeze(pieces);
}

function embeddingKernel(
  tensor: Qwen35TensorWeightView,
): { readonly layout: GemvLayout; readonly kernel: PackedEmbeddingKernel; readonly source: Qwen35KernelSource } {
  const layout = layoutOf(tensor);
  const kernel = PACKED_EMBEDDING_KERNELS.find(
    (candidate) => candidate.storageType === layout,
  );
  const source = EMBEDDING_KERNEL_SOURCES.get(layout);
  if (kernel === undefined || source === undefined) {
    throw diagnosticError(
      "forward-weight-layout-invalid",
      "Qwen3.5 packed embedding kernel is unavailable",
    );
  }
  return { layout, kernel, source };
}

/** Plans one direct packed token-row decode from the tied embedding table. */
export function planQwen35PackedEmbeddingDispatch(input: {
  readonly weights: Qwen35WeightDirectoryView;
  readonly tokenId: number;
  readonly output: Qwen35ForwardBufferSlice;
  readonly uniform: Qwen35ForwardBufferSlice;
  readonly limits: Qwen35ForwardDeviceLimits;
}): Qwen35ForwardDispatchPlan {
  const tensor = requireTensor(input.weights, "token_embd.weight");
  const shape = matrixShape(tensor);
  requireTiedEmbeddingShape(shape);
  const selected = embeddingKernel(tensor);
  const rowBytes = expectedRowBytes(tensor, shape.columns);
  const views = physicalRows(tensor, shape.rows, rowBytes);
  let logical: ReturnType<typeof planPackedEmbeddingRow>;
  try {
    logical = planPackedEmbeddingRow({
      ggmlType: selected.kernel.ggmlType,
      storageType: selected.layout,
      tokenId: input.tokenId,
      vocabSize: shape.rows,
      embeddingLength: shape.columns,
      maxComputeWorkgroupsPerDimension:
        input.limits.maxComputeWorkgroupsPerDimension,
    });
  } catch {
    throw diagnosticError(
      "forward-embedding-invalid",
      "Qwen3.5 packed embedding dispatch is invalid",
    );
  }
  const view = views.find(
    (candidate) =>
      input.tokenId >= candidate.firstRow &&
      input.tokenId < candidate.firstRow + candidate.rowCount,
  );
  if (view === undefined) {
    throw diagnosticError(
      "forward-weight-views-invalid",
      "Qwen3.5 token row has no physical weight buffer",
    );
  }
  const relativeByteOffset = logical.packedByteOffset - view.tensorByteOffset;
  if (
    relativeByteOffset < 0 ||
    relativeByteOffset + logical.packedByteLength > view.byteLength ||
    relativeByteOffset % 4 !== 0
  ) {
    throw diagnosticError(
      "forward-weight-views-invalid",
      "Qwen3.5 token row crosses a physical weight buffer",
    );
  }
  const packedRange = physicalPackedRangeBinding(
    selected.kernel.abi.bindings.packedTable,
    view,
    relativeByteOffset,
    logical.packedByteLength,
    input.limits,
  );
  const outputBinding = binding(
    selected.kernel.abi.bindings.output,
    "storage",
    input.output,
    shape.columns * 4,
    input.limits,
  );
  const uniformBinding = binding(
    selected.kernel.abi.bindings.uniforms,
    "uniform",
    input.uniform,
    selected.kernel.abi.uniformWords * 4,
    input.limits,
  );
  requireWritableOutputDisjoint(outputBinding, [
    packedRange.binding,
    uniformBinding,
  ]);
  return dispatchPlan({
    kernel: selected.source,
    bindings: [
      packedRange.binding,
      outputBinding,
      uniformBinding,
    ],
    uniformWords: [
      packedRange.wordOffset,
      logical.outputElements,
      shape.columns / selected.kernel.abi.valuesPerBlock,
      0,
    ],
    workgroups: logical.workgroups,
  });
}

/** Plans one exact Q6_K row already staged in the bounded tied-input cache. */
export function planQwen35StagedPackedEmbeddingDispatch(input: {
  readonly rows: Qwen35StagedPackedRows;
  readonly output: Qwen35ForwardBufferSlice;
  readonly uniform: Qwen35ForwardBufferSlice;
  readonly limits: Qwen35ForwardDeviceLimits;
}): Qwen35ForwardDispatchPlan {
  const selected = PACKED_EMBEDDING_KERNELS.find(
    (candidate) => candidate.storageType === "q6-k-212",
  );
  const source = EMBEDDING_KERNEL_SOURCES.get("q6-k-212");
  if (
    selected === undefined ||
    source === undefined ||
    input.rows.tensorName !== "token_embd.weight" ||
    input.rows.storageType !== "q6-k-212" ||
    input.rows.rowCount !== 1 ||
    input.rows.rowBytes !== 2_120 ||
    input.rows.byteLength !== 2_120 ||
    !Number.isSafeInteger(input.rows.firstRow) ||
    input.rows.firstRow < 0 ||
    input.rows.firstRow >= QWEN35_VOCABULARY_SIZE ||
    !Number.isSafeInteger(input.rows.bufferOffset) ||
    input.rows.bufferOffset < 0 ||
    input.rows.bufferOffset % 4 !== 0 ||
    !Number.isSafeInteger(input.rows.bufferOffset + input.rows.byteLength)
  ) {
    throw diagnosticError(
      "forward-staged-embedding-invalid",
      "The staged Qwen3.5 embedding row is invalid",
    );
  }
  let logical: ReturnType<typeof planPackedEmbeddingRow>;
  try {
    logical = planPackedEmbeddingRow({
      ggmlType: selected.ggmlType,
      storageType: selected.storageType,
      tokenId: 0,
      vocabSize: 1,
      embeddingLength: QWEN35_HIDDEN_SIZE,
      maxComputeWorkgroupsPerDimension:
        input.limits.maxComputeWorkgroupsPerDimension,
    });
  } catch {
    throw diagnosticError(
      "forward-staged-embedding-invalid",
      "The staged Qwen3.5 embedding row is invalid",
    );
  }
  const alignment = input.limits.minStorageBufferOffsetAlignment;
  if (!positiveLimit(alignment)) {
    throw diagnosticError(
      "forward-binding-invalid",
      "A packed Qwen3.5 weight range cannot be bound on this device",
    );
  }
  const alignedOffset = Math.floor(input.rows.bufferOffset / alignment) * alignment;
  const prefixBytes = input.rows.bufferOffset - alignedOffset;
  // Cache slots are row-aligned, not WebGPU-binding-aligned. Bind the safe
  // prefix and pass its u32 distance to the existing Q6_K kernel ABI.
  const stagedView: Qwen35PhysicalRowView = Object.freeze({
    buffer: input.rows.buffer,
    firstRow: 0,
    rowCount: 1,
    tensorByteOffset: 0,
    bufferByteOffset: alignedOffset,
    byteLength: prefixBytes + input.rows.rowBytes,
  });
  const packedRange = physicalPackedRangeBinding(
    selected.abi.bindings.packedTable,
    stagedView,
    prefixBytes,
    input.rows.rowBytes,
    input.limits,
  );
  const outputBinding = binding(
    selected.abi.bindings.output,
    "storage",
    input.output,
    QWEN35_HIDDEN_SIZE * 4,
    input.limits,
  );
  const uniformBinding = binding(
    selected.abi.bindings.uniforms,
    "uniform",
    input.uniform,
    selected.abi.uniformWords * 4,
    input.limits,
  );
  requireWritableOutputDisjoint(outputBinding, [
    packedRange.binding,
    uniformBinding,
  ]);
  return dispatchPlan({
    kernel: source,
    bindings: [packedRange.binding, outputBinding, uniformBinding],
    uniformWords: [
      packedRange.wordOffset,
      logical.outputElements,
      QWEN35_HIDDEN_SIZE / selected.abi.valuesPerBlock,
      0,
    ],
    workgroups: logical.workgroups,
  });
}

function gemvKernel(
  tensor: Qwen35TensorWeightView,
): { readonly layout: GemvLayout; readonly kernel: LanguageGemvKernel; readonly source: Qwen35KernelSource } {
  const layout = layoutOf(tensor);
  const kernel = LANGUAGE_GEMV_KERNELS.find(
    (candidate) => candidate.layout === layout,
  );
  const source = GEMV_KERNEL_SOURCES.get(layout);
  if (kernel === undefined || source === undefined) {
    throw diagnosticError(
      "forward-weight-layout-invalid",
      "Qwen3.5 packed GEMV kernel is unavailable",
    );
  }
  return { layout, kernel, source };
}

/** Copies one projected visual token into the language hidden workspace. */
export function planQwen35VisualEmbeddingDispatch(input: {
  readonly source: Qwen35ForwardBufferSlice;
  readonly output: Qwen35ForwardBufferSlice;
  readonly uniform: Qwen35ForwardBufferSlice;
  readonly limits: Qwen35ForwardDeviceLimits;
}): Qwen35ForwardDispatchPlan {
  const elements = QWEN35_HIDDEN_SIZE;
  const sourceBinding = binding(0, "storage", input.source, elements * 4, input.limits);
  const outputBinding = binding(1, "storage", input.output, elements * 4, input.limits);
  const uniformBinding = binding(2, "uniform", input.uniform, 16, input.limits);
  requireWritableOutputDisjoint(outputBinding, [sourceBinding, uniformBinding]);
  return dispatchPlan({
    kernel: VISUAL_EMBEDDING_KERNEL,
    bindings: [sourceBinding, outputBinding, uniformBinding],
    uniformWords: [elements, 0, 0, 0],
    workgroups: { x: Math.ceil(elements / 256), y: 1, z: 1 },
  });
}

function tiedLogitsGeometryData(input: {
  readonly weights: Qwen35WeightDirectoryView;
  readonly limits: Qwen35ForwardDeviceLimits;
}): {
  readonly shape: { readonly columns: number; readonly rows: number };
  readonly selected: ReturnType<typeof gemvKernel>;
  readonly rowBytes: number;
  readonly pieces: ReturnType<typeof logitsPieces>;
} {
  const tensor = requireTensor(input.weights, "token_embd.weight");
  const shape = matrixShape(tensor);
  requireTiedEmbeddingShape(shape);
  const selected = gemvKernel(tensor);
  const rowBytes = expectedRowBytes(tensor, shape.columns);
  const views = physicalRows(tensor, shape.rows, rowBytes);
  const pieces = logitsPieces(
    views,
    QWEN35_DECODABLE_LOGIT_ROWS,
    rowBytes,
    input.limits,
  );
  return { shape, selected, rowBytes, pieces };
}

/** Derives uniform capacity from physical rows before uniform allocation. */
export function planQwen35TiedLogitsGeometry(input: {
  readonly weights: Qwen35WeightDirectoryView;
  readonly limits: Qwen35ForwardDeviceLimits;
}): Qwen35TiedLogitsGeometry {
  const data = tiedLogitsGeometryData(input);
  return Object.freeze({
    modelRows: QWEN35_VOCABULARY_SIZE,
    decodableRows: QWEN35_DECODABLE_LOGIT_ROWS,
    logicalTileRows: MAX_LOGITS_TILE_ROWS,
    mathematicalTileCount: QWEN35_MATHEMATICAL_LOGITS_TILES,
    finalTileRows: QWEN35_FINAL_LOGITS_TILE_ROWS,
    physicalPieceCount: data.pieces.length,
    reductionDispatchCount: QWEN35_LOGITS_REDUCTION_DISPATCHES,
    uniformCount: data.pieces.length + QWEN35_LOGITS_REDUCTION_DISPATCHES,
  });
}

/** Plans one packed GEMV request for each row-sharded physical weight buffer. */
export function planQwen35PackedGemvDispatches(input: {
  readonly weights: Qwen35WeightDirectoryView;
  readonly tensorName: string;
  readonly activation: Qwen35ForwardBufferSlice;
  readonly output: Qwen35ForwardBufferSlice;
  readonly uniforms: readonly Qwen35ForwardBufferSlice[];
  readonly limits: Qwen35ForwardDeviceLimits;
}): readonly Qwen35ForwardDispatchPlan[] {
  if (input.tensorName === "token_embd.weight") {
    throw diagnosticError(
      "forward-tied-logits-required",
      "The tied embedding projection requires bounded logits tiles",
    );
  }
  const tensor = requireTensor(input.weights, input.tensorName);
  const shape = matrixShape(tensor);
  const selected = gemvKernel(tensor);
  const rowBytes = expectedRowBytes(tensor, shape.columns);
  const views = physicalRows(tensor, shape.rows, rowBytes);
  if (input.uniforms.length !== views.length) {
    throw diagnosticError(
      "forward-uniform-count-invalid",
      "Qwen3.5 GEMV requires one explicit uniform slot per physical row range",
    );
  }
  const uniformBindings = input.uniforms.map((uniform) => binding(
    selected.kernel.abi.bindings.uniforms,
    "uniform",
    uniform,
    selected.kernel.abi.uniformWords * 4,
    input.limits,
  ));
  requireDistinctUniformRanges(uniformBindings);
  const activationBinding = binding(
    selected.kernel.abi.bindings.activation,
    "storage",
    input.activation,
    shape.columns * 4,
    input.limits,
  );
  const outputBinding = binding(
    selected.kernel.abi.bindings.output,
    "storage",
    input.output,
    shape.rows * 4,
    input.limits,
  );
  const plans = views.map((view, index) => {
    let logical: ReturnType<typeof planGemvDispatch>;
    try {
      logical = planGemvDispatch({
        layout: selected.layout,
        ggmlType: selected.kernel.ggmlType,
        localRows: view.rowCount,
        columns: shape.columns,
        outputRowOffset: view.firstRow,
        maxWorkgroupsPerDimension:
          input.limits.maxComputeWorkgroupsPerDimension,
      });
    } catch {
      throw diagnosticError(
        "forward-gemv-invalid",
        "Qwen3.5 packed GEMV dispatch is invalid",
      );
    }
    const weightBinding = physicalBinding(
      selected.kernel.abi.bindings.packedWeights,
      view,
      input.limits,
    );
    const uniformBinding = uniformBindings[index]!;
    requireWritableOutputDisjoint(outputBinding, [
      weightBinding,
      activationBinding,
      uniformBinding,
    ]);
    return dispatchPlan({
      kernel: selected.source,
      bindings: [
        weightBinding,
        activationBinding,
        outputBinding,
        uniformBinding,
      ],
      uniformWords: [
        logical.uniforms.localRows,
        logical.uniforms.columns,
        logical.uniforms.blocksPerRow,
        0,
        logical.uniforms.outputRowOffset,
      ],
      workgroups: logical.workgroups,
    });
  });
  return Object.freeze(plans);
}

/** Logits reuse token_embd.weight; the package has no second output matrix. */
export function planQwen35TiedLogitsDispatches(input: Omit<
  Parameters<typeof planQwen35PackedGemvDispatches>[0],
  "tensorName"
>): readonly Qwen35TiedLogitsDispatchPlan[] {
  const { shape, selected, rowBytes, pieces } = tiedLogitsGeometryData(input);
  if (input.uniforms.length !== pieces.length) {
    throw diagnosticError(
      "forward-uniform-count-invalid",
      "Qwen3.5 tied logits require one explicit uniform slot per piece",
    );
  }
  const uniformBindings = input.uniforms.map((uniform) => binding(
    selected.kernel.abi.bindings.uniforms,
    "uniform",
    uniform,
    selected.kernel.abi.uniformWords * 4,
    input.limits,
  ));
  requireDistinctUniformRanges(uniformBindings);
  const activationBinding = binding(
    selected.kernel.abi.bindings.activation,
    "storage",
    input.activation,
    shape.columns * 4,
    input.limits,
  );
  const plans = pieces.map((piece, index) => {
    let logical: ReturnType<typeof planGemvDispatch>;
    try {
      logical = planGemvDispatch({
        layout: selected.layout,
        ggmlType: selected.kernel.ggmlType,
        localRows: piece.rowCount,
        columns: shape.columns,
        outputRowOffset: piece.pieceOutputOffset,
        maxWorkgroupsPerDimension:
          input.limits.maxComputeWorkgroupsPerDimension,
      });
    } catch {
      throw diagnosticError(
        "forward-gemv-invalid",
        "Qwen3.5 tied-logits GEMV dispatch is invalid",
      );
    }
    const outputBinding = binding(
      selected.kernel.abi.bindings.output,
      "storage",
      input.output,
      piece.tileRows * 4,
      input.limits,
    );
    const relativeByteOffset = piece.firstLocalRow * rowBytes;
    const packedRange = physicalPackedRangeBinding(
      selected.kernel.abi.bindings.packedWeights,
      piece.view,
      relativeByteOffset,
      piece.rowCount * rowBytes,
      input.limits,
    );
    const uniformBinding = uniformBindings[index]!;
    requireWritableOutputDisjoint(outputBinding, [
      packedRange.binding,
      activationBinding,
      uniformBinding,
    ]);
    const base = dispatchPlan({
      kernel: selected.source,
      bindings: [
        packedRange.binding,
        activationBinding,
        outputBinding,
        uniformBinding,
      ],
      uniformWords: [
        logical.uniforms.localRows,
        logical.uniforms.columns,
        logical.uniforms.blocksPerRow,
        packedRange.wordOffset,
        logical.uniforms.outputRowOffset,
      ],
      workgroups: logical.workgroups,
    });
    return Object.freeze({
      ...base,
      tileIndex: piece.tileIndex,
      vocabularyStart: piece.vocabularyStart,
      tileRows: piece.tileRows,
      pieceOutputOffset: piece.pieceOutputOffset,
      pieceRows: piece.rowCount,
      completesTile: piece.completesTile,
    });
  });
  return Object.freeze(plans);
}

/** Plans one complete logical logits tile from the bounded Q6_K output cache. */
export function planQwen35StagedTiedLogitsDispatch(input: {
  readonly tile: Qwen35StagedPackedRows;
  readonly activation: Qwen35ForwardBufferSlice;
  readonly output: Qwen35ForwardBufferSlice;
  readonly uniform: Qwen35ForwardBufferSlice;
  readonly limits: Qwen35ForwardDeviceLimits;
}): Qwen35TiedLogitsDispatchPlan {
  const selected = LANGUAGE_GEMV_KERNELS.find(
    (candidate) => candidate.layout === "q6-k-212",
  );
  const source = GEMV_KERNEL_SOURCES.get("q6-k-212");
  const tileIndex = input.tile.firstRow / MAX_LOGITS_TILE_ROWS;
  const expectedRows = Math.min(
    MAX_LOGITS_TILE_ROWS,
    QWEN35_DECODABLE_LOGIT_ROWS - input.tile.firstRow,
  );
  if (
    selected === undefined ||
    source === undefined ||
    input.tile.tensorName !== "token_embd.weight" ||
    input.tile.storageType !== "q6-k-212" ||
    input.tile.rowBytes !== 2_120 ||
    !Number.isSafeInteger(input.tile.firstRow) ||
    input.tile.firstRow < 0 ||
    input.tile.firstRow >= QWEN35_DECODABLE_LOGIT_ROWS ||
    input.tile.firstRow % MAX_LOGITS_TILE_ROWS !== 0 ||
    !Number.isSafeInteger(tileIndex) ||
    input.tile.rowCount !== expectedRows ||
    input.tile.byteLength !== input.tile.rowCount * input.tile.rowBytes ||
    !Number.isSafeInteger(input.tile.bufferOffset) ||
    input.tile.bufferOffset < 0 ||
    input.tile.bufferOffset % 4 !== 0 ||
    !Number.isSafeInteger(input.tile.bufferOffset + input.tile.byteLength)
  ) {
    throw diagnosticError(
      "forward-staged-logits-invalid",
      "The staged Qwen3.5 logits tile is invalid",
    );
  }
  let logical: ReturnType<typeof planGemvDispatch>;
  try {
    logical = planGemvDispatch({
      layout: selected.layout,
      ggmlType: selected.ggmlType,
      localRows: input.tile.rowCount,
      columns: QWEN35_HIDDEN_SIZE,
      outputRowOffset: 0,
      maxWorkgroupsPerDimension:
        input.limits.maxComputeWorkgroupsPerDimension,
    });
  } catch {
    throw diagnosticError(
      "forward-staged-logits-invalid",
      "The staged Qwen3.5 logits tile is invalid",
    );
  }
  const alignment = input.limits.minStorageBufferOffsetAlignment;
  if (!positiveLimit(alignment)) {
    throw diagnosticError(
      "forward-binding-invalid",
      "A packed Qwen3.5 weight range cannot be bound on this device",
    );
  }
  const alignedOffset = Math.floor(input.tile.bufferOffset / alignment) * alignment;
  const prefixBytes = input.tile.bufferOffset - alignedOffset;
  const stagedView: Qwen35PhysicalRowView = Object.freeze({
    buffer: input.tile.buffer,
    firstRow: 0,
    rowCount: input.tile.rowCount,
    tensorByteOffset: 0,
    bufferByteOffset: alignedOffset,
    byteLength: prefixBytes + input.tile.byteLength,
  });
  const packedRange = physicalPackedRangeBinding(
    selected.abi.bindings.packedWeights,
    stagedView,
    prefixBytes,
    input.tile.byteLength,
    input.limits,
  );
  const activationBinding = binding(
    selected.abi.bindings.activation,
    "storage",
    input.activation,
    QWEN35_HIDDEN_SIZE * 4,
    input.limits,
  );
  const outputBinding = binding(
    selected.abi.bindings.output,
    "storage",
    input.output,
    input.tile.rowCount * 4,
    input.limits,
  );
  const uniformBinding = binding(
    selected.abi.bindings.uniforms,
    "uniform",
    input.uniform,
    selected.abi.uniformWords * 4,
    input.limits,
  );
  requireWritableOutputDisjoint(outputBinding, [
    packedRange.binding,
    activationBinding,
    uniformBinding,
  ]);
  const base = dispatchPlan({
    kernel: source,
    bindings: [
      packedRange.binding,
      activationBinding,
      outputBinding,
      uniformBinding,
    ],
    uniformWords: [
      logical.uniforms.localRows,
      logical.uniforms.columns,
      logical.uniforms.blocksPerRow,
      packedRange.wordOffset,
      0,
    ],
    workgroups: logical.workgroups,
  });
  return Object.freeze({
    ...base,
    tileIndex,
    vocabularyStart: input.tile.firstRow,
    tileRows: input.tile.rowCount,
    pieceOutputOffset: 0,
    pieceRows: input.tile.rowCount,
    completesTile: true,
  });
}
