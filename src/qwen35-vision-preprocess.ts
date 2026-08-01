import { diagnosticError } from "./diagnostics.js";
import type { VisionProcessorSettings } from "./manifest.js";

/** The first product setting keeps one image near 1,024 post-merger tokens. */
export const QWEN35_DEFAULT_MAX_VISUAL_TOKENS = 1_024;

const MAX_SOURCE_EDGE = 16_777_216;
const MAX_PRODUCT_VISUAL_TOKENS = 16_384;
const MAX_ASPECT_RATIO = 200;
const MAX_ACCUMULATOR_BYTES = 768 * 1024;
const VISION_WORK_SLICE_MILLISECONDS = 12;

/** Default decoded-image staging policy; callers may raise it after device measurement. */
export const QWEN35_DEFAULT_DECODED_SOURCE_BYTE_BUDGET = 64 * 1024 * 1024;

export interface Qwen35VisionMaterializationEstimate {
  readonly sourceRgbBytes: number;
  readonly resizedRgbBytes: number;
  readonly patchBytes: number;
  /** Source RGB plus resized RGB plus packed patches; this is diagnostic, not a rejection limit. */
  readonly estimatedPeakMaterializedBytes: number;
}

export interface Qwen35VisionTaskScheduler {
  readonly now: () => number;
  readonly yieldToBrowser: () => Promise<void>;
}

export interface Qwen35VisionImagePlan {
  readonly sourceHeight: number;
  readonly sourceWidth: number;
  readonly factor: number;
  readonly resizedHeight: number;
  readonly resizedWidth: number;
  readonly effectiveMaxPixels: number;
  readonly gridTHW: readonly [1, number, number];
  readonly projectedVisualTokens: number;
  readonly patchCount: number;
  readonly patchVectorLength: number;
  readonly packedFloatCount: number;
  readonly requiresBicubicResampling: boolean;
  readonly materialization: Qwen35VisionMaterializationEstimate;
  /** This oracle plans Qwen2-VL resize geometry but does not resample pixels. */
  readonly resampling: "external-bicubic-required";
  readonly settings: Readonly<{
    readonly patchSize: number;
    readonly temporalPatchSize: number;
    readonly mergeSize: number;
    readonly shortestEdge: number;
    readonly longestEdge: number;
    readonly imageMean: readonly [number, number, number];
    readonly imageStd: readonly [number, number, number];
  }>;
}

export interface Qwen35VisionPatchBatch {
  readonly gridTHW: readonly [1, number, number];
  readonly projectedVisualTokens: number;
  readonly patchVectorLength: number;
  /** Mutable typed output is unavoidable; it does not alias caller RGB bytes. */
  readonly patches: Float32Array;
  /** Identifies whether the caller supplied resized bytes or used the browser oracle. */
  readonly resampling:
    | "caller-supplied-resized-rgb"
    | "qwen2-vl-fast-bicubic";
  readonly preprocessingMetrics?: Readonly<{
    readonly resize: Qwen35VisionResizeMetrics;
    readonly packingYieldCount: number;
    readonly resizeOutputAliasesSource: boolean;
    readonly estimatedPeakMaterializedBytes: number;
  }>;
}

export interface Qwen35VisionResizeMetrics {
  /** Target-sized coefficient tables are intentionally absent. */
  readonly coefficientTableBytes: 0;
  readonly horizontalRowSegmentsComputed: number;
  readonly horizontalSamplesComputed: number;
  /** Tracked temporary typed arrays only; excludes outputs, JS objects, and allocation churn. */
  readonly peakTemporaryBytes: number;
  readonly yieldCount: number;
  readonly outputAliasesSource: boolean;
}

export interface Qwen35VisionRgbResize {
  /** Aliases source only when the caller explicitly requests borrow-source for a no-op resize. */
  readonly rgb: Uint8Array;
  readonly metrics: Qwen35VisionResizeMetrics;
}

function materializationEstimate(
  sourceHeight: number,
  sourceWidth: number,
  resizedHeight: number,
  resizedWidth: number,
  packedFloatCount: number,
): Qwen35VisionMaterializationEstimate {
  const sourceRgbBytes = sourceHeight * sourceWidth * 3;
  const resizedRgbBytes = resizedHeight * resizedWidth * 3;
  const patchBytes = packedFloatCount * Float32Array.BYTES_PER_ELEMENT;
  // A borrowed no-op resize aliases the decoded source, so counting both would
  // report memory that is never live at the same time.
  const estimatedPeakMaterializedBytes =
    sourceRgbBytes + (sourceHeight === resizedHeight && sourceWidth === resizedWidth ? 0 : resizedRgbBytes) + patchBytes;
  if ([sourceRgbBytes, resizedRgbBytes, patchBytes, estimatedPeakMaterializedBytes].some(
    (value) => !Number.isSafeInteger(value),
  )) {
    fail("vision-resize-plan-invalid", "Vision resize plan is invalid");
  }
  return Object.freeze({ sourceRgbBytes, resizedRgbBytes, patchBytes, estimatedPeakMaterializedBytes });
}

interface SnapshotSettings {
  readonly patchSize: number;
  readonly temporalPatchSize: number;
  readonly mergeSize: number;
  readonly shortestEdge: number;
  readonly longestEdge: number;
  readonly imageMean: readonly [number, number, number];
  readonly imageStd: readonly [number, number, number];
}

function fail(code: string, message: string): never {
  throw diagnosticError(code, message);
}

function positiveInteger(value: unknown): value is number {
  return Number.isSafeInteger(value) && (value as number) > 0;
}

function finiteChannelTriplet(value: unknown): value is readonly [number, number, number] {
  return Array.isArray(value) && value.length === 3 && value.every(
    (channel) => typeof channel === "number" && Number.isFinite(channel),
  );
}

function snapshotSettings(input: Pick<
  VisionProcessorSettings,
  | "patchSize"
  | "temporalPatchSize"
  | "mergeSize"
  | "shortestEdge"
  | "longestEdge"
  | "imageMean"
  | "imageStd"
>): SnapshotSettings {
  if (
    !positiveInteger(input.patchSize) ||
    !positiveInteger(input.temporalPatchSize) ||
    !positiveInteger(input.mergeSize) ||
    !positiveInteger(input.shortestEdge) ||
    !positiveInteger(input.longestEdge) ||
    input.longestEdge > MAX_SOURCE_EDGE ||
    input.shortestEdge > input.longestEdge ||
    !finiteChannelTriplet(input.imageMean) ||
    !finiteChannelTriplet(input.imageStd) ||
    input.imageStd.some((channel) => channel === 0)
  ) {
    fail("vision-settings-invalid", "Vision preprocessing settings are invalid");
  }
  const factor = input.patchSize * input.mergeSize;
  if (
    !Number.isSafeInteger(factor) ||
    factor < 1 ||
    factor > 65_536 ||
    input.temporalPatchSize > 64
  ) {
    fail("vision-settings-invalid", "Vision preprocessing settings are invalid");
  }
  return Object.freeze({
    patchSize: input.patchSize,
    temporalPatchSize: input.temporalPatchSize,
    mergeSize: input.mergeSize,
    shortestEdge: input.shortestEdge,
    longestEdge: input.longestEdge,
    imageMean: Object.freeze([...input.imageMean]) as readonly [number, number, number],
    imageStd: Object.freeze([...input.imageStd]) as readonly [number, number, number],
  });
}

function roundHalfEven(value: number): number {
  const lower = Math.floor(value);
  const fraction = value - lower;
  if (fraction < 0.5) return lower;
  if (fraction > 0.5) return lower + 1;
  return lower % 2 === 0 ? lower : lower + 1;
}

function boundedSourceDimension(value: unknown): number {
  if (!positiveInteger(value) || value > MAX_SOURCE_EDGE) {
    fail("vision-source-dimensions-invalid", "Vision source dimensions are invalid");
  }
  return value;
}

function productMaxPixels(
  settings: SnapshotSettings,
  maxVisualTokens: number,
): number {
  if (
    !positiveInteger(maxVisualTokens) ||
    maxVisualTokens > MAX_PRODUCT_VISUAL_TOKENS
  ) {
    fail("vision-product-cap-invalid", "Vision product token cap is invalid");
  }
  const pixelsPerVisualToken =
    settings.patchSize * settings.patchSize * settings.mergeSize * settings.mergeSize;
  const requestedPixels = maxVisualTokens * pixelsPerVisualToken;
  if (
    !Number.isSafeInteger(requestedPixels) ||
    requestedPixels < settings.shortestEdge
  ) {
    fail("vision-product-cap-invalid", "Vision product token cap is invalid");
  }
  return Math.min(settings.longestEdge, requestedPixels);
}

/**
 * Matches the Qwen2-VL smart-resize geometry exactly. The caller still owns
 * resampling and must apply the returned dimensions with bicubic filtering.
 */
export function smartResizeQwen2Vl(input: {
  readonly sourceHeight: number;
  readonly sourceWidth: number;
  readonly factor: number;
  readonly shortestEdge: number;
  readonly longestEdge: number;
}): readonly [number, number] {
  const { sourceHeight: height, sourceWidth: width, factor } = input;
  if (
    !positiveInteger(height) ||
    !positiveInteger(width) ||
    height > MAX_SOURCE_EDGE ||
    width > MAX_SOURCE_EDGE ||
    !positiveInteger(factor) ||
    factor > 65_536 ||
    !positiveInteger(input.shortestEdge) ||
    !positiveInteger(input.longestEdge) ||
    input.shortestEdge > input.longestEdge ||
    input.longestEdge > MAX_SOURCE_EDGE
  ) {
    fail("vision-resize-plan-invalid", "Vision resize plan is invalid");
  }
  if (
    Math.max(height, width) / Math.min(height, width) > MAX_ASPECT_RATIO
  ) {
    fail("vision-aspect-ratio-invalid", "Vision source geometry is unsupported");
  }
  let resizedHeight = roundHalfEven(height / factor) * factor;
  let resizedWidth = roundHalfEven(width / factor) * factor;
  const sourcePixels = height * width;
  const roundedPixels = resizedHeight * resizedWidth;
  if (roundedPixels > input.longestEdge) {
    const beta = Math.sqrt(sourcePixels / input.longestEdge);
    resizedHeight = Math.max(factor, Math.floor(height / beta / factor) * factor);
    resizedWidth = Math.max(factor, Math.floor(width / beta / factor) * factor);
  } else if (roundedPixels < input.shortestEdge) {
    const beta = Math.sqrt(input.shortestEdge / sourcePixels);
    resizedHeight = Math.ceil(height * beta / factor) * factor;
    resizedWidth = Math.ceil(width * beta / factor) * factor;
  }
  if (
    !positiveInteger(resizedHeight) ||
    !positiveInteger(resizedWidth) ||
    resizedHeight % factor !== 0 ||
    resizedWidth % factor !== 0
  ) {
    fail("vision-resize-plan-invalid", "Vision resize plan is invalid");
  }
  return Object.freeze([resizedHeight, resizedWidth]);
}

/**
 * Plans official Qwen2-VL smart-resize geometry from processor settings already
 * authenticated by the vision-package manifest, without decoding or resampling.
 */
export function planQwen35VisionImage(input: {
  readonly sourceHeight: number;
  readonly sourceWidth: number;
  readonly settings: VisionProcessorSettings;
  readonly maxVisualTokens?: number;
}): Qwen35VisionImagePlan {
  const settings = snapshotSettings(input.settings);
  const maxVisualTokens = input.maxVisualTokens ?? QWEN35_DEFAULT_MAX_VISUAL_TOKENS;
  const effectiveMaxPixels = productMaxPixels(settings, maxVisualTokens);
  const sourceHeight = boundedSourceDimension(input.sourceHeight);
  const sourceWidth = boundedSourceDimension(input.sourceWidth);
  const factor = settings.patchSize * settings.mergeSize;
  const [resizedHeight, resizedWidth] = smartResizeQwen2Vl({
    sourceHeight,
    sourceWidth,
    factor,
    shortestEdge: settings.shortestEdge,
    longestEdge: effectiveMaxPixels,
  });
  const gridHeight = resizedHeight / settings.patchSize;
  const gridWidth = resizedWidth / settings.patchSize;
  const projectedVisualTokens =
    (gridHeight * gridWidth) / (settings.mergeSize * settings.mergeSize);
  const patchVectorLength = 3 * settings.temporalPatchSize * settings.patchSize * settings.patchSize;
  const patchCount = gridHeight * gridWidth;
  const packedFloatCount = patchCount * patchVectorLength;
  if (
    !Number.isSafeInteger(projectedVisualTokens) ||
    !Number.isSafeInteger(patchVectorLength) ||
    !Number.isSafeInteger(patchCount) ||
    !Number.isSafeInteger(packedFloatCount) ||
    projectedVisualTokens > maxVisualTokens ||
    gridHeight % settings.mergeSize !== 0 ||
    gridWidth % settings.mergeSize !== 0
  ) {
    fail("vision-resize-plan-invalid", "Vision resize plan is invalid");
  }
  return Object.freeze({
    sourceHeight,
    sourceWidth,
    factor,
    resizedHeight,
    resizedWidth,
    effectiveMaxPixels,
    gridTHW: Object.freeze([1, gridHeight, gridWidth]) as readonly [1, number, number],
    projectedVisualTokens,
    patchCount,
    patchVectorLength,
    packedFloatCount,
    requiresBicubicResampling:
      sourceHeight !== resizedHeight || sourceWidth !== resizedWidth,
    materialization: materializationEstimate(
      sourceHeight, sourceWidth, resizedHeight, resizedWidth, packedFloatCount,
    ),
    resampling: "external-bicubic-required",
    settings,
  });
}

function validatePackPlan(plan: Qwen35VisionImagePlan): void {
  const settings = snapshotSettings(plan.settings);
  const factor = settings.patchSize * settings.mergeSize;
  const [gridT, gridHeight, gridWidth] = plan.gridTHW;
  const expectedPatchCount = gridHeight * gridWidth;
  const expectedProjectedVisualTokens = expectedPatchCount / (settings.mergeSize * settings.mergeSize);
  const expectedVectorLength = 3 * settings.temporalPatchSize * settings.patchSize * settings.patchSize;
  const expectedPackedFloatCount = expectedPatchCount * expectedVectorLength;
  const expectedSourceRgbBytes = plan.sourceHeight * plan.sourceWidth * 3;
  const expectedResizedRgbBytes = plan.resizedHeight * plan.resizedWidth * 3;
  const expectedPatchBytes = expectedPackedFloatCount * Float32Array.BYTES_PER_ELEMENT;
  const expectedPeakMaterializedBytes = expectedSourceRgbBytes +
    (plan.sourceHeight === plan.resizedHeight && plan.sourceWidth === plan.resizedWidth
      ? 0
      : expectedResizedRgbBytes) +
    expectedPatchBytes;
  if (
    plan.resampling !== "external-bicubic-required" ||
    !positiveInteger(plan.sourceHeight) ||
    !positiveInteger(plan.sourceWidth) ||
    plan.factor !== factor ||
    gridT !== 1 ||
    !positiveInteger(plan.resizedHeight) ||
    !positiveInteger(plan.resizedWidth) ||
    !positiveInteger(gridHeight) ||
    !positiveInteger(gridWidth) ||
    !Number.isSafeInteger(expectedPatchCount) ||
    !Number.isSafeInteger(expectedProjectedVisualTokens) ||
    !Number.isSafeInteger(expectedVectorLength) ||
    !Number.isSafeInteger(expectedPackedFloatCount) ||
    !Number.isSafeInteger(expectedSourceRgbBytes) ||
    !Number.isSafeInteger(expectedResizedRgbBytes) ||
    !Number.isSafeInteger(expectedPatchBytes) ||
    !Number.isSafeInteger(expectedPeakMaterializedBytes) ||
    plan.resizedHeight > MAX_SOURCE_EDGE ||
    plan.resizedWidth > MAX_SOURCE_EDGE ||
    plan.resizedHeight !== gridHeight * settings.patchSize ||
    plan.resizedWidth !== gridWidth * settings.patchSize ||
    gridHeight % settings.mergeSize !== 0 ||
    gridWidth % settings.mergeSize !== 0 ||
    plan.patchCount !== expectedPatchCount ||
    plan.projectedVisualTokens !== expectedProjectedVisualTokens ||
    plan.projectedVisualTokens > MAX_PRODUCT_VISUAL_TOKENS ||
    plan.patchVectorLength !== expectedVectorLength ||
    plan.packedFloatCount !== expectedPackedFloatCount ||
    plan.materialization?.sourceRgbBytes !== expectedSourceRgbBytes ||
    plan.materialization?.resizedRgbBytes !== expectedResizedRgbBytes ||
    plan.materialization?.patchBytes !== expectedPatchBytes ||
    plan.materialization?.estimatedPeakMaterializedBytes !== expectedPeakMaterializedBytes
  ) {
    fail("vision-patch-plan-invalid", "Vision patch plan is invalid");
  }
}

function normalizeRgb(value: number, mean: number, standardDeviation: number): number {
  // The pinned fast processor normalizes in byte space. Keep every f32 boundary
  // because dividing the byte first changes deterministic vision fixtures.
  const byte = Math.fround(value);
  const meanBytes = Math.fround(mean * 255);
  const standardDeviationBytes = Math.fround(standardDeviation * 255);
  return Math.fround(Math.fround(byte - meanBytes) / standardDeviationBytes);
}

interface BicubicAxis {
  readonly sourceSize: number;
  readonly targetSize: number;
  readonly scale: number;
  readonly support: number;
  readonly inverseScale: number;
  readonly precision: number;
}

function validateRgbGeometry(input: {
  readonly rgb: Uint8Array;
  readonly sourceHeight: number;
  readonly sourceWidth: number;
  readonly targetHeight: number;
  readonly targetWidth: number;
  readonly decodedSourceByteBudget?: number;
}): void {
  const dimensions = [
    input.sourceHeight,
    input.sourceWidth,
    input.targetHeight,
    input.targetWidth,
  ];
  const sourceBytes = input.sourceHeight * input.sourceWidth * 3;
  const targetPixels = input.targetHeight * input.targetWidth;
  const targetBytes = targetPixels * 3;
  const decodedSourceByteBudget = input.decodedSourceByteBudget
    ?? QWEN35_DEFAULT_DECODED_SOURCE_BYTE_BUDGET;
  if (!positiveInteger(decodedSourceByteBudget)) {
    fail("vision-decoded-source-budget-invalid", "Decoded source staging budget is invalid");
  }
  if (
    !(input.rgb instanceof Uint8Array) ||
    dimensions.some((dimension) => !positiveInteger(dimension) || dimension > MAX_SOURCE_EDGE) ||
    !Number.isSafeInteger(sourceBytes) ||
    !Number.isSafeInteger(targetPixels) ||
    !Number.isSafeInteger(targetBytes) ||
    targetPixels > MAX_SOURCE_EDGE ||
    Math.max(input.sourceHeight, input.sourceWidth) / Math.min(input.sourceHeight, input.sourceWidth) > MAX_ASPECT_RATIO ||
    input.rgb.byteLength !== sourceBytes
  ) {
    fail("vision-source-rgb-size-invalid", "Source RGB bytes do not match safe image geometry");
  }
  if (sourceBytes > decodedSourceByteBudget) {
    fail("vision-decoded-source-budget-exceeded", "Decoded source exceeds the configured staging budget");
  }
}

function cubicKeysFilter(value: number): number {
  const distance = Math.abs(value);
  // Torchvision selects this Keys coefficient for BICUBIC with antialias=True.
  const a = -0.5;
  if (distance < 1) {
    return (a + 2) * distance * distance * distance - (a + 3) * distance * distance + 1;
  }
  if (distance < 2) {
    return a * distance * distance * distance - 5 * a * distance * distance + 8 * a * distance - 4 * a;
  }
  return 0;
}

function roundToInt16(value: number): number {
  return Math.trunc(value < 0 ? value - 0.5 : value + 0.5);
}

class CooperativeVisionWork {
  private yieldCounter = 0;
  private sliceStartedAt: number;
  private readonly scheduler: Qwen35VisionTaskScheduler;

  constructor(
    private readonly signal: AbortSignal | undefined,
    scheduler: Qwen35VisionTaskScheduler | undefined,
  ) {
    this.scheduler = scheduler ?? {
      now: () => globalThis.performance?.now?.() ?? Date.now(),
      yieldToBrowser: () => new Promise<void>((resolve) => setTimeout(resolve, 0)),
    };
    if (typeof this.scheduler.now !== "function" || typeof this.scheduler.yieldToBrowser !== "function") {
      fail("vision-task-scheduler-invalid", "Vision task scheduler is invalid");
    }
    this.sliceStartedAt = this.readNow();
  }

  async checkpoint(): Promise<void> {
    this.throwIfCancelled();
    const now = this.readNow();
    if (now - this.sliceStartedAt < VISION_WORK_SLICE_MILLISECONDS) return;
    this.yieldCounter += 1;
    await this.scheduler.yieldToBrowser();
    this.sliceStartedAt = this.readNow();
    this.throwIfCancelled();
  }

  private readNow(): number {
    const value = this.scheduler.now();
    if (!Number.isFinite(value)) fail("vision-task-scheduler-invalid", "Vision task scheduler is invalid");
    return value;
  }

  throwIfCancelled(): void {
    if (this.signal?.aborted) {
      fail("vision-preprocess-cancelled", "Vision preprocessing was cancelled");
    }
  }

  get yields(): number {
    return this.yieldCounter;
  }
}

function bicubicTapBounds(axis: BicubicAxis, targetIndex: number): {
  readonly center: number;
  readonly firstSourceIndex: number;
  readonly count: number;
} {
  const center = axis.scale * (targetIndex + 0.5);
  const firstSourceIndex = Math.max(Math.trunc(center - axis.support + 0.5), 0);
  const count = Math.min(
    Math.trunc(center + axis.support + 0.5), axis.sourceSize,
  ) - firstSourceIndex;
  if (count < 1) {
    fail("vision-bicubic-plan-invalid", "Bicubic coefficient generation failed");
  }
  return { center, firstSourceIndex, count };
}

function bicubicTapTotal(axis: BicubicAxis, targetIndex: number): {
  readonly center: number;
  readonly firstSourceIndex: number;
  readonly count: number;
  readonly totalWeight: number;
} {
  const bounds = bicubicTapBounds(axis, targetIndex);
  let totalWeight = 0;
  for (let index = 0; index < bounds.count; index += 1) {
    totalWeight += cubicKeysFilter(
      (index + bounds.firstSourceIndex - bounds.center + 0.5) * axis.inverseScale,
    );
  }
  if (!Number.isFinite(totalWeight) || totalWeight === 0) {
    fail("vision-bicubic-plan-invalid", "Bicubic coefficient generation failed");
  }
  return { ...bounds, totalWeight };
}

function quantizedTapWeight(
  axis: BicubicAxis,
  tap: ReturnType<typeof bicubicTapTotal>,
  index: number,
): number {
  const weight = cubicKeysFilter(
    (index + tap.firstSourceIndex - tap.center + 0.5) * axis.inverseScale,
  ) / tap.totalWeight;
  return roundToInt16(weight * 2 ** axis.precision);
}

/**
 * Builds compact Torchvision bicubic axis metadata without per-pixel objects.
 *
 * Coefficients are regenerated into scalar accumulators so even a strong
 * downscale cannot allocate an input-sized coefficient table.
 */
async function createTorchvisionBicubicAxis(
  sourceSize: number,
  targetSize: number,
  work: CooperativeVisionWork,
): Promise<BicubicAxis> {
  const scale = sourceSize / targetSize;
  const support = scale >= 1 ? 2 * scale : 2;
  const inverseScale = scale >= 1 ? 1 / scale : 1;
  const unquantizedAxis: BicubicAxis = {
    sourceSize,
    targetSize,
    scale,
    support,
    inverseScale,
    precision: 1,
  };
  let maximumWeight = 0;
  for (let targetIndex = 0; targetIndex < targetSize; targetIndex += 1) {
    const tap = bicubicTapTotal(unquantizedAxis, targetIndex);
    for (let index = 0; index < tap.count; index += 1) {
      const normalized = cubicKeysFilter(
        (index + tap.firstSourceIndex - tap.center + 0.5) * inverseScale,
      ) / tap.totalWeight;
      maximumWeight = Math.max(maximumWeight, normalized);
    }
    if ((targetIndex & 63) === 63 || targetIndex === targetSize - 1) {
      await work.checkpoint();
    }
  }
  let precision = 0;
  while (
    precision < 22 &&
    Math.trunc(0.5 + maximumWeight * 2 ** (precision + 1)) < 32_768
  ) {
    precision += 1;
  }
  // Torchvision intentionally permits 22 when no coefficient reaches int16 range earlier.
  if (precision === 0 || precision > 22) {
    fail("vision-bicubic-plan-invalid", "Bicubic coefficient precision is invalid");
  }
  return Object.freeze({
    sourceSize,
    targetSize,
    scale,
    support,
    inverseScale,
    precision,
  });
}

function clampUint8(value: number): number {
  return Math.min(255, Math.max(0, value));
}

async function resampleSourceRowSegment(
  rgb: Uint8Array,
  sourceWidth: number,
  sourceY: number,
  horizontal: BicubicAxis,
  firstTargetX: number,
  targetCount: number,
  work: CooperativeVisionWork,
): Promise<Uint8Array> {
  const row = new Uint8Array(targetCount * 3);
  const rounding = 2 ** (horizontal.precision - 1);
  const divisor = 2 ** horizontal.precision;
  for (let offset = 0; offset < targetCount; offset += 1) {
    const targetX = firstTargetX + offset;
    const tap = bicubicTapTotal(horizontal, targetX);
    let red = rounding;
    let green = rounding;
    let blue = rounding;
    for (let index = 0; index < tap.count; index += 1) {
      work.throwIfCancelled();
      const source = ((sourceY * sourceWidth + tap.firstSourceIndex + index) * 3);
      const weight = quantizedTapWeight(horizontal, tap, index);
      red += rgb[source]! * weight;
      green += rgb[source + 1]! * weight;
      blue += rgb[source + 2]! * weight;
    }
    row[offset * 3] = clampUint8(Math.floor(red / divisor));
    row[offset * 3 + 1] = clampUint8(Math.floor(green / divisor));
    row[offset * 3 + 2] = clampUint8(Math.floor(blue / divisor));
    if ((offset & 63) === 63 || offset === targetCount - 1) {
      await work.checkpoint();
    }
  }
  return row;
}

/**
 * Resizes decoded RGB with the Qwen2-VL fast processor's deterministic
 * Torchvision-compatible bicubic path, without depending on browser canvas.
 */
export async function resizeQwen35VisionRgbBicubic(input: {
  readonly rgb: Uint8Array;
  readonly sourceHeight: number;
  readonly sourceWidth: number;
  readonly targetHeight: number;
  readonly targetWidth: number;
  readonly signal?: AbortSignal;
  readonly decodedSourceByteBudget?: number;
  readonly scheduler?: Qwen35VisionTaskScheduler;
  readonly noOpOwnership?: "copy-source" | "borrow-source";
}): Promise<Qwen35VisionRgbResize> {
  validateRgbGeometry(input);
  const work = new CooperativeVisionWork(input.signal, input.scheduler);
  work.throwIfCancelled();
  if (input.sourceHeight === input.targetHeight && input.sourceWidth === input.targetWidth) {
    return Object.freeze({
      rgb: input.noOpOwnership === "borrow-source" ? input.rgb : new Uint8Array(input.rgb),
      metrics: Object.freeze({
        coefficientTableBytes: 0,
        horizontalRowSegmentsComputed: 0,
        horizontalSamplesComputed: 0,
        peakTemporaryBytes: 0,
        yieldCount: 0,
        outputAliasesSource: input.noOpOwnership === "borrow-source",
      }),
    });
  }
  const horizontal = await createTorchvisionBicubicAxis(input.sourceWidth, input.targetWidth, work);
  const vertical = await createTorchvisionBicubicAxis(input.sourceHeight, input.targetHeight, work);
  const output = new Uint8Array(input.targetHeight * input.targetWidth * 3);
  const verticalRounding = 2 ** (vertical.precision - 1);
  const verticalDivisor = 2 ** vertical.precision;
  // Chunking bounds the accumulator while every horizontal sample is still
  // computed exactly once for each vertical tap that consumes it.
  const maximumChunkWidth = Math.max(1, Math.floor(MAX_ACCUMULATOR_BYTES / (3 * Float64Array.BYTES_PER_ELEMENT)));
  let horizontalRowSegmentsComputed = 0;
  let horizontalSamplesComputed = 0;
  let peakTemporaryBytes = 0;
  for (let targetY = 0; targetY < input.targetHeight; targetY += 1) {
    work.throwIfCancelled();
    const verticalTap = bicubicTapTotal(vertical, targetY);
    for (let firstTargetX = 0; firstTargetX < input.targetWidth; firstTargetX += maximumChunkWidth) {
      const targetCount = Math.min(maximumChunkWidth, input.targetWidth - firstTargetX);
      const accumulator = new Float64Array(targetCount * 3);
      accumulator.fill(verticalRounding);
      for (let index = 0; index < verticalTap.count; index += 1) {
        const row = await resampleSourceRowSegment(
          input.rgb,
          input.sourceWidth,
          verticalTap.firstSourceIndex + index,
          horizontal,
          firstTargetX,
          targetCount,
          work,
        );
        horizontalRowSegmentsComputed += 1;
        horizontalSamplesComputed += targetCount;
        peakTemporaryBytes = Math.max(peakTemporaryBytes, accumulator.byteLength + row.byteLength);
        const weight = quantizedTapWeight(vertical, verticalTap, index);
        for (let offset = 0; offset < row.length; offset += 1) {
          accumulator[offset] = accumulator[offset]! + row[offset]! * weight;
        }
      }
      const outputOffset = (targetY * input.targetWidth + firstTargetX) * 3;
      for (let offset = 0; offset < accumulator.length; offset += 1) {
        output[outputOffset + offset] = clampUint8(
          Math.floor(accumulator[offset]! / verticalDivisor),
        );
      }
      await work.checkpoint();
    }
  }
  return Object.freeze({
    rgb: output,
    metrics: Object.freeze({
      coefficientTableBytes: 0 as const,
      horizontalRowSegmentsComputed,
      horizontalSamplesComputed,
      peakTemporaryBytes,
      yieldCount: work.yields,
      outputAliasesSource: false,
    }),
  });
}

/**
 * Packs one already-resized RGB frame in Qwen's reshape/permute order.
 *
 * The spatial merge cells precede channel, duplicated temporal frame, and
 * patch H/W. Changing this loop order changes every vision-token embedding.
 */
export function packQwen35VisionRgb(input: {
  readonly plan: Qwen35VisionImagePlan;
  readonly rgb: Uint8Array;
}): Qwen35VisionPatchBatch {
  validatePackPlan(input.plan);
  const { plan, rgb } = input;
  const expectedRgbBytes = plan.resizedHeight * plan.resizedWidth * 3;
  if (!(rgb instanceof Uint8Array) || rgb.byteLength !== expectedRgbBytes) {
    fail("vision-rgb-size-invalid", "Input bytes do not match the resize plan");
  }
  const { patchSize, temporalPatchSize, mergeSize, imageMean, imageStd } = plan.settings;
  const [, gridHeight, gridWidth] = plan.gridTHW;
  const mergedGridHeight = gridHeight / mergeSize;
  const mergedGridWidth = gridWidth / mergeSize;
  const patches = new Float32Array(plan.packedFloatCount);
  let output = 0;
  for (let groupHeight = 0; groupHeight < mergedGridHeight; groupHeight += 1) {
    for (let groupWidth = 0; groupWidth < mergedGridWidth; groupWidth += 1) {
      for (let mergedHeight = 0; mergedHeight < mergeSize; mergedHeight += 1) {
        for (let mergedWidth = 0; mergedWidth < mergeSize; mergedWidth += 1) {
          const patchY = (groupHeight * mergeSize + mergedHeight) * patchSize;
          const patchX = (groupWidth * mergeSize + mergedWidth) * patchSize;
          for (let channel = 0; channel < 3; channel += 1) {
            for (let temporal = 0; temporal < temporalPatchSize; temporal += 1) {
              for (let patchHeight = 0; patchHeight < patchSize; patchHeight += 1) {
                for (let patchWidth = 0; patchWidth < patchSize; patchWidth += 1) {
                  const source = ((patchY + patchHeight) * plan.resizedWidth + patchX + patchWidth) * 3 + channel;
                  patches[output] = normalizeRgb(
                    rgb[source]!,
                    imageMean[channel]!,
                    imageStd[channel]!,
                  );
                  output += 1;
                }
              }
            }
          }
        }
      }
    }
  }
  if (output !== patches.length) {
    fail("vision-patch-pack-invalid", "Vision patch packing failed");
  }
  return Object.freeze({
    gridTHW: Object.freeze([...plan.gridTHW]) as readonly [1, number, number],
    projectedVisualTokens: plan.projectedVisualTokens,
    patchVectorLength: plan.patchVectorLength,
    patches,
    resampling: "caller-supplied-resized-rgb",
  });
}

async function packQwen35VisionRgbCooperatively(input: {
  readonly plan: Qwen35VisionImagePlan;
  readonly rgb: Uint8Array;
  readonly signal?: AbortSignal;
  readonly scheduler?: Qwen35VisionTaskScheduler;
}): Promise<Readonly<{ batch: Qwen35VisionPatchBatch; yieldCount: number }>> {
  validatePackPlan(input.plan);
  const { plan, rgb } = input;
  const expectedRgbBytes = plan.resizedHeight * plan.resizedWidth * 3;
  if (!(rgb instanceof Uint8Array) || rgb.byteLength !== expectedRgbBytes) {
    fail("vision-rgb-size-invalid", "Input bytes do not match the resize plan");
  }
  const work = new CooperativeVisionWork(input.signal, input.scheduler);
  work.throwIfCancelled();
  const { patchSize, temporalPatchSize, mergeSize, imageMean, imageStd } = plan.settings;
  const [, gridHeight, gridWidth] = plan.gridTHW;
  const mergedGridHeight = gridHeight / mergeSize;
  const mergedGridWidth = gridWidth / mergeSize;
  const patches = new Float32Array(plan.packedFloatCount);
  let output = 0;
  for (let groupHeight = 0; groupHeight < mergedGridHeight; groupHeight += 1) {
    for (let groupWidth = 0; groupWidth < mergedGridWidth; groupWidth += 1) {
      for (let mergedHeight = 0; mergedHeight < mergeSize; mergedHeight += 1) {
        for (let mergedWidth = 0; mergedWidth < mergeSize; mergedWidth += 1) {
          const patchY = (groupHeight * mergeSize + mergedHeight) * patchSize;
          const patchX = (groupWidth * mergeSize + mergedWidth) * patchSize;
          for (let channel = 0; channel < 3; channel += 1) {
            for (let temporal = 0; temporal < temporalPatchSize; temporal += 1) {
              for (let patchHeight = 0; patchHeight < patchSize; patchHeight += 1) {
                for (let patchWidth = 0; patchWidth < patchSize; patchWidth += 1) {
                  const source = ((patchY + patchHeight) * plan.resizedWidth + patchX + patchWidth) * 3 + channel;
                  patches[output] = normalizeRgb(rgb[source]!, imageMean[channel]!, imageStd[channel]!);
                  output += 1;
                  // A single configured patch can contain many pixels; outer merge-cell
                  // checkpoints alone cannot bound cancellation or browser responsiveness.
                  if ((output & 63) === 0) work.throwIfCancelled();
                  if ((output & 255) === 0) await work.checkpoint();
                }
              }
            }
          }
        }
      }
      await work.checkpoint();
    }
  }
  if (output !== patches.length) {
    fail("vision-patch-pack-invalid", "Vision patch packing failed");
  }
  return Object.freeze({
    batch: Object.freeze({
      gridTHW: Object.freeze([...plan.gridTHW]) as readonly [1, number, number],
      projectedVisualTokens: plan.projectedVisualTokens,
      patchVectorLength: plan.patchVectorLength,
      patches,
      resampling: "qwen2-vl-fast-bicubic",
    }),
    yieldCount: work.yields,
  });
}

/** Resizes decoded source RGB and cooperatively packs Qwen vision patches. */
export async function preprocessQwen35VisionRgb(input: {
  readonly plan: Qwen35VisionImagePlan;
  readonly sourceRgb: Uint8Array;
  readonly sourceHeight: number;
  readonly sourceWidth: number;
  readonly signal?: AbortSignal;
  readonly decodedSourceByteBudget?: number;
  readonly scheduler?: Qwen35VisionTaskScheduler;
}): Promise<Qwen35VisionPatchBatch> {
  validatePackPlan(input.plan);
  if (
    input.sourceHeight !== input.plan.sourceHeight ||
    input.sourceWidth !== input.plan.sourceWidth
  ) {
    fail("vision-source-plan-mismatch", "Decoded source geometry does not match the vision plan");
  }
  const resized = await resizeQwen35VisionRgbBicubic({
    rgb: input.sourceRgb,
    sourceHeight: input.sourceHeight,
    sourceWidth: input.sourceWidth,
    targetHeight: input.plan.resizedHeight,
    targetWidth: input.plan.resizedWidth,
    noOpOwnership: "borrow-source",
    ...(input.signal === undefined ? {} : { signal: input.signal }),
    ...(input.scheduler === undefined ? {} : { scheduler: input.scheduler }),
    ...(input.decodedSourceByteBudget === undefined ? {} : {
      decodedSourceByteBudget: input.decodedSourceByteBudget,
    }),
  });
  const packed = await packQwen35VisionRgbCooperatively({
    plan: input.plan,
    rgb: resized.rgb,
    ...(input.signal === undefined ? {} : { signal: input.signal }),
    ...(input.scheduler === undefined ? {} : { scheduler: input.scheduler }),
  });
  return Object.freeze({
    ...packed.batch,
    preprocessingMetrics: Object.freeze({
      resize: resized.metrics,
      packingYieldCount: packed.yieldCount,
      resizeOutputAliasesSource: resized.metrics.outputAliasesSource,
      estimatedPeakMaterializedBytes: input.plan.materialization.estimatedPeakMaterializedBytes,
    }),
  });
}
