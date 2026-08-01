import assert from "node:assert/strict";
import test from "node:test";

import {
  QWEN35_DEFAULT_DECODED_SOURCE_BYTE_BUDGET,
  QWEN35_DEFAULT_MAX_VISUAL_TOKENS,
  packQwen35VisionRgb,
  planQwen35VisionImage,
  preprocessQwen35VisionRgb,
  resizeQwen35VisionRgbBicubic,
  smartResizeQwen2Vl,
} from "../src/qwen35-vision-preprocess.js";
import { QWEN35_VISION_BICUBIC_ORACLE } from "./fixtures/qwen35-vision-bicubic-oracle.js";
import type { VisionProcessorSettings } from "../src/manifest.js";

const tinySettings = (): VisionProcessorSettings => ({
  processorClass: "Qwen3VLProcessor",
  imageProcessorType: "Qwen2VLImageProcessorFast",
  patchSize: 2,
  temporalPatchSize: 2,
  mergeSize: 2,
  shortestEdge: 16,
  longestEdge: 256,
  imageMean: [0.5, 0.5, 0.5],
  imageStd: [0.5, 0.5, 0.5],
});

const pinnedSettings = (): VisionProcessorSettings => ({
  processorClass: "Qwen3VLProcessor",
  imageProcessorType: "Qwen2VLImageProcessorFast",
  patchSize: 16,
  temporalPatchSize: 2,
  mergeSize: 2,
  shortestEdge: 65_536,
  longestEdge: 16_777_216,
  imageMean: [0.5, 0.5, 0.5],
  imageStd: [0.5, 0.5, 0.5],
});

const pixelBytes = (height: number, width: number): Uint8Array =>
  Uint8Array.from(
    Array.from({ length: height * width * 3 }, (_, index) => index),
  );

test("plans official Qwen2-VL smart resize with product visual-token cap", () => {
  const plan = planQwen35VisionImage({
    sourceHeight: 4_096,
    sourceWidth: 4_096,
    settings: tinySettings(),
    maxVisualTokens: 4,
  });

  assert.equal(plan.factor, 4);
  assert.equal(plan.resizedHeight, 8);
  assert.equal(plan.resizedWidth, 8);
  assert.equal(plan.effectiveMaxPixels, 64);
  assert.deepEqual(plan.gridTHW, [1, 4, 4]);
  assert.equal(plan.projectedVisualTokens, 4);
  assert.equal(plan.resampling, "external-bicubic-required");
  assert.equal(plan.requiresBicubicResampling, true);

  const rounded = planQwen35VisionImage({
    sourceHeight: 5,
    sourceWidth: 7,
    settings: tinySettings(),
    maxVisualTokens: 4,
  });
  assert.deepEqual(
    [rounded.resizedHeight, rounded.resizedWidth, rounded.gridTHW],
    [4, 8, [1, 2, 4]],
  );
  assert.equal(rounded.projectedVisualTokens, 2);
});

test("accounts for pinned 16/2/2 grids without allocating image pixels", () => {
  const plan = planQwen35VisionImage({
    sourceHeight: 4_096,
    sourceWidth: 4_096,
    settings: pinnedSettings(),
  });

  assert.equal(QWEN35_DEFAULT_MAX_VISUAL_TOKENS, 1_024);
  assert.equal(plan.factor, 32);
  assert.equal(plan.effectiveMaxPixels, 1_048_576);
  assert.deepEqual(plan.gridTHW, [1, 64, 64]);
  assert.equal(plan.projectedVisualTokens, 1_024);
  assert.equal(plan.patchVectorLength, 1_536);
  assert.equal(plan.patchCount, 4_096);
  assert.equal(plan.packedFloatCount, 6_291_456);
  assert.deepEqual(plan.materialization, {
    sourceRgbBytes: 4_096 * 4_096 * 3,
    resizedRgbBytes: 1_024 * 1_024 * 3,
    patchBytes: 6_291_456 * 4,
    estimatedPeakMaterializedBytes: 78_643_200,
  });
});

test("honors official minimum pixels, aspect bounds, and cap validity", () => {
  const settings = tinySettings();
  const minimum = planQwen35VisionImage({
    sourceHeight: 4,
    sourceWidth: 4,
    settings,
    maxVisualTokens: 1,
  });
  assert.deepEqual(
    [minimum.resizedHeight, minimum.resizedWidth, minimum.projectedVisualTokens],
    [4, 4, 1],
  );
  assert.throws(
    () => planQwen35VisionImage({
      sourceHeight: 4,
      sourceWidth: 804,
      settings,
    }),
    { code: "vision-aspect-ratio-invalid" },
  );
  assert.throws(
    () => planQwen35VisionImage({
      sourceHeight: 4,
      sourceWidth: 4,
      settings: pinnedSettings(),
      maxVisualTokens: 63,
    }),
    { code: "vision-product-cap-invalid" },
  );
  const largePlan = planQwen35VisionImage({
    sourceHeight: 4_096,
    sourceWidth: 4_096,
    settings: pinnedSettings(),
    maxVisualTokens: 16_384,
  });
  assert.equal(largePlan.projectedVisualTokens, 16_384);
  assert.equal(largePlan.packedFloatCount, 100_663_296);
  assert.deepEqual(largePlan.materialization, {
    sourceRgbBytes: 50_331_648,
    resizedRgbBytes: 50_331_648,
    patchBytes: 402_653_184,
    estimatedPeakMaterializedBytes: 452_984_832,
  });

  const onePixelSettings: VisionProcessorSettings = {
    ...tinySettings(),
    patchSize: 1,
    temporalPatchSize: 1,
    mergeSize: 1,
    shortestEdge: 1,
    longestEdge: 16_777_216,
  };
  const maximumPlan = planQwen35VisionImage({
    sourceHeight: 128,
    sourceWidth: 128,
    settings: onePixelSettings,
    maxVisualTokens: 16_384,
  });
  const overProductPlan = {
    ...maximumPlan,
    resizedWidth: 129,
    gridTHW: [1, 128, 129] as const,
    projectedVisualTokens: 16_512,
    patchCount: 16_512,
    packedFloatCount: 49_536,
  };
  assert.throws(
    () => packQwen35VisionRgb({
      plan: overProductPlan,
      rgb: new Uint8Array(128 * 129 * 3),
    }),
    { code: "vision-patch-plan-invalid" },
  );
});

test("accepts an exact 200:1 aspect ratio under official Qwen2-VL geometry", () => {
  const plan = planQwen35VisionImage({
    sourceHeight: 32,
    sourceWidth: 6_400,
    settings: pinnedSettings(),
  });
  assert.deepEqual([plan.resizedHeight, plan.resizedWidth], [32, 6_400]);
  assert.equal(plan.projectedVisualTokens, 200);
});

test("upscales source dimensions below the spatial factor", () => {
  const plan = planQwen35VisionImage({
    sourceHeight: 16,
    sourceWidth: 16,
    settings: pinnedSettings(),
  });
  assert.deepEqual([plan.resizedHeight, plan.resizedWidth], [256, 256]);
  assert.equal(plan.projectedVisualTokens, 64);
});

test("clamps a downscaled short edge to one factor instead of zero", () => {
  const resized = smartResizeQwen2Vl({
    sourceHeight: 31,
    sourceWidth: 6_200,
    factor: 32,
    shortestEdge: 65_536,
    longestEdge: 65_536,
  });
  assert.deepEqual(resized, [32, 3_616]);
});

test("uses Python round-half-even for exact resize ties", () => {
  assert.deepEqual(
    smartResizeQwen2Vl({
      sourceHeight: 10,
      sourceWidth: 14,
      factor: 4,
      shortestEdge: 1,
      longestEdge: 400,
    }),
    [8, 16],
  );
  assert.deepEqual(
    smartResizeQwen2Vl({
      sourceHeight: 14,
      sourceWidth: 10,
      factor: 4,
      shortestEdge: 1,
      longestEdge: 400,
    }),
    [16, 8],
  );
});

test("accepts a smart-resize target that rounds past a 200:1 aspect ratio", async () => {
  const sourceHeight = 33;
  const sourceWidth = 6_417;
  const [targetHeight, targetWidth] = smartResizeQwen2Vl({
    sourceHeight,
    sourceWidth,
    factor: 32,
    shortestEdge: 1,
    longestEdge: 16_777_216,
  });
  assert.deepEqual([targetHeight, targetWidth], [32, 6_432]);

  // The source ratio is valid. Qwen2-VL does not reject the rounded target ratio.
  const resized = await resizeQwen35VisionRgbBicubic({
    rgb: new Uint8Array(sourceHeight * sourceWidth * 3),
    sourceHeight,
    sourceWidth,
    targetHeight,
    targetWidth,
  });
  assert.equal(resized.rgb.byteLength, targetHeight * targetWidth * 3);
});

test("packs already-resized RGB in official grouped spatial merge order", () => {
  const plan = planQwen35VisionImage({
    sourceHeight: 4,
    sourceWidth: 4,
    settings: tinySettings(),
    maxVisualTokens: 4,
  });
  const rgb = pixelBytes(4, 4);
  const packed = packQwen35VisionRgb({ plan, rgb });

  assert.equal(packed.resampling, "caller-supplied-resized-rgb");
  assert.deepEqual(packed.gridTHW, [1, 2, 2]);
  assert.equal(packed.patches.length, 4 * 24);
  // Patch row 1 is the right merge-cell before the lower merge-cell.
  const expectedFirstChannel = [
    0, 3, 12, 15,
    0, 3, 12, 15,
    1, 4, 13, 16,
    1, 4, 13, 16,
    2, 5, 14, 17,
    2, 5, 14, 17,
  ].map((value) => Math.fround(
    Math.fround(Math.fround(value) - Math.fround(Math.fround(0.5) * Math.fround(255))) /
      Math.fround(Math.fround(0.5) * Math.fround(255)),
  ));
  assert.deepEqual([...packed.patches.subarray(0, 24)], expectedFirstChannel);

  const secondPatchFirstValue = Math.fround(
    Math.fround(Math.fround(6) - Math.fround(Math.fround(0.5) * Math.fround(255))) /
      Math.fround(Math.fround(0.5) * Math.fround(255)),
  );
  assert.equal(packed.patches[24], secondPatchFirstValue);
  rgb.fill(255);
  assert.equal(packed.patches[0], expectedFirstChannel[0]);
});

test("packs externally resized RGB even when the plan requires bicubic work", () => {
  const plan = planQwen35VisionImage({
    sourceHeight: 4_096,
    sourceWidth: 4_096,
    settings: tinySettings(),
    maxVisualTokens: 4,
  });
  assert.equal(plan.requiresBicubicResampling, true);
  const packed = packQwen35VisionRgb({ plan, rgb: pixelBytes(8, 8) });
  assert.equal(packed.patches.length, 16 * 24);
  assert.equal(packed.resampling, "caller-supplied-resized-rgb");
});

test("normalizes RGB bytes with the pinned f32 operation order", () => {
  const settings: VisionProcessorSettings = {
    ...tinySettings(),
    patchSize: 1,
    mergeSize: 1,
    shortestEdge: 1,
    imageMean: [0.485, 0.456, 0.406],
    imageStd: [0.229, 0.224, 0.225],
  };
  const plan = planQwen35VisionImage({
    sourceHeight: 1,
    sourceWidth: 1,
    settings,
    maxVisualTokens: 1,
  });
  const packed = packQwen35VisionRgb({
    plan,
    rgb: Uint8Array.of(127, 128, 255),
  });
  // These are independent values from the pinned fast processor's byte-space
  // f32 path, not a copy of the runtime normalization expression.
  assert.deepEqual([...packed.patches], [
    0.056939754635095596, 0.056939754635095596,
    0.20518210530281067, 0.20518210530281067,
    2.640000104904175, 2.640000104904175,
  ]);
});

test("matches the pinned Qwen2-VL fast bicubic RGB oracle without canvas", async () => {
  const oracle = QWEN35_VISION_BICUBIC_ORACLE;
  assert.equal(oracle.transformersRevision, "3717b9cda226af7377da1944f395af0eac6e2b51");
  assert.deepEqual(
    (await resizeQwen35VisionRgbBicubic({
      rgb: oracle.sourceRgb,
      sourceHeight: oracle.sourceHeight,
      sourceWidth: oracle.sourceWidth,
      targetHeight: oracle.targetHeight,
      targetWidth: oracle.targetWidth,
    })).rgb,
    oracle.resizedRgb,
  );
  assert.deepEqual(
    (await resizeQwen35VisionRgbBicubic({
      rgb: oracle.downsample.sourceRgb,
      sourceHeight: oracle.downsample.sourceHeight,
      sourceWidth: oracle.downsample.sourceWidth,
      targetHeight: oracle.downsample.targetHeight,
      targetWidth: oracle.downsample.targetWidth,
    })).rgb,
    oracle.downsample.resizedRgb,
  );
});

test("matches precision-22 and eight broader strong-downscale oracles", async () => {
  const oracle = QWEN35_VISION_BICUBIC_ORACLE;
  const strong = oracle.strongDownsample;
  const source = Uint8Array.from(
    { length: strong.sourceHeight * strong.sourceWidth * 3 },
    (_, index) => (index * strong.sourceMultiplier + strong.sourceAddend) % 256,
  );
  const result = await resizeQwen35VisionRgbBicubic({
    rgb: source,
    sourceHeight: strong.sourceHeight,
    sourceWidth: strong.sourceWidth,
    targetHeight: strong.targetHeight,
    targetWidth: strong.targetWidth,
  });
  assert.deepEqual(result.rgb, strong.resizedRgb);
  assert.equal(result.metrics.horizontalRowSegmentsComputed, 300);
  assert.equal(result.metrics.horizontalSamplesComputed, 300 * 31);
  assert.ok(result.metrics.peakTemporaryBytes <= 1024 * 1024);

  for (const fixture of oracle.additionalDownsamples) {
    const rgb = Uint8Array.from(
      { length: fixture.sourceHeight * fixture.sourceWidth * 3 },
      (_, index) => (index * 37 + 11) % 256,
    );
    const resized = await resizeQwen35VisionRgbBicubic({
      rgb,
      sourceHeight: fixture.sourceHeight,
      sourceWidth: fixture.sourceWidth,
      targetHeight: fixture.targetHeight,
      targetWidth: fixture.targetWidth,
    });
    assert.deepEqual(resized.rgb, fixture.resizedRgb);
  }
});

test("uses elapsed-time task slices without creating operation-count yield storms", async () => {
  let clockReads = 0;
  let schedulerYields = 0;
  const result = await resizeQwen35VisionRgbBicubic({
    rgb: Uint8Array.from({ length: 256 * 256 * 3 }, (_, index) => index % 256),
    sourceHeight: 256,
    sourceWidth: 256,
    targetHeight: 64,
    targetWidth: 64,
    scheduler: {
      now: () => clockReads++ * 0.5,
      yieldToBrowser: async () => { schedulerYields += 1; },
    },
  });
  assert.equal(result.metrics.yieldCount, schedulerYields);
  assert.ok(schedulerYields > 0);
  assert.ok(schedulerYields < 100);

  const real = await resizeQwen35VisionRgbBicubic({
    rgb: new Uint8Array(256 * 256 * 3),
    sourceHeight: 256,
    sourceWidth: 256,
    targetHeight: 64,
    targetWidth: 64,
  });
  assert.ok(real.metrics.yieldCount < 20);
});

test("enforces a configurable decoded-source staging budget before work", async () => {
  assert.equal(QWEN35_DEFAULT_DECODED_SOURCE_BYTE_BUDGET, 64 * 1024 * 1024);
  const input = {
    rgb: new Uint8Array(12),
    sourceHeight: 2,
    sourceWidth: 2,
    targetHeight: 2,
    targetWidth: 2,
  } as const;
  await assert.rejects(
    resizeQwen35VisionRgbBicubic({ ...input, decodedSourceByteBudget: 11 }),
    { code: "vision-decoded-source-budget-exceeded" },
  );
  await resizeQwen35VisionRgbBicubic({ ...input, decodedSourceByteBudget: 12 });
});

test("preflights forged plans before scheduling, resizing, or allocation", async () => {
  const plan = planQwen35VisionImage({
    sourceHeight: 4,
    sourceWidth: 4,
    settings: tinySettings(),
    maxVisualTokens: 4,
  });
  let schedulerCalls = 0;
  await assert.rejects(preprocessQwen35VisionRgb({
    plan: { ...plan, projectedVisualTokens: 3 },
    sourceRgb: pixelBytes(4, 4),
    sourceHeight: 4,
    sourceWidth: 4,
    scheduler: {
      now: () => { schedulerCalls += 1; return 0; },
      yieldToBrowser: async () => { schedulerCalls += 1; },
    },
  }), { code: "vision-patch-plan-invalid" });
  assert.equal(schedulerCalls, 0);
});

test("states no-op RGB ownership and reports default-like packing yields", async () => {
  const source = pixelBytes(4, 4);
  const copied = await resizeQwen35VisionRgbBicubic({
    rgb: source,
    sourceHeight: 4,
    sourceWidth: 4,
    targetHeight: 4,
    targetWidth: 4,
  });
  assert.notEqual(copied.rgb, source);
  assert.equal(copied.metrics.outputAliasesSource, false);

  const smallPlan = planQwen35VisionImage({
    sourceHeight: 4,
    sourceWidth: 4,
    settings: tinySettings(),
    maxVisualTokens: 4,
  });
  const borrowed = await preprocessQwen35VisionRgb({
    plan: smallPlan,
    sourceRgb: source,
    sourceHeight: 4,
    sourceWidth: 4,
  });
  assert.equal(borrowed.preprocessingMetrics?.resizeOutputAliasesSource, true);
  assert.equal(
    borrowed.preprocessingMetrics?.estimatedPeakMaterializedBytes,
    4 * 4 * 3 + smallPlan.packedFloatCount * Float32Array.BYTES_PER_ELEMENT,
  );

  const defaultLikePlan = planQwen35VisionImage({
    sourceHeight: 1_024,
    sourceWidth: 1_024,
    settings: pinnedSettings(),
  });
  const defaultLike = await preprocessQwen35VisionRgb({
    plan: defaultLikePlan,
    sourceRgb: new Uint8Array(1_024 * 1_024 * 3),
    sourceHeight: 1_024,
    sourceWidth: 1_024,
  });
  assert.ok((defaultLike.preprocessingMetrics?.packingYieldCount ?? 1_000) < 50);
});

test("yields to the browser task queue and observes cancellation after work starts", async () => {
  const controller = new AbortController();
  const work = resizeQwen35VisionRgbBicubic({
    rgb: Uint8Array.from({ length: 512 * 512 * 3 }, (_, index) => index % 256),
    sourceHeight: 512,
    sourceWidth: 512,
    targetHeight: 256,
    targetWidth: 256,
    signal: controller.signal,
  });
  setTimeout(() => controller.abort(), 0);
  const outcome = await Promise.race([
    work.then(() => "completed", (error: unknown) => {
      assert.equal((error as { code?: unknown }).code, "vision-preprocess-cancelled");
      return "cancelled";
    }),
    new Promise<string>((resolve) => setTimeout(() => resolve("hung"), 1_000)),
  ]);
  assert.equal(outcome, "cancelled");
});

test("keeps cooperative cancellation active while preprocessing packs patches", async () => {
  const settings: VisionProcessorSettings = {
    ...tinySettings(),
    patchSize: 1,
    mergeSize: 1,
    shortestEdge: 1,
    longestEdge: 16_384,
  };
  const plan = planQwen35VisionImage({
    sourceHeight: 128,
    sourceWidth: 128,
    settings,
    maxVisualTokens: 16_384,
  });
  const controller = new AbortController();
  const work = preprocessQwen35VisionRgb({
    plan,
    sourceRgb: Uint8Array.from({ length: 128 * 128 * 3 }, (_, index) => index % 256),
    sourceHeight: 128,
    sourceWidth: 128,
    signal: controller.signal,
  });
  setTimeout(() => controller.abort(), 0);
  await assert.rejects(work, { code: "vision-preprocess-cancelled" });
});

test("checks elapsed time within one large patch-pack inner loop", async () => {
  const settings: VisionProcessorSettings = {
    ...tinySettings(),
    patchSize: 256,
    temporalPatchSize: 1,
    mergeSize: 1,
    shortestEdge: 65_536,
    longestEdge: 65_536,
  };
  const plan = planQwen35VisionImage({
    sourceHeight: 256,
    sourceWidth: 256,
    settings,
    maxVisualTokens: 1,
  });
  let now = 0;
  let yields = 0;
  const batch = await preprocessQwen35VisionRgb({
    plan,
    sourceRgb: new Uint8Array(256 * 256 * 3),
    sourceHeight: 256,
    sourceWidth: 256,
    scheduler: {
      now: () => now++,
      yieldToBrowser: async () => { yields += 1; },
    },
  });
  assert.ok((batch.preprocessingMetrics?.packingYieldCount ?? 0) > 1);
  assert.equal(batch.preprocessingMetrics?.packingYieldCount, yields);
});

test("resamples decoded source RGB then emits Qwen grouped patches", async () => {
  const plan = planQwen35VisionImage({
    sourceHeight: 2,
    sourceWidth: 2,
    settings: tinySettings(),
    maxVisualTokens: 4,
  });
  const sourceRgb = Uint8Array.of(
    0, 0, 0, 255, 0, 0,
    0, 255, 0, 0, 0, 255,
  );
  const batch = await preprocessQwen35VisionRgb({
    plan,
    sourceRgb,
    sourceHeight: 2,
    sourceWidth: 2,
  });
  assert.equal(batch.resampling, "qwen2-vl-fast-bicubic");
  assert.deepEqual(batch.gridTHW, [1, 2, 2]);
  assert.equal(batch.patches.length, 4 * 24);
  assert.notEqual(batch.patches[0], batch.patches[24]);

  await assert.rejects(
    preprocessQwen35VisionRgb({
      plan,
      sourceRgb: new Uint8Array(3),
      sourceHeight: 2,
      sourceWidth: 2,
    }),
    { code: "vision-source-rgb-size-invalid" },
  );
  await assert.rejects(
    preprocessQwen35VisionRgb({
      plan,
      sourceRgb: new Uint8Array(12),
      sourceHeight: 1,
      sourceWidth: 4,
    }),
    { code: "vision-source-plan-mismatch" },
  );
});

test("snapshots processor metadata and rejects non-resized or unsafe RGB input", () => {
  const settings = tinySettings();
  const plan = planQwen35VisionImage({
    sourceHeight: 4,
    sourceWidth: 4,
    settings,
    maxVisualTokens: 4,
  });
  settings.imageMean[0] = 0;
  settings.imageStd[0] = 1;
  assert.equal(plan.settings.imageMean[0], 0.5);
  assert.equal(plan.settings.imageStd[0], 0.5);
  assert.ok(Object.isFrozen(plan));
  assert.ok(Object.isFrozen(plan.settings));
  assert.ok(Object.isFrozen(plan.settings.imageMean));
  assert.ok(Object.isFrozen(plan.gridTHW));

  assert.throws(
    () => packQwen35VisionRgb({ plan, rgb: new Uint8Array(3) }),
    (error: unknown) => {
      assert.equal((error as { code?: unknown }).code, "vision-rgb-size-invalid");
      assert.doesNotMatch((error as Error).message, /4|rgb|path/i);
      return true;
    },
  );
  const packed = packQwen35VisionRgb({ plan, rgb: pixelBytes(4, 4) });
  assert.ok(Object.isFrozen(packed));
  assert.ok(Object.isFrozen(packed.gridTHW));

  const forgedPlan = {
    ...plan,
    gridTHW: [1, 2, 2] as const,
    projectedVisualTokens: 3,
  };
  assert.throws(
    () => packQwen35VisionRgb({ plan: forgedPlan, rgb: pixelBytes(4, 4) }),
    { code: "vision-patch-plan-invalid" },
  );
});
