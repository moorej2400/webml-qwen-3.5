/**
 * Browser-only public entrypoint.
 *
 * Node converters and development oracles are intentionally absent from this
 * graph so a static web build cannot pull filesystem or crypto Node builtins.
 */
export * from "./qwen-chat-template.js";
export * from "./qwen-tokenizer.js";
export * from "./qwen35-activation-workspace.js";
export * from "./qwen35-allocation-clear.js";
export * from "./qwen35-forward-dispatch.js";
export * from "./qwen35-greedy-driver.js";
export * from "./qwen35-deltanet-dispatch.js";
export * from "./qwen35-final-dispatch.js";
export * from "./qwen35-full-attention-dispatch.js";
export * from "./qwen35-logits-dispatch.js";
export * from "./qwen35-logits-reduction.js";
export * from "./qwen35-session.js";
export * from "./qwen35-uniform-arena.js";
export {
  QWEN35_DEFAULT_DECODED_SOURCE_BYTE_BUDGET,
  QWEN35_DEFAULT_MAX_VISUAL_TOKENS,
  packQwen35VisionRgb,
  planQwen35VisionImage,
  preprocessQwen35VisionRgb,
  resizeQwen35VisionRgbBicubic,
} from "./qwen35-vision-preprocess.js";
export type {
  Qwen35VisionImagePlan,
  Qwen35VisionMaterializationEstimate,
  Qwen35VisionPatchBatch,
  Qwen35VisionResizeMetrics,
  Qwen35VisionRgbResize,
  Qwen35VisionTaskScheduler,
} from "./qwen35-vision-preprocess.js";
// Production browser callers receive the fixed release only through the
// bootstrap. Generic caller-pinned validation is a development low-level API.
export {
  assertProductionTrustedQwen35VisionPackage,
  QWEN35_PRODUCTION_VISION_PACKAGE_PINS,
} from "./qwen35-vision-package-loader.js";
export type {
  Qwen35ProductionVisionPackage,
  Qwen35VisionLayer,
  Qwen35VisionLayerShard,
  Qwen35VisionLayerSink,
  Qwen35VisionPackage,
} from "./qwen35-vision-package-loader.js";
export * from "./qwen35-vision-package-bootstrap.js";
export * from "./qwen35-vision-program.js";
export * from "./qwen35-vision-foundation-kernels.js";
export * from "./qwen35-vision-layer-kernels.js";
export * from "./qwen35-vision-merger-kernels.js";
export * from "./qwen35-vision-encoder.js";
export * from "./qwen35-vision-streaming-executor.js";
export * from "./qwen35-webgpu-executor.js";
export type {
  Qwen35BrowserLoadOptions,
  Qwen35BorrowedModelDevice,
  Qwen35BorrowedDeviceProfile,
  Qwen35DriverFactoryContext,
  Qwen35ExecutionDriverFactory,
  Qwen35ModelDevice,
  Qwen35PackageDirectory,
  Qwen35PackageSegment,
  Qwen35PackageTensor,
  Qwen35StateAllocationClearContext,
} from "./qwen35-model-loader.js";
