import { GgmlType } from "./gguf.js";

export const MTP_EXCLUSION_REASON = "excluded-by-mtp-name-policy-v1";

/**
 * Mirrors model-sources.json for the pinned Qwen language artifact. Keeping the
 * values in the runtime policy makes block ownership available without a JSON
 * fetch; the descriptor test prevents the two public pins from drifting.
 */
export const PINNED_LANGUAGE_BLOCK_POLICY = Object.freeze({
  blockCount: 33,
  baseBlockCount: 32,
  excludedBlock: 32,
});
const PINNED_MTP_BLOCK_PREFIX =
  `blk.${PINNED_LANGUAGE_BLOCK_POLICY.excludedBlock}`;

export type WebGpuTensorStorageType =
  | "f32"
  | "q8-0-36"
  | "q3-k-112"
  | "q3-k-nibble-148"
  | "q3-k-fused-f32-192"
  | "q4-k-144"
  | "q4-k-fused-f32-192"
  | "q5-k-176"
  | "q5-k-fused-f32-224"
  | "q6-k-212"
  | "q6-k-fused-f32-256";

export type WebGpuTensorTransform =
  | "copy"
  | "q8-0-34-to-36"
  | "q3-k-110-to-112"
  | "q3-k-110-to-nibble-148"
  | "q3-k-110-to-fused-f32-192"
  | "q4-k-144-to-fused-f32-192"
  | "q5-k-176-to-fused-f32-224"
  | "q6-k-210-to-212"
  | "q6-k-210-to-fused-f32-256";

export interface WebGpuTensorLayoutPolicy {
  readonly ggmlType: GgmlType;
  readonly storageType: WebGpuTensorStorageType;
  readonly blockElements: number;
  readonly sourceBlockBytes: number;
  readonly outputBlockBytes: number;
  readonly transform: WebGpuTensorTransform;
}

/**
 * The production package uses exact value-preserving browser layouts selected
 * from measured decode evidence. Q3 through Q6 precompute exact FP32 factors
 * so their hot kernels avoid repeated metadata reconstruction; Q6 also enables
 * fused GPU vocabulary selection. The source quantized values are unchanged.
 */
export const WEBGPU_LANGUAGE_TENSOR_LAYOUTS: readonly WebGpuTensorLayoutPolicy[] =
  Object.freeze([
    Object.freeze({
      ggmlType: GgmlType.F32,
      storageType: "f32",
      blockElements: 1,
      sourceBlockBytes: 4,
      outputBlockBytes: 4,
      transform: "copy",
    }),
    Object.freeze({
      ggmlType: GgmlType.Q8_0,
      storageType: "q8-0-36",
      blockElements: 32,
      sourceBlockBytes: 34,
      outputBlockBytes: 36,
      transform: "q8-0-34-to-36",
    }),
    Object.freeze({
      ggmlType: GgmlType.Q3_K,
      storageType: "q3-k-fused-f32-192",
      blockElements: 256,
      sourceBlockBytes: 110,
      outputBlockBytes: 192,
      transform: "q3-k-110-to-fused-f32-192",
    }),
    Object.freeze({
      ggmlType: GgmlType.Q4_K,
      storageType: "q4-k-fused-f32-192",
      blockElements: 256,
      sourceBlockBytes: 144,
      outputBlockBytes: 192,
      transform: "q4-k-144-to-fused-f32-192",
    }),
    Object.freeze({
      ggmlType: GgmlType.Q5_K,
      storageType: "q5-k-fused-f32-224",
      blockElements: 256,
      sourceBlockBytes: 176,
      outputBlockBytes: 224,
      transform: "q5-k-176-to-fused-f32-224",
    }),
    Object.freeze({
      ggmlType: GgmlType.Q6_K,
      storageType: "q6-k-fused-f32-256",
      blockElements: 256,
      sourceBlockBytes: 210,
      outputBlockBytes: 256,
      transform: "q6-k-210-to-fused-f32-256",
    }),
  ]);

export function webGpuLanguageTensorLayout(
  ggmlType: GgmlType,
): WebGpuTensorLayoutPolicy | undefined {
  return WEBGPU_LANGUAGE_TENSOR_LAYOUTS.find(
    (policy) => policy.ggmlType === ggmlType,
  );
}

/**
 * The pinned artifact stores MTP as the complete final block and also uses
 * explicit MTP/nextn name segments. Anchors prevent similar base-model names
 * such as `blk.320.*` and `attempt.weight` from being excluded.
 */
export function isMtpTensorName(name: string): boolean {
  return (
    name === PINNED_MTP_BLOCK_PREFIX ||
    name.startsWith(`${PINNED_MTP_BLOCK_PREFIX}.`) ||
    /(?:^|\.)(?:mtp|nextn)(?:\.|$)/i.test(name)
  );
}
