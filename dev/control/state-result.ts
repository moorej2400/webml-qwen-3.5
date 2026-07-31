const MAX_STATE_BYTES = 8_192;
const ALLOWED_STATE_KEYS = new Set([
  "modelState",
  "generationState",
  "cacheState",
  "deviceState",
  "loaded",
  "generating",
  "contextTokens",
  "maxContextTokens",
  "cpuBytes",
  "gpuBytes",
  "activeCommandId",
]);

export type SanitizedStateResult = Record<string, string | number | boolean>;

/**
 * Keeps getState useful for recovery while preventing prompts, responses, and
 * arbitrary application objects from crossing the local control boundary.
 */
export const sanitizeStateResult = (input: unknown): SanitizedStateResult => {
  const result: SanitizedStateResult = {};
  if (typeof input !== "object" || input === null || Array.isArray(input)) return result;
  for (const [key, value] of Object.entries(input).slice(0, 32)) {
    if (!ALLOWED_STATE_KEYS.has(key)) continue;
    if (typeof value === "boolean") result[key] = value;
    if (typeof value === "number" && Number.isFinite(value)) result[key] = value;
    if (typeof value === "string") result[key] = value.slice(0, 128).replace(/[^\x20-\x7e]/g, "");
  }
  return Buffer.byteLength(JSON.stringify(result)) <= MAX_STATE_BYTES ? result : {};
};
