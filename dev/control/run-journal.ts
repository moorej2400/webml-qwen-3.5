import { appendFile, mkdir, readdir, stat } from "node:fs/promises";
import path from "node:path";

import { validateProtocolId } from "./protocol.js";

const MAX_EVENT_BYTES = 16_384;
const ALLOWED_CATEGORIES = new Set([
  "phase",
  "download",
  "cache",
  "shader",
  "memory",
  "image",
  "prefill",
  "generation",
  "thermal",
  "lifecycle",
  "error",
  "socket",
  "device",
  "benchmark",
]);
const ALLOWED_METRIC_KEYS = new Set([
  "phase",
  "durationMs",
  "bytes",
  "cpuBytes",
  "gpuBytes",
  "downloadBytes",
  "cacheHit",
  "shaderCompilationMs",
  "imageEncodingMs",
  "prefillTokens",
  "prefillTokensPerSecond",
  "ttftMs",
  "tokensPerSecond",
  "thermalState",
  "allocationBytes",
  "peakBytes",
  "code",
  "status",
  "count",
  "expected",
  "observed",
  "reason",
  "lifecycle",
  "deviceLost",
  "sequence",
  "nested",
]);

const sanitizeMetricValue = (value: unknown, depth: number): unknown => {
  if (depth > 4) return undefined;
  if (typeof value === "number") return Number.isFinite(value) ? value : undefined;
  if (typeof value === "boolean") return value;
  if (typeof value === "string") return value.slice(0, 128).replace(/[^\x20-\x7e]/g, "");
  if (Array.isArray(value)) {
    return value
      .slice(0, 64)
      .map((entry) => sanitizeMetricValue(entry, depth + 1))
      .filter((entry) => entry !== undefined);
  }
  if (typeof value === "object" && value !== null) {
    const result: Record<string, unknown> = {};
    for (const [key, entry] of Object.entries(value).slice(0, 64)) {
      if (!ALLOWED_METRIC_KEYS.has(key)) continue;
      const sanitized = sanitizeMetricValue(entry, depth + 1);
      if (sanitized !== undefined) result[key] = sanitized;
    }
    return result;
  }
  return undefined;
};

export interface SanitizedTelemetryEvent {
  schemaVersion: 1;
  category: string;
  name: string;
  timestampMs: number;
  metrics: Record<string, unknown>;
  deviceId?: string;
  tabId?: string;
  documentId?: string;
  commandId?: string;
  benchmarkId?: string;
  eventSeq?: number;
}

/**
 * The journal is an allowlist boundary. Unknown fields are discarded instead
 * of redacted because sensitive values can hide under arbitrary key names.
 */
export const sanitizeTelemetryEvent = (input: Record<string, unknown>): SanitizedTelemetryEvent => {
  const category =
    typeof input.category === "string" && ALLOWED_CATEGORIES.has(input.category)
      ? input.category
      : "error";
  const event: SanitizedTelemetryEvent = {
    schemaVersion: 1,
    category,
    name:
      typeof input.name === "string"
        ? input.name.slice(0, 64).replace(/[^A-Za-z0-9_.:-]/g, "_")
        : "invalid_event",
    timestampMs:
      typeof input.timestampMs === "number" && Number.isFinite(input.timestampMs)
        ? Math.max(0, Math.trunc(input.timestampMs))
        : 0,
    metrics: {},
  };
  if (typeof input.metrics === "object" && input.metrics !== null) {
    event.metrics =
      (sanitizeMetricValue(input.metrics, 0) as Record<string, unknown> | undefined) ?? {};
  }
  for (const key of ["deviceId", "tabId", "documentId", "commandId", "benchmarkId"] as const) {
    try {
      if (input[key] !== undefined) event[key] = validateProtocolId(input[key], key);
    } catch {
      // Invalid correlation IDs are omitted so telemetry cannot block control.
    }
  }
  if (Number.isSafeInteger(input.eventSeq) && (input.eventSeq as number) > 0) {
    event.eventSeq = input.eventSeq as number;
  }
  if (Buffer.byteLength(JSON.stringify(event)) > MAX_EVENT_BYTES) {
    event.metrics = { reason: "event_size_limit" };
  }
  return event;
};

export interface RunJournalOptions {
  runsDirectory: string;
  runId: string;
}

export class RunJournal {
  readonly #runsDirectory: string;
  readonly #filePath: string;
  #pending = Promise.resolve();

  constructor(options: RunJournalOptions) {
    const runId = validateProtocolId(options.runId, "runId");
    this.#runsDirectory = path.resolve(options.runsDirectory);
    this.#filePath = path.join(this.#runsDirectory, `${runId}.jsonl`);
  }

  append(input: Record<string, unknown>): Promise<void> {
    const line = `${JSON.stringify(sanitizeTelemetryEvent(input))}\n`;
    this.#pending = this.#pending.then(async () => {
      await mkdir(this.#runsDirectory, { recursive: true, mode: 0o700 });
      await appendFile(this.#filePath, line, { encoding: "utf8", mode: 0o600 });
    });
    return this.#pending;
  }

  close(): Promise<void> {
    return this.#pending;
  }
}

export interface SanitizedRunSummary {
  runId: string;
  bytes: number;
  modifiedAtMs: number;
}

export const listRunSummaries = async (runsDirectory: string): Promise<SanitizedRunSummary[]> => {
  let names: string[];
  try {
    names = await readdir(runsDirectory);
  } catch {
    return [];
  }
  const summaries: SanitizedRunSummary[] = [];
  for (const name of names) {
    if (!name.endsWith(".jsonl")) continue;
    const runId = name.slice(0, -".jsonl".length);
    try {
      validateProtocolId(runId, "runId");
      const info = await stat(path.join(runsDirectory, name));
      if (info.isFile()) summaries.push({ runId, bytes: info.size, modifiedAtMs: info.mtimeMs });
    } catch {
      // Ignore non-protocol files in the local directory.
    }
  }
  return summaries.sort((left, right) => right.modifiedAtMs - left.modifiedAtMs);
};
