import { appendFile, mkdir, readFile, readdir, stat } from "node:fs/promises";
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
  maxSegmentBytes?: number;
  maxSegments?: number;
}

export class RunJournal {
  readonly #runsDirectory: string;
  readonly #runId: string;
  readonly #maxSegmentBytes: number;
  readonly #maxSegments: number;
  #segmentIndex = 0;
  #segmentBytes = 0;
  #capped = false;
  #pending = Promise.resolve();

  constructor(options: RunJournalOptions) {
    const runId = validateProtocolId(options.runId, "runId");
    this.#runsDirectory = path.resolve(options.runsDirectory);
    this.#runId = runId;
    this.#maxSegmentBytes = options.maxSegmentBytes ?? 8 * 1024 * 1024;
    this.#maxSegments = options.maxSegments ?? 8;
    if (!Number.isSafeInteger(this.#maxSegmentBytes) || this.#maxSegmentBytes < 256) {
      throw new RangeError("maxSegmentBytes must be at least 256");
    }
    if (!Number.isSafeInteger(this.#maxSegments) || this.#maxSegments < 1 || this.#maxSegments > 64) {
      throw new RangeError("maxSegments must be between 1 and 64");
    }
  }

  append(input: Record<string, unknown>): Promise<void> {
    const sanitized = sanitizeTelemetryEvent(input);
    if (
      sanitized.deviceId === undefined ||
      sanitized.tabId === undefined ||
      sanitized.documentId === undefined ||
      sanitized.eventSeq === undefined
    ) {
      return Promise.reject(new Error("journal event requires authenticated identity and eventSeq"));
    }
    const line = `${JSON.stringify(sanitized)}\n`;
    const lineBytes = Buffer.byteLength(line);
    if (lineBytes > this.#maxSegmentBytes) {
      return Promise.reject(new Error("sanitized event exceeds journal segment cap"));
    }
    this.#pending = this.#pending.then(async () => {
      if (this.#capped) return;
      await mkdir(this.#runsDirectory, { recursive: true, mode: 0o700 });
      if (this.#segmentBytes > 0 && this.#segmentBytes + lineBytes > this.#maxSegmentBytes) {
        if (this.#segmentIndex + 1 >= this.#maxSegments) {
          // Stop this run at its hard cap; overwriting local evidence would
          // violate the recovery and audit contract.
          this.#capped = true;
          return;
        }
        this.#segmentIndex += 1;
        this.#segmentBytes = 0;
      }
      await appendFile(this.#segmentPath(), line, { encoding: "utf8", mode: 0o600 });
      this.#segmentBytes += lineBytes;
    });
    return this.#pending;
  }

  close(): Promise<void> {
    return this.#pending;
  }

  #segmentPath(): string {
    const suffix = this.#segmentIndex === 0 ? "" : `.${String(this.#segmentIndex).padStart(4, "0")}`;
    return path.join(this.#runsDirectory, `${this.#runId}${suffix}.jsonl`);
  }
}

export interface NumericMetricSummary {
  count: number;
  min: number;
  max: number;
  last: number;
}

export interface SanitizedRunSummary {
  runId: string;
  bytes: number;
  modifiedAtMs: number;
  segments: number;
  eventCount: number;
  benchmarkIds: string[];
  metrics: Record<string, NumericMetricSummary>;
}

export const listRunSummaries = async (runsDirectory: string): Promise<SanitizedRunSummary[]> => {
  let names: string[];
  try {
    names = await readdir(runsDirectory);
  } catch {
    return [];
  }
  const grouped = new Map<string, string[]>();
  for (const name of names) {
    if (!name.endsWith(".jsonl")) continue;
    const stem = name.slice(0, -".jsonl".length);
    const segmentMatch = /^(.*)\.\d{4}$/.exec(stem);
    const runId = segmentMatch?.[1] ?? stem;
    try {
      validateProtocolId(runId, "runId");
      const existing = grouped.get(runId) ?? [];
      existing.push(name);
      grouped.set(runId, existing);
    } catch {
      // Ignore non-protocol files in the local directory.
    }
  }
  const summaries: SanitizedRunSummary[] = [];
  for (const [runId, files] of grouped) {
    let bytes = 0;
    let modifiedAtMs = 0;
    let eventCount = 0;
    const benchmarkIds = new Set<string>();
    const metrics: Record<string, NumericMetricSummary> = {};
    const segmentOrder = (name: string): number => {
      const stem = name.slice(0, -".jsonl".length);
      const match = /\.(\d{4})$/.exec(stem);
      return match?.[1] === undefined ? 0 : Number(match[1]);
    };
    for (const name of files.sort((left, right) => segmentOrder(left) - segmentOrder(right))) {
      const filePath = path.join(runsDirectory, name);
      const info = await stat(filePath);
      if (!info.isFile()) continue;
      bytes += info.size;
      modifiedAtMs = Math.max(modifiedAtMs, info.mtimeMs);
      // Journal-created segments are capped; skip external oversized files.
      if (info.size > 64 * 1024 * 1024) continue;
      const lines = (await readFile(filePath, "utf8")).split("\n");
      for (const line of lines) {
        if (line.length === 0) continue;
        try {
          const event = JSON.parse(line) as SanitizedTelemetryEvent;
          eventCount += 1;
          if (event.benchmarkId !== undefined) benchmarkIds.add(event.benchmarkId);
          for (const [key, value] of Object.entries(event.metrics)) {
            if (typeof value !== "number" || !Number.isFinite(value)) continue;
            const prior = metrics[key];
            metrics[key] =
              prior === undefined
                ? { count: 1, min: value, max: value, last: value }
                : {
                    count: prior.count + 1,
                    min: Math.min(prior.min, value),
                    max: Math.max(prior.max, value),
                    last: value,
                  };
          }
        } catch {
          // A partial final line after a power loss does not poison other runs.
        }
      }
    }
    summaries.push({
      runId,
      bytes,
      modifiedAtMs,
      segments: files.length,
      eventCount,
      benchmarkIds: [...benchmarkIds].sort(),
      metrics,
    });
  }
  return summaries.sort((left, right) => right.modifiedAtMs - left.modifiedAtMs);
};
