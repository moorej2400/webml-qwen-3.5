import { appendFile, mkdir, readFile, readdir, stat } from "node:fs/promises";
import path from "node:path";

import type { CoarseOsFamily } from "./device-correlation.js";
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
const ALLOWED_EVENT_NAMES = new Set([
  "connected",
  "pagehide",
  "load_started",
  "load_completed",
  "load_failed",
  "download_started",
  "download_completed",
  "cache_hit",
  "cache_miss",
  "shader_compiled",
  "allocation",
  "image_encoded",
  "prefill_completed",
  "token_rate",
  "generation_started",
  "generation_completed",
  "generation_cancelled",
  "device_lost",
  "socket_disconnected",
  "socket_reconnected",
  "benchmark_started",
  "benchmark_completed",
  "runtime_error",
  "telemetry_omitted",
]);
const NUMERIC_METRIC_KEYS = new Set([
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
  "allocationBytes",
  "peakBytes",
  "count",
  "expected",
  "observed",
  "sequence",
]);
const BOOLEAN_METRIC_KEYS = new Set(["cacheHit", "deviceLost"]);
const STRING_METRIC_VALUES: Readonly<Record<string, ReadonlySet<string>>> = {
  phase: new Set(["adapter", "model", "download", "cache", "shader", "vision", "prefill", "generation", "ready", "disposed"]),
  thermalState: new Set(["unknown", "nominal", "fair", "serious", "critical"]),
  code: new Set(["unknown", "device_lost", "out_of_memory", "network_error", "timeout", "cancelled", "unsupported", "runtime_error"]),
  status: new Set(["accepted", "started", "completed", "failed", "cancelled", "timed_out", "indeterminate", "connected", "disconnected", "ready", "idle", "loading", "loaded", "unavailable"]),
  reason: new Set(["unknown", "device_lost", "out_of_memory", "network_error", "timeout", "cancelled", "unsupported"]),
  lifecycle: new Set(["pagehide", "navigation", "visibilitychange", "freeze", "resume", "reload"]),
};

const sanitizeMetrics = (value: unknown): Record<string, unknown> => {
  if (typeof value !== "object" || value === null || Array.isArray(value)) return {};
  const result: Record<string, unknown> = {};
  for (const [key, entry] of Object.entries(value)) {
    if (NUMERIC_METRIC_KEYS.has(key) && typeof entry === "number" && Number.isFinite(entry)) {
      result[key] = entry;
      continue;
    }
    if (BOOLEAN_METRIC_KEYS.has(key) && typeof entry === "boolean") {
      result[key] = entry;
      continue;
    }
    if (typeof entry === "string" && STRING_METRIC_VALUES[key]?.has(entry)) result[key] = entry;
  }
  return result;
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
  deviceMetadata?: {
    osFamily: CoarseOsFamily;
    osVersion?: string;
    remoteIp: string;
  };
}

const OS_FAMILIES = new Set<CoarseOsFamily>([
  "ios", "macos", "android", "windows", "linux", "other",
]);

const sanitizeDeviceMetadata = (input: unknown): SanitizedTelemetryEvent["deviceMetadata"] => {
  if (typeof input !== "object" || input === null || Array.isArray(input)) return undefined;
  const value = input as Record<string, unknown>;
  if (typeof value.osFamily !== "string" || !OS_FAMILIES.has(value.osFamily as CoarseOsFamily)) {
    return undefined;
  }
  const remoteIp = typeof value.remoteIp === "string" && /^(?:[0-9a-fA-F:.]{1,45}|unknown)$/.test(value.remoteIp)
    ? value.remoteIp
    : "unknown";
  const osVersion =
    typeof value.osVersion === "string" && /^\d{1,2}(?:\.\d{1,2}){0,2}$/.test(value.osVersion)
      ? value.osVersion
      : undefined;
  return { osFamily: value.osFamily as CoarseOsFamily, ...(osVersion === undefined ? {} : { osVersion }), remoteIp };
};

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
    name: typeof input.name === "string" && ALLOWED_EVENT_NAMES.has(input.name) ? input.name : "telemetry_omitted",
    timestampMs:
      typeof input.timestampMs === "number" && Number.isFinite(input.timestampMs)
        ? Math.max(0, Math.trunc(input.timestampMs))
        : 0,
    metrics: {},
  };
  event.metrics = sanitizeMetrics(input.metrics);
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
  const deviceMetadata = sanitizeDeviceMetadata(input.deviceMetadata);
  if (deviceMetadata !== undefined) event.deviceMetadata = deviceMetadata;
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
  now?: () => number;
  appendLine?: (filePath: string, line: string) => Promise<void>;
}

export interface RunJournalHealth {
  state: "healthy" | "degraded";
  writeFailures: number;
  lastFailureAtMs?: number;
}

export class RunJournal {
  readonly #runsDirectory: string;
  readonly #runId: string;
  readonly #maxSegmentBytes: number;
  readonly #maxSegments: number;
  readonly #now: () => number;
  readonly #appendLine: (filePath: string, line: string) => Promise<void>;
  readonly #prepareDirectory: () => Promise<void>;
  #segmentIndex = 0;
  #segmentBytes = 0;
  #capped = false;
  #pending = Promise.resolve();
  #writeFailures = 0;
  #lastFailureAtMs?: number;

  constructor(options: RunJournalOptions) {
    const runId = validateProtocolId(options.runId, "runId");
    this.#runsDirectory = path.resolve(options.runsDirectory);
    this.#runId = runId;
    this.#maxSegmentBytes = options.maxSegmentBytes ?? 8 * 1024 * 1024;
    this.#maxSegments = options.maxSegments ?? 8;
    this.#now = options.now ?? Date.now;
    this.#appendLine =
      options.appendLine ??
      (async (filePath, line) => {
        await appendFile(filePath, line, { encoding: "utf8", mode: 0o600 });
      });
    this.#prepareDirectory =
      options.appendLine === undefined
        ? async () => mkdir(this.#runsDirectory, { recursive: true, mode: 0o700 }).then(() => undefined)
        : async () => undefined;
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
    const operation = this.#pending.then(async () => {
      if (this.#capped) return;
      await this.#prepareDirectory();
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
      await this.#appendLine(this.#segmentPath(), line);
      this.#segmentBytes += lineBytes;
    });
    const reported = operation.catch(() => {
      this.#writeFailures += 1;
      this.#lastFailureAtMs = this.#now();
      throw new Error("journal append failed");
    });
    // A failed append rejects its caller but cannot poison later serialized writes.
    this.#pending = reported.catch(() => undefined);
    return reported;
  }

  close(): Promise<void> {
    return this.#pending;
  }

  getHealth(): Readonly<RunJournalHealth> {
    return Object.freeze({
      state: this.#writeFailures === 0 ? "healthy" : "degraded",
      writeFailures: this.#writeFailures,
      ...(this.#lastFailureAtMs === undefined ? {} : { lastFailureAtMs: this.#lastFailureAtMs }),
    });
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
