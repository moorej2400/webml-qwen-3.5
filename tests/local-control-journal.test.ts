import assert from "node:assert/strict";
import { mkdtemp, readFile, readdir } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";

import {
  RunJournal,
  listRunSummaries,
  sanitizeTelemetryEvent,
} from "../dev/control/run-journal.js";
import { deriveSocketDeviceMetadata } from "../dev/control/device-correlation.js";

const LOAD_PHASES = [
  "lock_wait",
  "cache_scan",
  "cache_download",
  "cache_verify",
  "tokenizer_load",
  "device_probe",
  "state_allocate",
  "weights_allocate",
  "weights_upload",
  "driver_initialize",
  "ready",
  "failed",
] as const;

const ALLOCATION_DIAGNOSTIC_CODES = [
  "gpu_out_of_memory",
  "gpu_validation",
  "buffer_creation",
  "error_scope",
  "allocation_conflict",
  "unknown",
] as const;

test("server-derived local metadata keeps only coarse OS and the direct socket IP", () => {
  assert.deepEqual(
    deriveSocketDeviceMetadata({
      remoteAddress: "::ffff:192.0.2.44",
      userAgent: "Mozilla/5.0 (iPhone; CPU iPhone OS 18_4 like Mac OS X)",
    }),
    { osFamily: "ios", osVersion: "18.4", remoteIp: "192.0.2.44" },
  );
  assert.deepEqual(
    deriveSocketDeviceMetadata({
      remoteAddress: "not-an-ip",
      userAgent: "untrusted raw value 100.200.300.400",
    }),
    { osFamily: "other", remoteIp: "unknown" },
  );
});

test("journal records only server-authoritative device metadata", () => {
  const sanitized = sanitizeTelemetryEvent({
    schemaVersion: 1,
    category: "device",
    name: "connected",
    timestampMs: 100,
    deviceId: "device_0123456789abcdef",
    tabId: "tab_0123456789abcdef",
    documentId: "document_0123456789abcdef",
    eventSeq: 1,
    deviceMetadata: {
      osFamily: "ios",
      osVersion: "18.4.1",
      remoteIp: "192.0.2.44",
      forwardedFor: "203.0.113.10",
      rawUserAgent: "private",
    },
    metrics: { status: "connected" },
  });

  assert.deepEqual(sanitized.deviceMetadata, {
    osFamily: "ios",
    osVersion: "18.4.1",
    remoteIp: "192.0.2.44",
  });
  assert.doesNotMatch(JSON.stringify(sanitized), /forwarded|private|userAgent/i);
});

test("telemetry uses a flat allowlist and omits private content", () => {
  const sanitized = sanitizeTelemetryEvent({
    schemaVersion: 1,
    category: "generation",
    name: "token_rate",
    timestampMs: 100,
    metrics: {
      tokensPerSecond: 31.5,
      nested: { cpuBytes: 4096, prompt: "private", cookie: "secret" },
      stack: "private stack",
      url: "https://private.invalid/path",
    },
    prompt: "raw prompt",
    response: "raw response",
    headers: { authorization: "secret" },
  });

  const encoded = JSON.stringify(sanitized);
  assert.match(encoded, /tokensPerSecond/);
  assert.doesNotMatch(encoded, /cpuBytes|nested/i);
  assert.doesNotMatch(encoded, /private|secret|authorization|cookie|prompt|response|url|stack/i);
});

test("journal preserves bounded per-command performance counters", () => {
  const sanitized = sanitizeTelemetryEvent({
    schemaVersion: 1,
    category: "generation",
    name: "generation_completed",
    timestampMs: 100,
    metrics: {
      durationMs: 12_625,
      count: 8,
      contextTokens: 23,
      decodedTextCodeUnits: 19,
      referenceTokenCount: 16,
      referenceTokenMismatchCount: 1,
      referenceFirstMismatchIndex: 0,
      referenceExpectedTokenId: 5_793,
      referenceObservedTokenId: 760,
      performanceSnapshotCount: 2,
      diskReadBytes: 11_596_081_000,
      gpuUploadBytes: 11_596_081_000,
      dispatchCount: 4_416,
      queueSubmissionCount: 216,
      queueRetirementCount: 192,
      gpuReadbackCount: 8,
      prompt: "private prompt",
      response: "private response",
    },
  });

  assert.deepEqual(sanitized.metrics, {
    durationMs: 12_625,
    count: 8,
    contextTokens: 23,
    decodedTextCodeUnits: 19,
    referenceTokenCount: 16,
    referenceTokenMismatchCount: 1,
    referenceFirstMismatchIndex: 0,
    referenceExpectedTokenId: 5_793,
    referenceObservedTokenId: 760,
    performanceSnapshotCount: 2,
    diskReadBytes: 11_596_081_000,
    gpuUploadBytes: 11_596_081_000,
    dispatchCount: 4_416,
    queueSubmissionCount: 216,
    queueRetirementCount: 192,
    gpuReadbackCount: 8,
  });
  assert.doesNotMatch(JSON.stringify(sanitized), /private|prompt|response/i);
});

test("journal preserves allocation diagnostic codes and bounded numbers", () => {
  for (const code of ALLOCATION_DIAGNOSTIC_CODES) {
    const sanitized = sanitizeTelemetryEvent({
      schemaVersion: 1,
      category: "error",
      name: "runtime_error",
      timestampMs: 100,
      metrics: {
        code,
        allocationBytes: 2_097_152,
        count: 2,
        message: "private GPU message at local-path:<path>/<model-id>.gguf",
        name: "PrivateGpuError",
        path: "local-path:<path>/<model-id>.gguf",
        stack: "private stack local-path:<path>/<model-id>.gguf",
      },
    });

    assert.deepEqual(sanitized.metrics, {
      code,
      allocationBytes: 2_097_152,
      count: 2,
    });
    assert.doesNotMatch(
      JSON.stringify(sanitized),
      /private|local|model\.gguf|message|stack|path|PrivateGpuError/i,
    );
  }
});

test("journal preserves stable uppercase runtime diagnostic codes", () => {
  for (const code of [
    "KERNEL_COMPILE_FAILED",
    "GPU_BUFFER_USAGE_INVALID",
    "ALLOCATION_LIMIT_EXCEEDED",
  ]) {
    const sanitized = sanitizeTelemetryEvent({
      schemaVersion: 1,
      category: "error",
      name: "runtime_error",
      timestampMs: 100,
      metrics: { code, message: "private detail" },
    });
    assert.deepEqual(sanitized.metrics, { code });
  }
});

test("journal maps an unapproved allocation diagnostic code to unknown", () => {
  const sanitized = sanitizeTelemetryEvent({
    schemaVersion: 1,
    category: "error",
    name: "runtime_error",
    timestampMs: 100,
    metrics: {
      code: "private_dynamic_gpu_error",
      allocationBytes: 131_072,
      message: "private message",
      stack: "private stack",
    },
  });

  assert.deepEqual(sanitized.metrics, {
    code: "unknown",
    allocationBytes: 131_072,
  });
  assert.doesNotMatch(JSON.stringify(sanitized), /private|dynamic|message|stack/i);
});

test("journal preserves exact bounded load progress and drops private runtime text", () => {
  const privateText = "private prompt model URL path response error and stack";
  for (const phase of LOAD_PHASES) {
    const sanitized = sanitizeTelemetryEvent({
      schemaVersion: 1,
      category: "phase",
      name: "load_started",
      timestampMs: 100,
      metrics: {
        phase,
        completedBytes: 32,
        totalBytes: 64,
        shardIndex: 0,
        shardCount: 2,
        currentGpuBytes: 16,
        peakGpuBytes: 24,
        prompt: privateText,
        response: privateText,
        url: privateText,
        path: privateText,
        error: privateText,
        stack: privateText,
      },
    });

    assert.deepEqual(sanitized.metrics, {
      phase,
      completedBytes: 32,
      totalBytes: 64,
      shardIndex: 0,
      shardCount: 2,
      currentGpuBytes: 16,
      peakGpuBytes: 24,
    });
    assert.doesNotMatch(
      JSON.stringify(sanitized),
      /private|prompt|response|url|path|error|stack/i,
    );
  }
});

test("journal preserves only the bounded upload probe schema", () => {
  for (const name of ["before_write", "after_write", "after_retire"] as const) {
    const sanitized = sanitizeTelemetryEvent({
      schemaVersion: 1,
      category: "upload",
      name,
      timestampMs: 100,
      metrics: {
        ordinal: 17,
        shardIndex: 13,
        shardCount: 20,
        segmentIndex: 4,
        segmentCount: 18,
        globalOffset: 1_824_496_640,
        byteCount: 8 * 1024 * 1024,
        bufferShardBytes: 128 * 1024 * 1024,
        uploadLaneBytes: 8 * 1024 * 1024,
        retireAfterEachWrite: false,
        tensorName: "private tensor identity",
        url: "https://private.invalid/model",
        path: "/private/local/path",
        prompt: "private prompt",
        response: "private response",
        stack: "private stack",
        secret: "private secret",
      },
    });

    assert.equal(sanitized.category, "upload");
    assert.equal(sanitized.name, name);
    assert.deepEqual(sanitized.metrics, {
      ordinal: 17,
      shardIndex: 13,
      shardCount: 20,
      segmentIndex: 4,
      segmentCount: 18,
      globalOffset: 1_824_496_640,
      byteCount: 8 * 1024 * 1024,
      bufferShardBytes: 128 * 1024 * 1024,
      uploadLaneBytes: 8 * 1024 * 1024,
      retireAfterEachWrite: false,
    });
    assert.doesNotMatch(
      JSON.stringify(sanitized),
      /tensor|private|url|path|prompt|response|stack|secret/i,
    );
  }
});

test("journal rejects invalid upload ordinals, boundary pairs, and byte ranges", () => {
  const sanitized = sanitizeTelemetryEvent({
    schemaVersion: 1,
    category: "upload",
    name: "before_write",
    timestampMs: 100,
    metrics: {
      ordinal: 0,
      shardIndex: 20,
      shardCount: 20,
      segmentIndex: 4,
      segmentCount: 0,
      globalOffset: Number.MAX_SAFE_INTEGER - 3,
      byteCount: 8,
      bufferShardBytes: 0,
      uploadLaneBytes: 3,
      retireAfterEachWrite: "false",
    },
  });

  assert.deepEqual(sanitized.metrics, {});

  const exceedsLane = sanitizeTelemetryEvent({
    schemaVersion: 1,
    category: "upload",
    name: "after_write",
    timestampMs: 100,
    metrics: {
      ordinal: 1,
      shardIndex: 0,
      shardCount: 1,
      segmentIndex: 0,
      segmentCount: 1,
      globalOffset: 0,
      byteCount: 8,
      bufferShardBytes: 64,
      uploadLaneBytes: 4,
      retireAfterEachWrite: false,
    },
  });
  assert.deepEqual(exceedsLane.metrics, {});
});

test("journal binds upload stage names to the upload category", () => {
  const wrongCategory = sanitizeTelemetryEvent({
    schemaVersion: 1,
    category: "generation",
    name: "before_write",
    timestampMs: 100,
    metrics: { tokensPerSecond: 10 },
  });
  const wrongName = sanitizeTelemetryEvent({
    schemaVersion: 1,
    category: "upload",
    name: "token_rate",
    timestampMs: 100,
    metrics: { tokensPerSecond: 10 },
  });

  assert.equal(wrongCategory.name, "telemetry_omitted");
  assert.deepEqual(wrongCategory.metrics, { tokensPerSecond: 10 });
  assert.equal(wrongName.name, "telemetry_omitted");
  assert.deepEqual(wrongName.metrics, { tokensPerSecond: 10 });
});

test("journal omits free-form telemetry strings, nested values, arrays, and unknown event names", () => {
  const privateText = "prompt and response text must never reach the journal";
  const sanitized = sanitizeTelemetryEvent({
    schemaVersion: 1,
    category: "generation",
    name: privateText,
    timestampMs: 100,
    metrics: {
      status: privateText,
      reason: privateText,
      code: privateText,
      phase: privateText,
      lifecycle: privateText,
      nested: { status: privateText, cpuBytes: 4096 },
      list: [privateText, { status: privateText }],
      tokensPerSecond: 31.5,
      cacheHit: true,
    },
  });

  const encoded = JSON.stringify(sanitized);
  assert.equal(sanitized.name, "telemetry_omitted");
  assert.deepEqual(sanitized.metrics, { tokensPerSecond: 31.5, cacheHit: true });
  assert.doesNotMatch(encoded, /prompt|response|never|journal|privateText|nested|list/i);
});

test("JSONL is written only inside the caller-supplied run directory", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "qwen-control-journal-"));
  const journal = new RunJournal({ runsDirectory: root, runId: "run_0123456789abcdef" });

  await journal.append({
    schemaVersion: 1,
    category: "memory",
    name: "allocation",
    timestampMs: 100,
    deviceId: "device_0123456789abcdef",
    tabId: "tab_0123456789abcdef",
    documentId: "document_0123456789abcdef",
    eventSeq: 1,
    metrics: { cpuBytes: 1024, gpuBytes: 2048 },
  });
  await journal.close();

  const file = path.join(root, "run_0123456789abcdef.jsonl");
  const content = await readFile(file, "utf8");
  assert.equal(content.trim().split("\n").length, 1);
  assert.doesNotMatch(content, /Users|private|cookie|prompt/i);
});

test("oversized telemetry is bounded before storage", () => {
  const sanitized = sanitizeTelemetryEvent({
    schemaVersion: 1,
    category: "error",
    name: "runtime_error",
    timestampMs: 100,
    metrics: { message: "x".repeat(100_000) },
  });
  assert.ok(Buffer.byteLength(JSON.stringify(sanitized)) <= 16_384);
});

test("journal rotates at its configured cap and summaries aggregate safe metrics", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "qwen-control-rotation-"));
  const runId = "run_0123456789abcdef";
  const journal = new RunJournal({
    runsDirectory: root,
    runId,
    maxSegmentBytes: 700,
    maxSegments: 3,
  });
  for (let index = 0; index < 5; index += 1) {
    await journal.append({
      schemaVersion: 1,
      category: "generation",
      name: "token_rate",
      timestampMs: 100 + index,
      benchmarkId: "benchmark_0123456789abcdef",
      deviceId: "device_0123456789abcdef",
      tabId: "tab_0123456789abcdef",
      documentId: "document_0123456789abcdef",
      eventSeq: index + 1,
      metrics: { tokensPerSecond: 30 + index },
    });
  }
  await journal.close();

  const files = (await readdir(root)).filter((name) => name.endsWith(".jsonl"));
  assert.ok(files.length >= 2);
  assert.ok(files.length <= 3);
  const [summary] = await listRunSummaries(root);
  assert.equal(summary?.runId, runId);
  assert.equal(summary?.eventCount, 5);
  assert.deepEqual(summary?.benchmarkIds, ["benchmark_0123456789abcdef"]);
  assert.deepEqual(summary?.metrics.tokensPerSecond, {
    count: 5,
    min: 30,
    max: 34,
    last: 34,
  });
});

test("journal recovers after one write failure and exposes bounded health", async () => {
  let calls = 0;
  const written: string[] = [];
  const journal = new RunJournal({
    runsDirectory: "ignored-by-injected-writer",
    runId: "run_0123456789abcdef",
    now: () => 1234,
    appendLine: async (_filePath: string, line: string) => {
      calls += 1;
      if (calls === 1) throw new Error("private path and raw write failure");
      written.push(line);
    },
  });
  const event = (eventSeq: number) => ({
    schemaVersion: 1,
    category: "generation",
    name: "token_rate",
    timestampMs: eventSeq,
    deviceId: "device_0123456789abcdef",
    tabId: "tab_0123456789abcdef",
    documentId: "document_0123456789abcdef",
    eventSeq,
    metrics: { tokensPerSecond: 30 },
  });

  await assert.rejects(journal.append(event(1)), /journal append failed/i);
  await journal.append(event(2));
  await journal.close();

  assert.equal(written.length, 1);
  const health = journal.getHealth();
  assert.deepEqual(health, {
    state: "degraded",
    writeFailures: 1,
    lastFailureAtMs: 1234,
  });
  assert.doesNotMatch(JSON.stringify(health), /private|path|raw|write failure/i);
});
