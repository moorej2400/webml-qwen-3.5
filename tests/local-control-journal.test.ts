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
