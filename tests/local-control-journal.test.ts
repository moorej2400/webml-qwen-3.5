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

test("telemetry uses an allowlist and recursively removes private content", () => {
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
  assert.match(encoded, /cpuBytes/);
  assert.doesNotMatch(encoded, /private|secret|authorization|cookie|prompt|response|url|stack/i);
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
