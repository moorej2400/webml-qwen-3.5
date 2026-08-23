import assert from "node:assert/strict";
import test from "node:test";

import {
  createTextRuntimeController,
  type TextRuntimeRunMetrics,
  type TextRuntimeSession,
} from "../dev/browser/text-runtime.js";
import type { Qwen35RuntimeCoordinator } from "../src/chat-app.js";
import type { Qwen35BrowserLoadOptions } from "../src/qwen35-model-loader.js";
import type { RuntimeMetrics } from "../src/qwen35-session.js";

const metrics = (state: RuntimeMetrics["state"]): RuntimeMetrics => ({
  state,
  phases: {},
  cacheHit: true,
  trackedCpuBytes: 100,
  trackedGpuBytes: 200,
  peakTrackedGpuBytes: 300,
  contextTokens: 4,
  timeToFirstTokenMilliseconds: 5,
  prefillTokens: 4,
  prefillTokensPerSecond: 10,
  generatedTokens: 2,
  generatedTokensPerSecond: 3,
  cancellationCount: 0,
  deviceLostCount: 0,
});

const performanceMetrics = (input: Partial<NonNullable<RuntimeMetrics["performance"]>> = {}) => ({
  permanentGpuBytes: 0,
  transientGpuBytes: 0,
  stateGpuBytes: 0,
  diskReadBytes: 0,
  gpuUploadBytes: 0,
  dispatchCount: 0,
  queueSubmissionCount: 0,
  queueRetirementCount: 0,
  gpuReadbackCount: 0,
  warmTimeToFirstTokenMilliseconds: null,
  prefillTokensPerSecond: null,
  generatedTokensPerSecond: null,
  millisecondsPerGeneratedToken: null,
  ...input,
});

test("text runtime uses one fixed session for load, prompt streaming, reset, and lifecycle commands", async () => {
  const calls: string[] = [];
  let state: RuntimeMetrics["state"] = "idle";
  const session: TextRuntimeSession = {
    get state() { return state; },
    async load(options) {
      calls.push(`load:${String(options.packageBaseUrl)}`);
      state = "ready";
    },
    async prefill(input) {
      calls.push(`prefill:${String(input[0]?.content)}`);
      return { rendered: "private-rendered-prompt", contextTokens: 4, remainingContextTokens: 16_380 };
    },
    async *generate(options) {
      calls.push(`generate:${options.maxNewTokens}`);
      yield { id: 1, text: "A", index: 0 };
      yield { id: 2, text: "B", index: 1 };
    },
    async cancel() { calls.push("cancel"); },
    async reset() { calls.push("reset"); },
    async dispose() { calls.push("dispose"); state = "disposed"; },
    getMetrics() { return metrics(state); },
  };
  const loadOptions = Object.freeze({ packageBaseUrl: "https://example.invalid/pinned/" }) as Qwen35BrowserLoadOptions;
  let output = "";
  const controller = createTextRuntimeController({
    session,
    loadOptions,
    onText: (text) => { output += text; },
  });

  await controller.load();
  await controller.runPrompt({ prompt: "first", maxNewTokens: 12 });
  await controller.runPrompt({ prompt: "second", maxNewTokens: 8 });
  await controller.cancelPrompt();
  await controller.dispose();

  assert.equal(output, "ABAB");
  assert.deepEqual(calls, [
    "load:https://example.invalid/pinned/",
    "prefill:first",
    "generate:12",
    "reset",
    "prefill:second",
    "generate:8",
    "cancel",
    "dispose",
  ]);
});

test("text runtime accepts only a bounded prompt payload and reports structured state without text", async () => {
  const session: TextRuntimeSession = {
    state: "ready",
    async load() {},
    async prefill() {
      return { rendered: "secret rendered prompt", contextTokens: 4, remainingContextTokens: 16_380 };
    },
    async *generate() { yield { id: 1, text: "secret output", index: 0 }; },
    async cancel() {},
    async reset() {},
    async dispose() {},
    getMetrics() { return metrics("ready"); },
  };
  const controller = createTextRuntimeController({
    session,
    loadOptions: {} as Qwen35BrowserLoadOptions,
  });

  await assert.rejects(controller.runPrompt(undefined), /payload/i);
  await assert.rejects(controller.runPrompt({ prompt: "" }), /prompt/i);
  await assert.rejects(controller.runPrompt({ prompt: "ok", maxNewTokens: 0 }), /token/i);
  await assert.rejects(
    controller.runPrompt({ prompt: "ok", maxNewTokens: 1, packageBaseUrl: "https://forged.invalid" }),
    /field/i,
  );

  const state = controller.getState();
  assert.deepEqual(state, {
    modelState: "ready",
    generationState: "idle",
    cacheState: "hit",
    deviceState: "ready",
    loaded: true,
    generating: false,
    contextTokens: 4,
    maxContextTokens: 16_384,
    cpuBytes: 100,
    gpuBytes: 200,
  });
  assert.doesNotMatch(JSON.stringify(state), /secret|prompt|output/i);
});

test("text runtime reports per-command numeric measurements without prompt or output text", async () => {
  let stage = 0;
  const reported: unknown[] = [];
  const before = {
    ...metrics("ready"),
    prefillTokens: 10,
    generatedTokens: 4,
    performance: performanceMetrics({
      diskReadBytes: 100,
      gpuUploadBytes: 200,
      dispatchCount: 30,
      queueSubmissionCount: 4,
      queueRetirementCount: 4,
      gpuReadbackCount: 1,
    }),
    phases: {
      prefill: { count: 1, totalMilliseconds: 100, lastMilliseconds: 100 },
      generate: { count: 1, totalMilliseconds: 80, lastMilliseconds: 80 },
    },
  } as const;
  const afterPrefill = {
    ...before,
    prefillTokens: 16,
    performance: performanceMetrics({
      diskReadBytes: 400,
      gpuUploadBytes: 500,
      dispatchCount: 60,
      queueSubmissionCount: 10,
      queueRetirementCount: 10,
      gpuReadbackCount: 1,
    }),
    phases: {
      ...before.phases,
      prefill: { count: 2, totalMilliseconds: 220, lastMilliseconds: 120 },
    },
  } as const;
  const afterGenerate = {
    ...afterPrefill,
    generatedTokens: 7,
    timeToFirstTokenMilliseconds: 35,
    performance: performanceMetrics({
      diskReadBytes: 500,
      gpuUploadBytes: 600,
      dispatchCount: 80,
      queueSubmissionCount: 16,
      queueRetirementCount: 16,
      gpuReadbackCount: 2,
    }),
    phases: {
      ...afterPrefill.phases,
      generate: { count: 2, totalMilliseconds: 280, lastMilliseconds: 200 },
    },
  } as const;
  const session: TextRuntimeSession = {
    state: "ready",
    async load() {},
    async prefill() {
      stage = 1;
      return { rendered: "private rendered prompt", contextTokens: 16, remainingContextTokens: 16_368 };
    },
    async *generate() {
      stage = 2;
      yield { id: 1, text: "private", index: 0 };
      yield { id: 2, text: " output", index: 1 };
      yield { id: 3, text: " text", index: 2 };
    },
    async cancel() {},
    async reset() {},
    async dispose() {},
    getMetrics() {
      return stage === 0 ? before : stage === 1 ? afterPrefill : afterGenerate;
    },
  };
  const controller = createTextRuntimeController({
    session,
    loadOptions: {} as Qwen35BrowserLoadOptions,
    now: (() => {
      const values = [0, 0, 1, 1, 41, 41, 81, 81, 82];
      return () => values.shift() ?? 82;
    })(),
    onRunMetrics(measurement) { reported.push(measurement); },
  });

  await controller.runPrompt({ prompt: "private operator prompt", maxNewTokens: 3 });

  assert.deepEqual(reported, [{
    contextTokens: 4,
    trackedCpuBytes: 100,
    trackedGpuBytes: 200,
    peakTrackedGpuBytes: 300,
    prefillTokens: 6,
    prefillDurationMilliseconds: 120,
    prefillTokensPerSecond: 50,
    generatedTokens: 3,
    targetStepCount: 2,
    targetStepDurationMilliseconds: 80,
    targetStepsPerSecond: 25,
    generationDurationMilliseconds: 200,
    generatedTokensPerSecond: 15,
    timeToFirstTokenMilliseconds: 1,
    decodedTextCodeUnits: 19,
    referenceTokenCount: 0,
    referenceTokenMismatchCount: 0,
    performanceSnapshotCount: 2,
    performance: {
      diskReadBytes: 400,
      gpuUploadBytes: 400,
      dispatchCount: 50,
      queueSubmissionCount: 12,
      queueRetirementCount: 12,
      gpuReadbackCount: 1,
    },
  }]);
  assert.doesNotMatch(JSON.stringify(reported), /private|output|prompt/i);
});

test("shared coordinator load is idempotent while an existing runtime is busy", async () => {
  let loadCalls = 0;
  const coordinator: Qwen35RuntimeCoordinator = {
    state: "generating",
    async load() { loadCalls += 1; },
    async replaceConversation() {
      return { rendered: "", contextTokens: 0, remainingContextTokens: 16_384 };
    },
    async *generate() {},
    async cancel() {},
    async reset() {},
    async dispose() {},
    getMetrics() { return metrics("generating"); },
    subscribeLoadEvents() { return () => undefined; },
  };
  const controller = createTextRuntimeController({ coordinator });

  await controller.load();

  assert.equal(loadCalls, 0, "load must not replace a shared runtime during generation");
});

test("controlled oracle disables thinking and rejects any token mismatch", async () => {
  let thinking: boolean | undefined;
  let observedMetrics: TextRuntimeRunMetrics | undefined;
  const coordinator: Qwen35RuntimeCoordinator = {
    state: "ready",
    async load() {},
    async replaceConversation(_input, options) {
      thinking = options?.enableThinking;
      return { rendered: "", contextTokens: 0, remainingContextTokens: 16_384 };
    },
    async *generate() { yield { id: 1, text: "x", index: 0 }; },
    async cancel() {},
    async reset() {},
    async dispose() {},
    getMetrics() { return metrics("ready"); },
    subscribeLoadEvents() { return () => undefined; },
  };
  const controller = createTextRuntimeController({
    coordinator,
    onRunMetrics(metrics) { observedMetrics = metrics; },
  });

  await assert.rejects(
    controller.runPrompt({
      prompt: "Write one short sentence about WebGPU.",
      maxNewTokens: 16,
    }),
    (error: unknown) =>
      (error as { readonly code?: unknown }).code === "reference-token-mismatch",
  );
  assert.equal(thinking, false);
  assert.equal(observedMetrics?.referenceFirstMismatchIndex, 0);
  assert.equal(observedMetrics?.referenceExpectedTokenId, 5_793);
  assert.equal(observedMetrics?.referenceObservedTokenId, 1);
});

test("controlled oracle counts a decoder flush as text but not a model token", async () => {
  const tokenIds = [
    5_793, 48_213, 369, 264, 6_278, 13_775, 5_165, 5_995,
    310, 3_300, 1_496, 55_549, 11, 3_238, 11_258, 2_528,
  ];
  let stage = 0;
  let output = "";
  const session: TextRuntimeSession = {
    state: "ready",
    async load() {},
    async prefill(_input, options) {
      assert.equal(options?.enableThinking, false);
      stage = 1;
      return { rendered: "", contextTokens: 0, remainingContextTokens: 16_384 };
    },
    async *generate() {
      for (const [index, id] of tokenIds.entries()) {
        yield { id, text: "x", index };
      }
      yield { id: tokenIds.at(-1)!, text: "!", index: tokenIds.length - 1 };
      stage = 2;
    },
    async cancel() {},
    async reset() {},
    async dispose() {},
    getMetrics() {
      return { ...metrics("ready"), generatedTokens: stage === 2 ? 16 : 0 };
    },
  };
  const controller = createTextRuntimeController({
    session,
    loadOptions: {} as Qwen35BrowserLoadOptions,
    onText(text) { output += text; },
  });

  await controller.runPrompt({
    prompt: "Write one short sentence about WebGPU.",
    maxNewTokens: 16,
  });
  assert.equal(output, `${"x".repeat(16)}!`);
});
