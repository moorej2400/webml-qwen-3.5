import assert from "node:assert/strict";
import test from "node:test";

import {
  createTextRuntimeController,
  type TextRuntimeSession,
} from "../dev/browser/text-runtime.js";
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
