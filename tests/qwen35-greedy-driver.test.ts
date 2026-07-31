import assert from "node:assert/strict";
import test from "node:test";

import {
  createQwen35GreedyTextDriver,
  disposeQwen35GreedyOwnedResources,
  executeQwen35GreedyGpuBatch,
  uploadQwen35GreedyUniformCommands,
  type Qwen35GreedyTokenEngine,
  type Qwen35GreedyTokenStep,
} from "../src/qwen35-greedy-driver.js";
import type { Qwen35DispatchRequest } from "../src/qwen35-webgpu-executor.js";

const request = Object.freeze({}) as Qwen35DispatchRequest;

class FakeTokenEngine implements Qwen35GreedyTokenEngine {
  readonly capacity = 16_384;
  readonly steps: Qwen35GreedyTokenStep[] = [];
  position = 0;
  poisoned = false;
  resetCount = 0;
  disposed = false;
  readonly #predictions: number[];

  constructor(predictions: readonly number[]) {
    this.#predictions = [...predictions];
  }

  async execute(step: Qwen35GreedyTokenStep): Promise<number | null> {
    step.signal.throwIfAborted();
    this.steps.push(step);
    this.position += 1;
    return step.predict ? (this.#predictions.shift() ?? null) : null;
  }

  async reset(): Promise<void> {
    this.position = 0;
    this.resetCount += 1;
  }

  async dispose(): Promise<void> {
    this.disposed = true;
  }
}

test("serial prefill predicts only on the last token and generation reuses it", async () => {
  const engine = new FakeTokenEngine([41, 42, 43]);
  const driver = createQwen35GreedyTextDriver(engine);

  await driver.prefill({
    tokenIds: [11, 12, 13],
    signal: new AbortController().signal,
  });
  assert.deepEqual(engine.steps.map(({ tokenId, predict, phase }) => ({
    tokenId,
    predict,
    phase,
  })), [
    { tokenId: 11, predict: false, phase: "prefill" },
    { tokenId: 12, predict: false, phase: "prefill" },
    { tokenId: 13, predict: true, phase: "prefill" },
  ]);

  const generated: number[] = [];
  for await (const token of driver.generate({
    maxNewTokens: 3,
    signal: new AbortController().signal,
    logitMask: { start: 248_070, count: 250 },
  })) {
    generated.push(token);
  }
  assert.deepEqual(generated, [41, 42, 43]);
  assert.deepEqual(engine.steps.slice(3).map(({ tokenId, predict, phase }) => ({
    tokenId,
    predict,
    phase,
  })), [
    { tokenId: 41, predict: true, phase: "generation" },
    { tokenId: 42, predict: true, phase: "generation" },
  ]);
  await driver.dispose();
  assert.equal(engine.disposed, true);
});

test("rejects sentinel, masked, and malformed token contracts", async () => {
  for (const predicted of [0xffff_ffff, 248_070]) {
    const driver = createQwen35GreedyTextDriver(
      new FakeTokenEngine([predicted]),
    );
    await assert.rejects(
      driver.prefill({
        tokenIds: [1],
        signal: new AbortController().signal,
      }),
      { code: "greedy-token-selection-invalid" },
    );
    await driver.dispose();
  }

  const driver = createQwen35GreedyTextDriver(new FakeTokenEngine([7]));
  await assert.rejects(
    driver.prefill({
      tokenIds: [248_070],
      signal: new AbortController().signal,
    }),
    { code: "greedy-prefill-token-invalid" },
  );
  await driver.prefill({
    tokenIds: [1],
    signal: new AbortController().signal,
  });
  const iterator = driver.generate({
    maxNewTokens: 1,
    signal: new AbortController().signal,
    logitMask: { start: 0, count: 1 },
  })[Symbol.asyncIterator]();
  await assert.rejects(iterator.next(), { code: "greedy-logit-mask-invalid" });
  await driver.dispose();
});

test("continues after a natural token limit without repeating the last token", async () => {
  const engine = new FakeTokenEngine([41, 42, 43]);
  const driver = createQwen35GreedyTextDriver(engine);
  await driver.prefill({
    tokenIds: [1],
    signal: new AbortController().signal,
  });
  const first: number[] = [];
  for await (const token of driver.generate({
    maxNewTokens: 1,
    signal: new AbortController().signal,
    logitMask: { start: 248_070, count: 250 },
  })) first.push(token);
  const continued: number[] = [];
  for await (const token of driver.generate({
    maxNewTokens: 2,
    signal: new AbortController().signal,
    logitMask: { start: 248_070, count: 250 },
  })) continued.push(token);

  assert.deepEqual(first, [41]);
  assert.deepEqual(continued, [42, 43]);
  assert.deepEqual(engine.steps.map(({ tokenId }) => tokenId), [1, 41, 42]);
  await driver.dispose();
});

test("counts a retained pending token against the logical context capacity", async () => {
  const engine = new FakeTokenEngine([41]);
  engine.position = engine.capacity - 2;
  const driver = createQwen35GreedyTextDriver(engine);
  await driver.prefill({
    tokenIds: [1],
    signal: new AbortController().signal,
  });
  for await (const _token of driver.generate({
    maxNewTokens: 1,
    signal: new AbortController().signal,
    logitMask: { start: 248_070, count: 250 },
  })) {
    // Consume the one final logical context slot.
  }
  const iterator = driver.generate({
    maxNewTokens: 1,
    signal: new AbortController().signal,
    logitMask: { start: 248_070, count: 250 },
  })[Symbol.asyncIterator]();
  await assert.rejects(iterator.next(), { code: "greedy-context-full" });
  await driver.dispose();
});

test("does not emit a cached token after prefill fills the context", async () => {
  const engine = new FakeTokenEngine([41]);
  engine.position = engine.capacity - 1;
  const driver = createQwen35GreedyTextDriver(engine);
  await driver.prefill({
    tokenIds: [1],
    signal: new AbortController().signal,
  });
  const iterator = driver.generate({
    maxNewTokens: 1,
    signal: new AbortController().signal,
    logitMask: { start: 248_070, count: 250 },
  })[Symbol.asyncIterator]();
  await assert.rejects(iterator.next(), { code: "greedy-context-full" });
  await driver.dispose();
});

test("continues after cancellation while paused at a yielded token", async () => {
  const engine = new FakeTokenEngine([41, 42]);
  const driver = createQwen35GreedyTextDriver(engine);
  await driver.prefill({
    tokenIds: [1],
    signal: new AbortController().signal,
  });
  const iterator = driver.generate({
    maxNewTokens: 2,
    signal: new AbortController().signal,
    logitMask: { start: 248_070, count: 250 },
  })[Symbol.asyncIterator]();
  assert.deepEqual(await iterator.next(), { done: false, value: 41 });
  await driver.cancel();
  await iterator.return?.();

  const continued: number[] = [];
  for await (const token of driver.generate({
    maxNewTokens: 1,
    signal: new AbortController().signal,
    logitMask: { start: 248_070, count: 250 },
  })) continued.push(token);
  assert.deepEqual(continued, [42]);
  assert.deepEqual(engine.steps.map(({ tokenId }) => tokenId), [1, 41]);
  await driver.dispose();
});

test("cancel fails closed when generation cancellation crosses submitted work", async () => {
  let started!: () => void;
  const inFlight = new Promise<void>((resolve) => { started = resolve; });
  class PoisoningEngine extends FakeTokenEngine {
    override async execute(step: Qwen35GreedyTokenStep): Promise<number | null> {
      if (this.steps.length === 1) {
        started();
        await new Promise<never>((_resolve, reject) => {
          const fail = () => {
            this.poisoned = true;
            reject(new DOMException("Cancelled", "AbortError"));
          };
          if (step.signal.aborted) fail();
          else step.signal.addEventListener("abort", fail, { once: true });
        });
      }
      return super.execute(step);
    }
  }
  const engine = new PoisoningEngine([41]);
  const driver = createQwen35GreedyTextDriver(engine);
  await driver.prefill({
    tokenIds: [1],
    signal: new AbortController().signal,
  });
  const iterator = driver.generate({
    maxNewTokens: 2,
    signal: new AbortController().signal,
    logitMask: { start: 248_070, count: 250 },
  })[Symbol.asyncIterator]();
  assert.deepEqual(await iterator.next(), { done: false, value: 41 });
  const next = iterator.next();
  await inFlight;
  await assert.rejects(driver.cancel(), { code: "greedy-driver-poisoned" });
  await assert.rejects(next, { name: "AbortError" });
  await driver.dispose();
});

test("cancelled partial prefill requires a quiescent reset", async () => {
  let release!: () => void;
  const gate = new Promise<void>((resolve) => { release = resolve; });
  class BlockingEngine extends FakeTokenEngine {
    override async execute(step: Qwen35GreedyTokenStep): Promise<number | null> {
      if (this.steps.length === 1) {
        await Promise.race([
          gate,
          new Promise<never>((_resolve, reject) => {
            step.signal.addEventListener("abort", () => reject(
              new DOMException("Cancelled", "AbortError"),
            ), { once: true });
          }),
        ]);
      }
      return super.execute(step);
    }
  }
  const engine = new BlockingEngine([5]);
  const driver = createQwen35GreedyTextDriver(engine);
  const controller = new AbortController();
  const prefill = driver.prefill({ tokenIds: [1, 2], signal: controller.signal });
  while (engine.steps.length < 1) await Promise.resolve();
  controller.abort();
  await driver.cancel();
  release();
  await assert.rejects(prefill, { name: "AbortError" });
  await assert.rejects(
    driver.prefill({ tokenIds: [3], signal: new AbortController().signal }),
    { code: "greedy-driver-reset-required" },
  );
  await driver.reset();
  assert.equal(engine.resetCount, 1);
  await driver.prefill({
    tokenIds: [3],
    signal: new AbortController().signal,
  });
  await driver.dispose();
});

test("dispose aborts active work and is idempotent", async () => {
  class AbortEngine extends FakeTokenEngine {
    override async execute(step: Qwen35GreedyTokenStep): Promise<number | null> {
      await new Promise<never>((_resolve, reject) => {
        step.signal.addEventListener("abort", () => reject(
          new DOMException("Cancelled", "AbortError"),
        ), { once: true });
      });
    }
  }
  const engine = new AbortEngine([]);
  const driver = createQwen35GreedyTextDriver(engine);
  const prefill = driver.prefill({
    tokenIds: [1],
    signal: new AbortController().signal,
  });
  await Promise.resolve();
  await Promise.all([driver.dispose(), driver.dispose()]);
  await assert.rejects(prefill, { name: "AbortError" });
  assert.equal(engine.disposed, true);
  await assert.rejects(
    driver.reset(),
    { code: "greedy-driver-disposed" },
  );
});

test("GPU batch submits, retires, advances, then reads only one selected u32", async () => {
  const events: string[] = [];
  let poisoned = false;
  const selectedBuffer = {};
  const result = await executeQwen35GreedyGpuBatch({
    commands: [request],
    executor: {
      async dispatchBatch(commands) {
        assert.deepEqual(commands, [request]);
        events.push("submit");
      },
      async submittedWorkDone() { events.push("retire"); },
      async readU32(buffer, offset) {
        assert.equal(buffer, selectedBuffer);
        assert.equal(offset, 12);
        events.push("read-u32");
        return 77;
      },
    },
    state: {
      advance(tokens) {
        assert.equal(tokens, 1);
        events.push("advance");
      },
    },
    selected: { buffer: selectedBuffer, offset: 12 },
    step: {
      tokenId: 1,
      predict: true,
      phase: "prefill",
      signal: new AbortController().signal,
    },
    poison: () => { poisoned = true; },
  });
  assert.equal(result, 77);
  assert.equal(poisoned, false);
  assert.deepEqual(events, ["submit", "retire", "advance", "read-u32"]);
});

test("uploads interleaved logits commands to their planner-assigned slots", () => {
  const buffer = {};
  const updates: number[][] = [[], [], [], []];
  const slots = updates.map((values, index) => ({
    binding: { buffer, offset: index * 256, byteLength: 32 },
    update(words: Uint32Array<ArrayBuffer>) {
      values.push(...words);
    },
  }));
  const command = (
    slot: number,
    words: readonly number[],
  ) => ({
    uniformWords: words,
    bindings: [{
      binding: 3,
      kind: "uniform" as const,
      buffer,
      offset: slot * 256,
      size: words.length * 4,
    }],
    kernel: { id: "test-kernel", source: "source", entryPoint: "main" },
    workgroups: { x: 1, y: 1, z: 1 },
  });

  // Logits reserves all GEMV slots first, then all reduction slots, even
  // though execution interleaves each tile's GEMV and reduction command.
  uploadQwen35GreedyUniformCommands({
    commands: [
      command(0, [10]),
      command(2, [20]),
      command(1, [11]),
      command(3, [21]),
    ],
    slots,
    expectedSlotCount: 4,
  });

  assert.deepEqual(updates.map((words) => words.slice(0, 2)), [
    [10, 0],
    [11, 0],
    [20, 0],
    [21, 0],
  ]);
});

test("GPU batch sanitizes every ambiguous post-submit failure and poisons", async () => {
  const privateMarker = "PRIVATE_DEVICE_MARKER";
  for (const failure of ["submit", "retire", "advance", "read"] as const) {
    let poisoned = false;
    const executor = {
      async dispatchBatch() {
        if (failure === "submit") throw new Error(privateMarker);
      },
      async submittedWorkDone() {
        if (failure === "retire") throw new Error(privateMarker);
      },
      async readU32() {
        if (failure === "read") throw new Error(privateMarker);
        return 7;
      },
    };
    const operation = executeQwen35GreedyGpuBatch({
      commands: [request],
      executor,
      state: {
        advance() {
          if (failure === "advance") throw new Error(privateMarker);
        },
      },
      selected: { buffer: {}, offset: 0 },
      step: {
        tokenId: 1,
        predict: true,
        phase: "generation",
        signal: new AbortController().signal,
      },
      poison: () => { poisoned = true; },
    });
    await assert.rejects(operation, (error: unknown) => {
      assert.equal((error as { code?: string }).code, "greedy-gpu-batch-failed");
      assert.doesNotMatch((error as Error).message, /PRIVATE_DEVICE_MARKER/);
      return true;
    });
    assert.equal(poisoned, true, failure);
  }
});

test("owned GPU cleanup fences the executor before releasing bound arenas", async () => {
  const events: string[] = [];
  let finishFence!: () => void;
  const fence = new Promise<void>((resolve) => { finishFence = resolve; });
  const cleanup = disposeQwen35GreedyOwnedResources({
    executor: {
      async dispose() {
        events.push("executor-start");
        await fence;
        events.push("executor-end");
      },
    },
    uniformArena: {
      async dispose() { events.push("uniform"); },
    },
    workspace: {
      async dispose() { events.push("workspace"); },
    },
  });
  await Promise.resolve();
  assert.deepEqual(events, ["executor-start"]);
  finishFence();
  await cleanup;
  assert.deepEqual(events, [
    "executor-start",
    "executor-end",
    "uniform",
    "workspace",
  ]);
});

test("cleanup attempts every owner and never exposes a private failure", async () => {
  const events: string[] = [];
  await assert.rejects(
    disposeQwen35GreedyOwnedResources({
      executor: {
        async dispose() {
          events.push("executor");
          throw new Error("PRIVATE_DEVICE_MARKER");
        },
      },
      uniformArena: {
        async dispose() { events.push("uniform"); },
      },
      workspace: {
        async dispose() {
          events.push("workspace");
          throw new Error("PRIVATE_DEVICE_MARKER");
        },
      },
    }),
    (error: unknown) => {
      assert.equal((error as { code?: string }).code, "greedy-engine-cleanup-failed");
      assert.doesNotMatch((error as Error).message, /PRIVATE_DEVICE_MARKER/);
      return true;
    },
  );
  assert.deepEqual(events, ["executor", "uniform", "workspace"]);
});
