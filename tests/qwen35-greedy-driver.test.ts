import assert from "node:assert/strict";
import test from "node:test";

import {
  createQwen35GreedyTextDriver,
  disposeQwen35GreedyOwnedResources,
  ensureQwen35GreedyTokenState,
  executeQwen35GreedyGpuBatch,
  executeQwen35GreedyRollingLayerDispatch,
  uploadQwen35GreedyUniformCommands,
  type Qwen35GreedyTokenEngine,
  type Qwen35GreedyTokenStep,
} from "../src/qwen35-greedy-driver.js";
import type { Qwen35DispatchRequest } from "../src/qwen35-webgpu-executor.js";
import type {
  Qwen35DiskBackedTiedEmbeddingStore,
  Qwen35LogitCandidate,
  Qwen35StagedPackedRows,
} from "../src/qwen35-disk-backed-tied-embedding.js";
import type { Qwen35ForwardBufferSlice } from "../src/qwen35-forward-dispatch.js";

interface DiskBackedGreedySubject {
  uploadQwen35GreedyRollingPlanUniforms(input: {
    readonly planUniformCount: number;
    readonly reservedUniformCount: number;
    readonly commands: readonly (Qwen35DispatchRequest & {
      readonly uniformWords: readonly number[];
    })[];
    readonly slots: readonly {
      readonly binding: Qwen35ForwardBufferSlice;
      update(words: Uint32Array<ArrayBuffer>): void;
    }[];
  }): void;
  stageQwen35GreedyInputEmbedding(input: {
    readonly tiedEmbedding: Pick<
      Qwen35DiskBackedTiedEmbeddingStore,
      "stageInputRow"
    >;
    readonly step: Qwen35GreedyTokenStep;
  }): Promise<
    | { readonly kind: "visual"; readonly source: Qwen35ForwardBufferSlice }
    | { readonly kind: "packed"; readonly rows: Qwen35StagedPackedRows }
  >;
  selectQwen35GreedyTiedToken(input: {
    readonly tiedEmbedding: Pick<
      Qwen35DiskBackedTiedEmbeddingStore,
      "selectTopK"
    >;
    readonly phase: Qwen35GreedyTokenStep["phase"];
    readonly signal: AbortSignal;
    readonly scoreTile: (
      tile: Qwen35StagedPackedRows,
    ) => Promise<readonly Qwen35LogitCandidate[]> | readonly Qwen35LogitCandidate[];
  }): Promise<number>;
  executeQwen35GreedyTiedTileScore(input: {
    readonly commands: readonly Qwen35DispatchRequest[];
    readonly executor: {
      dispatchBatch(commands: readonly Qwen35DispatchRequest[]): Promise<void>;
      submittedWorkDone(): Promise<void>;
      readU32(buffer: object, byteOffset: number): Promise<number>;
    };
    readonly tile: Qwen35StagedPackedRows;
    readonly candidateScoreReadback: { readonly buffer: object; readonly offset: number };
    readonly candidateTokenReadback: { readonly buffer: object; readonly offset: number };
    readonly signal: AbortSignal;
  }): Promise<readonly Qwen35LogitCandidate[]>;
}

async function diskBackedGreedySubject(): Promise<DiskBackedGreedySubject> {
  return await import("../src/qwen35-greedy-driver.js") as unknown as DiskBackedGreedySubject;
}

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

test("makes the next token state resident before execution can continue", async () => {
  let release!: () => void;
  const gate = new Promise<void>((resolve) => { release = resolve; });
  const events: string[] = [];
  const signal = new AbortController().signal;
  const state = {
    position: 256,
    async ensureCapacity(requiredEnd: number, receivedSignal?: AbortSignal) {
      assert.equal(requiredEnd, 257);
      assert.equal(receivedSignal, signal);
      events.push("ensure-start");
      await gate;
      events.push("ensure-end");
    },
  };

  const residency = ensureQwen35GreedyTokenState(state, signal);
  await Promise.resolve();
  assert.deepEqual(events, ["ensure-start"]);
  release();
  await residency;
  assert.deepEqual(events, ["ensure-start", "ensure-end"]);
});

test("state residency rejects cancellation before and after page growth", async () => {
  const before = new AbortController();
  before.abort();
  let called = false;
  await assert.rejects(
    ensureQwen35GreedyTokenState({
      position: 0,
      async ensureCapacity() { called = true; },
    }, before.signal),
    { name: "AbortError" },
  );
  assert.equal(called, false);

  const during = new AbortController();
  await assert.rejects(
    ensureQwen35GreedyTokenState({
      position: 0,
      async ensureCapacity() { during.abort(); },
    }, during.signal),
    { name: "AbortError" },
  );
});

test("rejects an invalid state position before requesting GPU pages", async () => {
  for (const position of [-1, 1.5, Number.NaN]) {
    let called = false;
    await assert.rejects(
      ensureQwen35GreedyTokenState({
        position,
        async ensureCapacity() { called = true; },
      }, new AbortController().signal),
      { code: "greedy-state-position-invalid" },
    );
    assert.equal(called, false);
  }
});

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

test("expands one image marker into projected visual rows during prefill", async () => {
  const engine = new FakeTokenEngine([41]);
  const driver = createQwen35GreedyTextDriver(engine);
  const source = { buffer: {}, offset: 0, byteLength: 2 * 2_560 * 4 };
  await driver.prefill({
    tokenIds: [11, 999_999, 12],
    visualEmbeddings: [{ tokenId: 999_999, tokenCount: 2, source }],
    signal: new AbortController().signal,
  });
  assert.equal(engine.steps.length, 4);
  assert.equal(engine.steps[1]!.embeddingOverride?.offset, 0);
  assert.equal(engine.steps[2]!.embeddingOverride?.offset, 2_560 * 4);
  assert.equal(engine.steps[3]!.predict, true);
  assert.equal(engine.steps[3]!.embeddingOverride, undefined);
});

test("stages text rows from disk while visual rows bypass the tied store", async () => {
  const subject = await diskBackedGreedySubject();
  assert.equal(
    typeof subject.stageQwen35GreedyInputEmbedding,
    "function",
    "the greedy driver must expose its disk-backed input boundary",
  );
  const calls: Array<{
    readonly tokenId: number;
    readonly phase: "prefill" | "decode";
    readonly signal: AbortSignal;
  }> = [];
  const staged: Qwen35StagedPackedRows = Object.freeze({
    tensorName: "token_embd.weight",
    storageType: "q6-k-212",
    firstRow: 17,
    rowCount: 1,
    rowBytes: 2_120,
    buffer: {},
    bufferOffset: 2_120,
    byteLength: 2_120,
  });
  const tiedEmbedding = {
    async stageInputRow(input: typeof calls[number]) {
      calls.push(input);
      return staged;
    },
  };
  const prefillSignal = new AbortController().signal;
  const prefill = await subject.stageQwen35GreedyInputEmbedding({
    tiedEmbedding,
    step: {
      tokenId: 17,
      predict: false,
      phase: "prefill",
      signal: prefillSignal,
    },
  });
  assert.deepEqual(prefill, { kind: "packed", rows: staged });
  assert.deepEqual(calls, [{ tokenId: 17, phase: "prefill", signal: prefillSignal }]);

  const decodeSignal = new AbortController().signal;
  const decode = await subject.stageQwen35GreedyInputEmbedding({
    tiedEmbedding,
    step: {
      tokenId: 23,
      predict: true,
      phase: "generation",
      signal: decodeSignal,
    },
  });
  assert.deepEqual(decode, { kind: "packed", rows: staged });
  assert.deepEqual(calls.at(-1), {
    tokenId: 23,
    phase: "decode",
    signal: decodeSignal,
  });

  const visual: Qwen35ForwardBufferSlice = {
    buffer: {},
    offset: 32,
    byteLength: 2_560 * 4,
  };
  const beforeVisual = calls.length;
  const visualResult = await subject.stageQwen35GreedyInputEmbedding({
    tiedEmbedding,
    step: {
      tokenId: 0,
      embeddingOverride: visual,
      predict: false,
      phase: "prefill",
      signal: new AbortController().signal,
    },
  });
  assert.deepEqual(visualResult, { kind: "visual", source: visual });
  assert.equal(calls.length, beforeVisual);
});

test("selects one tied token through the disk-backed output boundary", async () => {
  const subject = await diskBackedGreedySubject();
  assert.equal(
    typeof subject.selectQwen35GreedyTiedToken,
    "function",
    "the greedy driver must expose its disk-backed output boundary",
  );
  const signal = new AbortController().signal;
  const tile: Qwen35StagedPackedRows = Object.freeze({
    tensorName: "token_embd.weight",
    storageType: "q6-k-212",
    firstRow: 40,
    rowCount: 2,
    rowBytes: 2_120,
    buffer: {},
    bufferOffset: 0,
    byteLength: 4_240,
  });
  const scorer = async (received: Qwen35StagedPackedRows) => {
    assert.equal(received, tile);
    return [{ tokenId: 41, score: 9 }];
  };
  const calls: unknown[] = [];
  const selected = await subject.selectQwen35GreedyTiedToken({
    tiedEmbedding: {
      async selectTopK(input) {
        calls.push(input);
        return await input.scoreTile(tile);
      },
    },
    phase: "generation",
    signal,
    scoreTile: scorer,
  });
  assert.equal(selected, 41);
  assert.equal(calls.length, 1);
  const request = calls[0] as {
    readonly phase: string;
    readonly topK: number;
    readonly signal: AbortSignal;
    readonly scoreTile: unknown;
  };
  assert.equal(request.phase, "decode");
  assert.equal(request.topK, 1);
  assert.equal(request.signal, signal);
  assert.equal(request.scoreTile, scorer);
});

test("executes a staged logits tile without advancing recurrent state", async () => {
  const subject = await diskBackedGreedySubject();
  assert.equal(
    typeof subject.executeQwen35GreedyTiedTileScore,
    "function",
    "the output path must execute and read each real staged tile",
  );
  const command = Object.freeze({}) as Qwen35DispatchRequest;
  const candidateBuffer = {};
  const events: string[] = [];
  const scoreBits = new Uint32Array(new Float32Array([3.5]).buffer)[0]!;
  const tile: Qwen35StagedPackedRows = {
    tensorName: "token_embd.weight",
    storageType: "q6-k-212",
    firstRow: 1_024,
    rowCount: 1_024,
    rowBytes: 2_120,
    buffer: {},
    bufferOffset: 0,
    byteLength: 2_120 * 1_024,
  };
  const candidates = await subject.executeQwen35GreedyTiedTileScore({
    commands: [command],
    executor: {
      async dispatchBatch(commands) {
        assert.deepEqual(commands, [command]);
        events.push("submit");
      },
      async submittedWorkDone() { events.push("retire"); },
      async readU32(buffer, byteOffset) {
        assert.equal(buffer, candidateBuffer);
        if (byteOffset === 4) {
          events.push("read-token");
          return 1_025;
        }
        assert.equal(byteOffset, 0);
        events.push("read-score");
        return scoreBits;
      },
    },
    tile,
    candidateScoreReadback: { buffer: candidateBuffer, offset: 0 },
    candidateTokenReadback: { buffer: candidateBuffer, offset: 4 },
    signal: new AbortController().signal,
  });

  assert.deepEqual(candidates, [{ tokenId: 1_025, score: 3.5 }]);
  assert.deepEqual(events.slice(0, 2), ["submit", "retire"]);
  assert.deepEqual(events.slice(2).sort(), ["read-score", "read-token"]);
});

test("rejects an empty or masked disk-backed tied selection", async () => {
  const subject = await diskBackedGreedySubject();
  for (const candidates of [
    [] as const,
    [{ tokenId: 248_070, score: 1 }] as const,
  ]) {
    await assert.rejects(
      subject.selectQwen35GreedyTiedToken({
        tiedEmbedding: {
          async selectTopK() { return candidates; },
        },
        phase: "prefill",
        signal: new AbortController().signal,
        scoreTile: () => candidates,
      }),
      { code: "greedy-token-selection-invalid" },
    );
  }
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

test("rolling layer dispatch marks persistent mutation before submission and retires", async () => {
  const events: string[] = [];
  await executeQwen35GreedyRollingLayerDispatch({
    commands: Object.freeze([]),
    executor: {
      async dispatchBatch() { events.push("dispatch"); },
      async submittedWorkDone() { events.push("retire"); },
    },
    mutation: {
      markStateMutation() { events.push("mutation"); },
    },
    signal: new AbortController().signal,
  });
  assert.deepEqual(events, ["mutation", "dispatch", "retire"]);

  const cancelled = new AbortController();
  cancelled.abort();
  await assert.rejects(executeQwen35GreedyRollingLayerDispatch({
    commands: Object.freeze([]),
    executor: {
      async dispatchBatch() { assert.fail("cancelled work must not submit"); },
      async submittedWorkDone() { assert.fail("cancelled work must not retire"); },
    },
    mutation: {
      markStateMutation() { assert.fail("cancelled work must not mutate"); },
    },
    signal: cancelled.signal,
  }), (error: unknown) => error instanceof DOMException && error.name === "AbortError");
});

test("rolling full-attention uploads its 14 used uniforms inside 77 reserved slots", async () => {
  const subject = await diskBackedGreedySubject();
  assert.equal(
    typeof subject.uploadQwen35GreedyRollingPlanUniforms,
    "function",
    "rolling uniform capacity must be distinct from actual page usage",
  );
  const buffer = {};
  const updates: number[][] = Array.from({ length: 77 }, () => []);
  const slots = updates.map((values, index) => ({
    binding: { buffer, offset: index * 256, byteLength: 32 },
    update(words: Uint32Array<ArrayBuffer>) { values.push(...words); },
  }));
  const commands = Array.from({ length: 14 }, (_, index) => ({
    uniformWords: [index + 1],
    bindings: [{
      binding: 3,
      kind: "uniform" as const,
      buffer,
      offset: index * 256,
      size: 4,
    }],
    kernel: { id: `rolling-${index}`, source: "source", entryPoint: "main" },
    workgroups: { x: 1, y: 1, z: 1 },
  }));

  subject.uploadQwen35GreedyRollingPlanUniforms({
    planUniformCount: 14,
    reservedUniformCount: 77,
    commands,
    slots,
  });
  assert.deepEqual(
    updates.map((words) => words[0] ?? 0),
    [...Array.from({ length: 14 }, (_, index) => index + 1), ...Array(63).fill(0)],
  );
  assert.throws(() => subject.uploadQwen35GreedyRollingPlanUniforms({
    planUniformCount: 78,
    reservedUniformCount: 77,
    commands,
    slots,
  }), { code: "greedy-uniform-schedule-invalid" });
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

test("owned GPU cleanup releases the staged-candidate scratch after its fence", async () => {
  const events: string[] = [];
  const cleanup = disposeQwen35GreedyOwnedResources as unknown as (input: {
    readonly executor: { dispose(): Promise<void> };
    readonly uniformArena: null;
    readonly workspace: null;
    readonly candidateScratch: { destroy(): void };
  }) => Promise<void>;

  await cleanup({
    executor: {
      async dispose() { events.push("executor"); },
    },
    uniformArena: null,
    workspace: null,
    candidateScratch: {
      destroy() { events.push("candidate-scratch"); },
    },
  });

  assert.deepEqual(events, ["executor", "candidate-scratch"]);
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
