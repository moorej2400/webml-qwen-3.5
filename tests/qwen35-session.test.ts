import assert from "node:assert/strict";
import test from "node:test";

import { RuntimeDiagnosticError } from "../src/diagnostics.js";
import type { ExclusiveLockManager } from "../src/origin-model-lock.js";
import { Qwen35Tokenizer } from "../src/qwen-tokenizer.js";
import {
  Qwen35Session,
  type Qwen35ExecutionDriver,
  type Qwen35LoadedResources,
  type Qwen35SessionRuntime,
} from "../src/qwen35-session.js";
import type { CompiledTokenizerTables } from "../src/tokenizer-binary.js";

function tokenizer(): Qwen35Tokenizer {
  const added = [
    "<|im_start|>",
    "<|im_end|>",
    "<think>",
    "</think>",
    "<|vision_start|>",
    "<|image_pad|>",
    "<|vision_end|>",
  ];
  const parts = [
    ...Array.from({ length: 256 }, (_, byte) => Uint8Array.of(byte)),
    ...added.map((value) => new TextEncoder().encode(value)),
  ];
  const offsets = new Uint32Array(parts.length + 1);
  const bytes = new Uint8Array(
    parts.reduce((sum, part) => sum + part.byteLength, 0),
  );
  let offset = 0;
  for (const [index, part] of parts.entries()) {
    offsets[index] = offset;
    bytes.set(part, offset);
    offset += part.byteLength;
  }
  offsets[parts.length] = offset;
  const tables: CompiledTokenizerTables = {
    baseVocabSize: 256,
    tokenCount: parts.length,
    tokenOffsets: offsets,
    tokenBytes: bytes,
    merges: new Uint32Array(),
    addedTokenIds: Uint32Array.from(added.map((_, index) => 256 + index)),
    addedTokenFlags: new Uint8Array(added.length).fill(1),
  };
  return Qwen35Tokenizer.fromUnsafeTablesForTests(tables);
}

class ImmediateLockManager implements ExclusiveLockManager {
  constructor(readonly events: string[] = []) {}

  async request<T>(
    _name: string,
    _options: { mode: "exclusive"; signal: AbortSignal },
    callback: () => Promise<T> | T,
  ): Promise<T> {
    this.events.push("lock");
    return callback();
  }
}

class QueuedLockManager implements ExclusiveLockManager {
  #active = false;
  readonly #pending: Array<{
    signal: AbortSignal;
    callback: () => Promise<unknown> | unknown;
    resolve(value: unknown): void;
    reject(reason: unknown): void;
  }> = [];

  request<T>(
    _name: string,
    options: { mode: "exclusive"; signal: AbortSignal },
    callback: () => Promise<T> | T,
  ): Promise<T> {
    return new Promise<T>((resolve, reject) => {
      const entry = {
        signal: options.signal,
        callback,
        resolve: resolve as (value: unknown) => void,
        reject,
      };
      options.signal.addEventListener("abort", () => {
        const index = this.#pending.indexOf(entry);
        if (index >= 0) {
          this.#pending.splice(index, 1);
          reject(new DOMException("aborted", "AbortError"));
        }
      }, { once: true });
      this.#pending.push(entry);
      this.#drain();
    });
  }

  #drain(): void {
    if (this.#active) return;
    const entry = this.#pending.shift();
    if (entry === undefined) return;
    this.#active = true;
    Promise.resolve(entry.callback()).then(entry.resolve, entry.reject).finally(() => {
      this.#active = false;
      this.#drain();
    });
  }
}

function runtime(
  driver: Qwen35ExecutionDriver,
  events: string[] = [],
  manager = new ImmediateLockManager(),
): Qwen35SessionRuntime {
  return {
    lockManager: manager,
    now: () => 10,
    async load(signal): Promise<Qwen35LoadedResources> {
      signal.throwIfAborted();
      events.push("load");
      return {
        tokenizer: tokenizer(),
        driver,
        cacheHit: true,
        trackedCpuBytes: 12,
        trackedGpuBytes: 34,
        async dispose() {
          events.push("resources-dispose");
          await driver.dispose();
        },
      };
    },
  };
}

function driver(ids: readonly number[] = [97, 226, 130, 172]): {
  driver: Qwen35ExecutionDriver;
  events: string[];
  masks: Array<{ start: number; count: number }>;
} {
  const events: string[] = [];
  const masks: Array<{ start: number; count: number }> = [];
  return {
    events,
    masks,
    driver: {
      async prefill(input) {
        events.push(`prefill:${input.tokenIds.length}`);
      },
      async *generate(input) {
        masks.push(input.logitMask);
        for (const id of ids) {
          input.signal.throwIfAborted();
          events.push(`token:${id}`);
          yield id;
        }
      },
      async reset() {
        events.push("reset");
      },
      async dispose() {
        events.push("driver-dispose");
      },
    },
  };
}

test("loads only after origin lock acquisition and exposes a ready state", async () => {
  const events: string[] = [];
  const fake = driver();
  const manager = new ImmediateLockManager(events);
  const session = new Qwen35Session(runtime(fake.driver, events, manager));

  await session.load({});

  assert.equal(session.state, "ready");
  assert.deepEqual(events, ["lock", "load"]);
  await session.dispose();
});

test("a second loaded session does no work until the first cleanup finishes", async () => {
  const manager = new QueuedLockManager();
  const firstEvents: string[] = [];
  const secondEvents: string[] = [];
  const first = new Qwen35Session(runtime(driver().driver, firstEvents, manager));
  const second = new Qwen35Session(runtime(driver().driver, secondEvents, manager));
  await first.load({});

  const secondLoad = second.load({});
  await Promise.resolve();
  assert.deepEqual(secondEvents, []);

  await first.dispose();
  await secondLoad;
  assert.deepEqual(secondEvents, ["load"]);
  await second.dispose();
});

test("cancels a blocked load without starting model work", async () => {
  const manager = new QueuedLockManager();
  const first = new Qwen35Session(runtime(driver().driver, [], manager));
  const secondEvents: string[] = [];
  const second = new Qwen35Session(runtime(driver().driver, secondEvents, manager));
  await first.load({});

  const secondLoad = second.load({});
  await Promise.resolve();
  await second.cancel();

  await assert.rejects(secondLoad, { name: "AbortError" });
  assert.deepEqual(secondEvents, []);
  assert.equal(second.state, "failed");
  await first.dispose();
});

test("abort at loader resolution transfers resources into lock-held cleanup", async () => {
  const controller = new AbortController();
  let disposeCount = 0;
  const fake = driver();
  const abortingRuntime = runtime(fake.driver);
  abortingRuntime.load = async () => {
    controller.abort();
    return {
      tokenizer: tokenizer(),
      driver: fake.driver,
      cacheHit: false,
      trackedCpuBytes: 0,
      trackedGpuBytes: 0,
      async dispose() {
        disposeCount += 1;
      },
    };
  };
  const session = new Qwen35Session(abortingRuntime);

  await assert.rejects(session.load({ signal: controller.signal }), {
    name: "AbortError",
  });

  assert.equal(disposeCount, 1);
  assert.equal(session.state, "failed");
});

test("dispose during acquired load treats its AbortError as successful cleanup", async () => {
  const fake = driver();
  const loadingRuntime = runtime(fake.driver);
  loadingRuntime.load = async (signal) =>
    new Promise<Qwen35LoadedResources>((_resolve, reject) => {
      signal.addEventListener(
        "abort",
        () => reject(new DOMException("disposed", "AbortError")),
        { once: true },
      );
    });
  const session = new Qwen35Session(loadingRuntime);
  const load = session.load({}).then(
    () => null,
    (error: unknown) => error,
  );
  await Promise.resolve();

  await session.dispose();

  assert.equal((await load as Error).name, "AbortError");
  assert.equal(session.state, "disposed");
});

test("dispose succeeds after a failed load whose lock cleanup completed", async () => {
  const fake = driver();
  const failedRuntime = runtime(fake.driver);
  failedRuntime.load = async () => {
    throw new Error("private load failure");
  };
  const session = new Qwen35Session(failedRuntime);
  await assert.rejects(session.load({}), /private load failure/);

  await session.dispose();

  assert.equal(session.state, "disposed");
});

test("rejects illegal and reentrant calls with stable lifecycle diagnostics", async () => {
  const fake = driver();
  const session = new Qwen35Session(runtime(fake.driver));
  await assert.rejects(
    session.prefill([{ role: "user", content: "early" }]),
    (error: unknown) =>
      error instanceof RuntimeDiagnosticError &&
      error.code === "session-not-ready",
  );
  await session.load({});
  await assert.rejects(
    session.load({}),
    (error: unknown) =>
      error instanceof RuntimeDiagnosticError &&
      error.code === "session-load-illegal",
  );
  assert.equal(session.state, "ready");
  await session.dispose();
});

test("renders and prefills exact text while rejecting image input until vision loads", async () => {
  const fake = driver();
  const session = new Qwen35Session(runtime(fake.driver));
  await session.load({});

  const state = await session.prefill([{ role: "user", content: "hello" }]);
  assert.ok(state.contextTokens > 5);
  assert.match(state.rendered, /^<\|im_start\|>user\nhello/);
  await assert.rejects(
    session.prefill([
      {
        role: "user",
        content: [{ type: "image" }],
      },
    ]),
    (error: unknown) =>
      error instanceof RuntimeDiagnosticError &&
      error.code === "vision-not-loaded",
  );
  await session.dispose();
});

test("streams split UTF-8 and masks every tokenizer-unmapped logit row", async () => {
  const fake = driver();
  const session = new Qwen35Session(runtime(fake.driver));
  await session.load({});
  await session.prefill([{ role: "user", content: "hello" }]);

  const chunks = [];
  for await (const token of session.generate({ maxNewTokens: 4 })) {
    chunks.push(token.text);
  }

  assert.equal(chunks.join(""), "a€");
  assert.deepEqual(fake.masks, [{ start: 263, count: 248_320 - 263 }]);
  assert.equal(session.state, "ready");
  await session.dispose();
});

test("validates driver token ids before decode", async () => {
  const fake = driver([248_319]);
  const session = new Qwen35Session(runtime(fake.driver));
  await session.load({});
  await session.prefill([{ role: "user", content: "hello" }]);

  await assert.rejects(
    async () => {
      for await (const _token of session.generate({ maxNewTokens: 1 })) {
        // consume
      }
    },
    (error: unknown) =>
      error instanceof RuntimeDiagnosticError &&
      error.code === "driver-token-id-invalid",
  );
  assert.equal(session.state, "failed");
  await session.dispose();
});

test("fails closed when the generation driver rejects unexpectedly", async () => {
  let cancelCount = 0;
  const fake = driver();
  fake.driver.generate = async function* () {
    throw new Error("private driver detail");
  };
  fake.driver.cancel = async () => {
    cancelCount += 1;
  };
  const session = new Qwen35Session(runtime(fake.driver));
  await session.load({});
  await session.prefill([{ role: "user", content: "hello" }]);

  await assert.rejects(async () => {
    for await (const _token of session.generate({ maxNewTokens: 1 })) {
      // consume
    }
  }, /private driver detail/);

  assert.equal(session.state, "failed");
  assert.equal(cancelCount, 1);
  await session.dispose();
});

test("rejects generation options that exceed greedy or context bounds", async () => {
  const fake = driver();
  const session = new Qwen35Session(runtime(fake.driver));
  await session.load({});
  await session.prefill([{ role: "user", content: "hello" }]);

  assert.throws(
    () => session.generate({ maxNewTokens: 1, temperature: 1 as 0 }),
    { code: "sampling-not-supported" },
  );
  assert.throws(
    () => session.generate({ maxNewTokens: 16_384 }),
    { code: "context-limit-exceeded" },
  );
  await session.dispose();
});

test("generator early return cancels once and reset reuses loaded resources", async () => {
  let cancellationCount = 0;
  const fake = driver([97, 98, 99]);
  fake.driver.cancel = async () => {
    cancellationCount += 1;
  };
  const session = new Qwen35Session(runtime(fake.driver));
  await session.load({});
  await session.prefill([{ role: "user", content: "hello" }]);

  for await (const _token of session.generate({ maxNewTokens: 3 })) {
    break;
  }
  await session.cancel();
  await session.reset();

  assert.equal(cancellationCount, 1);
  assert.deepEqual(fake.events.filter((event) => event === "reset"), ["reset"]);
  assert.equal(session.getMetrics().contextTokens, 0);
  await session.dispose();
});

test("cancel aborts an active prefill and returns session ownership to ready", async () => {
  const fake = driver();
  fake.driver.prefill = async ({ signal }) =>
    new Promise<void>((_resolve, reject) => {
      signal.addEventListener(
        "abort",
        () => reject(new DOMException("cancelled", "AbortError")),
        { once: true },
      );
    });
  const session = new Qwen35Session(runtime(fake.driver));
  await session.load({});

  const prefill = session.prefill([{ role: "user", content: "hello" }]);
  await Promise.resolve();
  await session.cancel();

  await assert.rejects(prefill, { name: "AbortError" });
  assert.equal(session.state, "ready");
  assert.equal(session.getMetrics().cancellationCount, 1);
  assert.equal(fake.events.includes("reset"), true);
  await session.dispose();
});

test("cancellation idempotence resets for each new operation", async () => {
  let cancelCount = 0;
  const fake = driver([97, 98]);
  fake.driver.cancel = async () => {
    cancelCount += 1;
  };
  const session = new Qwen35Session(runtime(fake.driver));
  await session.load({});
  await session.prefill([{ role: "user", content: "first" }]);
  for await (const _token of session.generate({ maxNewTokens: 2 })) {
    break;
  }

  let releaseSecond!: () => void;
  fake.driver.prefill = async ({ signal }) =>
    new Promise<void>((resolve, reject) => {
      releaseSecond = resolve;
      signal.addEventListener(
        "abort",
        () => reject(new DOMException("cancelled", "AbortError")),
        { once: true },
      );
    });
  const second = session.prefill([{ role: "user", content: "second" }]);
  await Promise.resolve();
  await session.cancel();
  releaseSecond();
  await second.catch(() => undefined);

  assert.equal(cancelCount, 2);
  await session.dispose();
});

test("a failed prefill rollback poisons the session instead of reporting ready", async () => {
  const fake = driver();
  fake.driver.prefill = async ({ signal }) =>
    new Promise<void>((_resolve, reject) => {
      signal.addEventListener(
        "abort",
        () => reject(new DOMException("cancelled", "AbortError")),
        { once: true },
      );
    });
  fake.driver.reset = async () => {
    throw new Error("private rollback failure");
  };
  const session = new Qwen35Session(runtime(fake.driver));
  await session.load({});

  const prefill = session.prefill([{ role: "user", content: "hello" }]);
  await Promise.resolve();
  await session.cancel();

  await assert.rejects(prefill, /prefill rollback/i);
  assert.equal(session.state, "failed");
  await session.dispose();
});

test("cancel stops an active generation iterator exactly once", async () => {
  let cancelCount = 0;
  const fake = driver();
  fake.driver.generate = async function* ({ signal }) {
    yield 97;
    signal.throwIfAborted();
    await new Promise<void>((_resolve, reject) => {
      signal.addEventListener(
        "abort",
        () => reject(new DOMException("cancelled", "AbortError")),
        { once: true },
      );
    });
  };
  fake.driver.cancel = async () => {
    cancelCount += 1;
  };
  const session = new Qwen35Session(runtime(fake.driver));
  await session.load({});
  await session.prefill([{ role: "user", content: "hello" }]);
  const iterator = session.generate({ maxNewTokens: 2 })[Symbol.asyncIterator]();
  await iterator.next();

  await session.cancel();
  await session.cancel();
  await assert.rejects(iterator.next(), { name: "AbortError" });

  assert.equal(cancelCount, 1);
  assert.equal(session.state, "ready");
  await session.dispose();
});

test("cancel settles a generation iterator that never started", async () => {
  let cancelCount = 0;
  const fake = driver([97]);
  fake.driver.cancel = async () => {
    cancelCount += 1;
  };
  const session = new Qwen35Session(runtime(fake.driver));
  await session.load({});
  await session.prefill([{ role: "user", content: "hello" }]);

  session.generate({ maxNewTokens: 1 });
  await session.cancel();

  assert.equal(session.state, "ready");
  assert.equal(cancelCount, 1);
  const output = [];
  for await (const token of session.generate({ maxNewTokens: 1 })) {
    output.push(token.text);
  }
  assert.equal(output.join(""), "a");
  await session.dispose();
});

test("dispose waits for resource cleanup before lock ownership returns", async () => {
  const events: string[] = [];
  const fake = driver();
  const session = new Qwen35Session(runtime(fake.driver, events));
  await session.load({});

  await session.dispose();

  assert.equal(session.state, "disposed");
  assert.deepEqual(events, ["load", "resources-dispose"]);
  assert.equal(fake.events.at(-1), "driver-dispose");
  await session.dispose();
});

test("cleanup failure rejects dispose and leaves ownership failed", async () => {
  const fake = driver();
  const broken = runtime(fake.driver);
  broken.load = async () => ({
    tokenizer: tokenizer(),
    driver: fake.driver,
    cacheHit: false,
    trackedCpuBytes: 0,
    trackedGpuBytes: 0,
    async dispose() {
      throw new Error("private model path");
    },
  });
  const session = new Qwen35Session(broken);
  await session.load({});

  await assert.rejects(session.dispose(), {
    code: "origin-model-lock-cleanup-failed",
  });
  assert.equal(session.state, "failed");
});

test("metrics contain structured values without prompt or output content", async () => {
  const fake = driver([115, 101, 99, 114, 101, 116]);
  const session = new Qwen35Session(runtime(fake.driver));
  await session.load({});
  await session.prefill([{ role: "user", content: "PRIVATE PROMPT" }]);
  for await (const _token of session.generate({ maxNewTokens: 6 })) {
    // consume
  }

  const metrics = session.getMetrics();
  assert.equal(metrics.cacheHit, true);
  assert.equal(metrics.trackedCpuBytes, 12);
  assert.equal(metrics.trackedGpuBytes, 34);
  assert.equal(JSON.stringify(metrics).includes("PRIVATE"), false);
  assert.equal(JSON.stringify(metrics).includes("secret"), false);
  await session.dispose();
});

test("device loss is counted and fails closed until disposal", async () => {
  let lose!: () => void;
  const lost = new Promise<void>((resolve) => {
    lose = resolve;
  });
  const fake = driver();
  const withLoss = runtime(fake.driver);
  withLoss.load = async () => ({
    tokenizer: tokenizer(),
    driver: fake.driver,
    cacheHit: false,
    trackedCpuBytes: 0,
    trackedGpuBytes: 0,
    deviceLost: lost,
    async dispose() {
      await fake.driver.dispose();
    },
  });
  const session = new Qwen35Session(withLoss);
  await session.load({});

  lose();
  await Promise.resolve();

  assert.equal(session.state, "failed");
  assert.equal(session.getMetrics().deviceLostCount, 1);
  await session.dispose();
});
