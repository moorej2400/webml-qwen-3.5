import assert from "node:assert/strict";
import test from "node:test";

import {
  OriginModelLock,
  OriginModelLockCleanupError,
  type ExclusiveLockManager,
  type ExclusiveLockRequestOptions,
} from "../src/origin-model-lock.js";

interface PendingRequest<T = unknown> {
  options: ExclusiveLockRequestOptions;
  callback: () => Promise<T> | T;
  resolve(value: T): void;
  reject(reason: unknown): void;
  abort?: () => void;
}

class FakeExclusiveLockManager implements ExclusiveLockManager {
  private active = false;
  private readonly pending: PendingRequest[] = [];

  request<T>(
    _name: string,
    options: ExclusiveLockRequestOptions,
    callback: () => Promise<T> | T,
  ): Promise<T> {
    return new Promise<T>((resolve, reject) => {
      const entry: PendingRequest<T> = {
        options,
        callback,
        resolve,
        reject,
      };
      const abort = (): void => {
        const index = this.pending.indexOf(entry as PendingRequest);
        if (index >= 0) {
          this.pending.splice(index, 1);
          reject(new DOMException("The lock request was aborted", "AbortError"));
        }
      };
      entry.abort = abort;
      options.signal.addEventListener("abort", abort, { once: true });
      this.pending.push(entry as PendingRequest);
      this.drain();
    });
  }

  private drain(): void {
    if (this.active) {
      return;
    }
    const entry = this.pending.shift();
    if (entry === undefined) {
      return;
    }
    entry.options.signal.removeEventListener("abort", entry.abort!);
    if (entry.options.signal.aborted) {
      entry.reject(new DOMException("The lock request was aborted", "AbortError"));
      this.drain();
      return;
    }
    this.active = true;
    Promise.resolve(entry.callback()).then(entry.resolve, entry.reject).finally(() => {
      this.active = false;
      this.drain();
    });
  }
}

function deferred(): {
  promise: Promise<void>;
  resolve(): void;
} {
  let resolve!: () => void;
  const promise = new Promise<void>((settle) => {
    resolve = settle;
  });
  return { promise, resolve };
}

test("a blocked tab performs no model work until the exclusive lock is acquired", async () => {
  const manager = new FakeExclusiveLockManager();
  const firstGate = deferred();
  const first = new OriginModelLock(manager);
  const second = new OriginModelLock(manager);
  let firstModelWork = 0;
  let secondModelWork = 0;

  const firstRun = first.runExclusive({
    run: async () => {
      firstModelWork += 1;
      await firstGate.promise;
      return "first";
    },
    cleanup: async () => {},
  });
  const secondRun = second.runExclusive({
    run: async () => {
      secondModelWork += 1;
      return "second";
    },
    cleanup: async () => {},
  });

  await Promise.resolve();
  assert.equal(firstModelWork, 1);
  assert.equal(secondModelWork, 0);
  assert.equal(second.state, "blocked");

  firstGate.resolve();
  assert.equal(await firstRun, "first");
  assert.equal(await secondRun, "second");
  assert.deepEqual(second.transitions, ["blocked", "acquired", "released"]);
});

test("ownership remains held until asynchronous cleanup completes", async () => {
  const manager = new FakeExclusiveLockManager();
  const cleanupGate = deferred();
  const first = new OriginModelLock(manager);
  const second = new OriginModelLock(manager);
  let secondModelWork = 0;

  const firstRun = first.runExclusive({
    run: async () => "first",
    cleanup: async () => cleanupGate.promise,
  });
  const secondRun = second.runExclusive({
    run: async () => {
      secondModelWork += 1;
      return "second";
    },
    cleanup: async () => {},
  });

  await Promise.resolve();
  await Promise.resolve();
  assert.equal(first.state, "acquired");
  assert.equal(secondModelWork, 0);

  cleanupGate.resolve();
  assert.equal(await firstRun, "first");
  assert.equal(await secondRun, "second");
  assert.equal(secondModelWork, 1);
});

test("cleanup failure is reported before ownership can transfer", async () => {
  const manager = new FakeExclusiveLockManager();
  const cleanupGate = deferred();
  const first = new OriginModelLock(manager);
  const second = new OriginModelLock(manager);
  let secondModelWork = 0;

  const firstRun = first.runExclusive({
    run: async () => "unused",
    cleanup: async () => {
      await cleanupGate.promise;
      throw new Error("private cleanup detail");
    },
  });
  const secondRun = second.runExclusive({
    run: async () => {
      secondModelWork += 1;
      return "second";
    },
    cleanup: async () => {},
  });

  await Promise.resolve();
  await Promise.resolve();
  assert.equal(secondModelWork, 0);
  cleanupGate.resolve();
  await assert.rejects(firstRun, OriginModelLockCleanupError);
  assert.equal(await secondRun, "second");
  assert.equal(secondModelWork, 1);
  assert.equal(first.state, "released");
});

test("cancellation aborts a waiting lock without model work", async () => {
  const manager = new FakeExclusiveLockManager();
  const firstGate = deferred();
  const first = new OriginModelLock(manager);
  const second = new OriginModelLock(manager);
  let secondModelWork = 0;

  const firstRun = first.runExclusive({
    run: async () => firstGate.promise,
    cleanup: async () => {},
  });
  const secondRun = second.runExclusive({
    run: async () => {
      secondModelWork += 1;
    },
    cleanup: async () => {},
  });
  second.cancel();

  await assert.rejects(secondRun, { name: "AbortError" });
  assert.equal(secondModelWork, 0);
  assert.equal(second.state, "aborted");
  assert.deepEqual(second.transitions, ["blocked", "aborted"]);

  firstGate.resolve();
  await firstRun;
});

test("work cancellation cannot abort cleanup or release ownership early", async () => {
  const manager = new FakeExclusiveLockManager();
  const cleanupGate = deferred();
  const cleanupStarted = deferred();
  const first = new OriginModelLock(manager);
  const second = new OriginModelLock(manager);
  let cleanupSignalWasAborted = true;
  let secondModelWork = 0;

  const firstRun = first.runExclusive({
    run: async (signal) =>
      new Promise<void>((_resolve, reject) => {
        signal.addEventListener(
          "abort",
          () => reject(new DOMException("cancelled", "AbortError")),
          { once: true },
        );
    }),
    cleanup: async (cleanupSignal) => {
      cleanupSignalWasAborted = cleanupSignal.aborted;
      cleanupStarted.resolve();
      await cleanupGate.promise;
    },
  });
  const secondRun = second.runExclusive({
    run: async () => {
      secondModelWork += 1;
    },
    cleanup: async () => {},
  });

  await Promise.resolve();
  first.cancel();
  await cleanupStarted.promise;
  assert.equal(cleanupSignalWasAborted, false);
  assert.equal(secondModelWork, 0);

  cleanupGate.resolve();
  await assert.rejects(firstRun, { name: "AbortError" });
  await secondRun;
  assert.equal(secondModelWork, 1);
});
