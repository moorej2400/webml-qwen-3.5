import assert from "node:assert/strict";
import test from "node:test";

import { AllocationLedger } from "../src/allocation-ledger.js";

test("reserves and releases bytes atomically by category", () => {
  const ledger = new AllocationLedger(100n);

  const weights = ledger.reserve({
    id: "weights",
    category: "model",
    bytes: 60n,
  });
  const work = ledger.reserve({
    id: "work",
    category: "scratch",
    bytes: 20n,
  });
  ledger.release(weights);

  assert.deepEqual(ledger.snapshot(), {
    limitBytes: 100n,
    currentBytes: 20n,
    peakBytes: 80n,
    currentByCategory: { scratch: 20n },
    allocationCount: 1,
  });
  ledger.release(work);
  assert.doesNotThrow(() => ledger.assertAllReleased());
});

test("rejects over-limit, duplicate, negative, and double-release operations", () => {
  const ledger = new AllocationLedger(64n);

  assert.throws(
    () => ledger.reserve({ id: "negative", category: "model", bytes: -1n }),
    /greater than zero/i,
  );
  const one = ledger.reserve({ id: "one", category: "model", bytes: 48n });
  assert.throws(
    () => ledger.reserve({ id: "one", category: "upload", bytes: 1n }),
    /duplicate/i,
  );
  assert.throws(
    () => ledger.reserve({ id: "two", category: "upload", bytes: 17n }),
    /limit/i,
  );
  assert.equal(ledger.snapshot().currentBytes, 48n);
  ledger.release(one);
  assert.throws(() => ledger.release(one), /stale|already released/i);
});

test("allows a released stable id to be reused but rejects stale handles", () => {
  const ledger = new AllocationLedger(64n);
  const first = ledger.reserve({ id: "scratch", category: "scratch", bytes: 8n });
  ledger.release(first);

  const second = ledger.reserve({
    id: "scratch",
    category: "scratch",
    bytes: 16n,
  });

  assert.throws(() => ledger.release(first), /stale|already released/i);
  assert.equal(ledger.snapshot().currentBytes, 16n);
  ledger.release(second);
  ledger.assertAllReleased();
});

test("redacts allocation ids from duplicate and leak diagnostics", () => {
  const ledger = new AllocationLedger(64n);
  const privateId = "local-path:<path>/<tensor-id>.gguf";
  ledger.reserve({ id: privateId, category: "activation", bytes: 8n });

  assert.throws(
    () =>
      ledger.reserve({
        id: privateId,
        category: "activation",
        bytes: 8n,
      }),
    (error: unknown) => {
      assert.ok(error instanceof Error);
      assert.doesNotMatch(error.message, /local-path|tensor-id|gguf/i);
      return true;
    },
  );
  assert.throws(
    () => ledger.assertAllReleased(),
    (error: unknown) => {
      assert.ok(error instanceof Error);
      assert.match(error.message, /1.*activation/i);
      assert.doesNotMatch(error.message, /local-path|tensor-id|gguf/i);
      return true;
    },
  );
});
