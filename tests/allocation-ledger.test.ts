import assert from "node:assert/strict";
import test from "node:test";

import { AllocationLedger } from "../src/allocation-ledger.js";

test("reserves and releases bytes atomically by category", () => {
  const ledger = new AllocationLedger(100n);

  ledger.reserve({ id: "weights", category: "model", bytes: 60n });
  ledger.reserve({ id: "work", category: "scratch", bytes: 20n });
  ledger.release("weights");

  assert.deepEqual(ledger.snapshot(), {
    limitBytes: 100n,
    currentBytes: 20n,
    peakBytes: 80n,
    currentByCategory: { scratch: 20n },
    allocationCount: 1,
  });
  ledger.release("work");
  assert.doesNotThrow(() => ledger.assertAllReleased());
});

test("rejects over-limit, duplicate, negative, and double-release operations", () => {
  const ledger = new AllocationLedger(64n);

  assert.throws(
    () => ledger.reserve({ id: "negative", category: "model", bytes: -1n }),
    /greater than zero/i,
  );
  ledger.reserve({ id: "one", category: "model", bytes: 48n });
  assert.throws(
    () => ledger.reserve({ id: "one", category: "upload", bytes: 1n }),
    /duplicate/i,
  );
  assert.throws(
    () => ledger.reserve({ id: "two", category: "upload", bytes: 17n }),
    /limit/i,
  );
  assert.equal(ledger.snapshot().currentBytes, 48n);
  ledger.release("one");
  assert.throws(() => ledger.release("one"), /not reserved|already released/i);
});

test("reports unreleased ownership", () => {
  const ledger = new AllocationLedger(64n);
  ledger.reserve({ id: "live", category: "activation", bytes: 8n });

  assert.throws(() => ledger.assertAllReleased(), /live.*8 bytes/i);
});
