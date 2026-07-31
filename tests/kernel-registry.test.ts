import assert from "node:assert/strict";
import test from "node:test";

import {
  KernelRegistry,
  type KernelKey,
} from "../src/kernel-registry.js";

const exact: KernelKey = {
  operation: "q3k-gemv",
  layout: "q3k-aligned-112",
  phase: "prefill",
  profile: "apple-f16",
};
const fallback: KernelKey = { ...exact, profile: "portable-f32" };

test("selects exact kernels deterministically without a pivot record", () => {
  const registry = new KernelRegistry();
  registry.register({ id: "portable", key: fallback, source: "fallback" });
  registry.register({ id: "exact", key: exact, source: "exact" });

  const selected = registry.select({ key: exact, fallbackProfiles: [] });

  assert.equal(selected.kernel.id, "exact");
  assert.equal(selected.pivot, null);
  assert.deepEqual(selected.attemptedProfiles, ["apple-f16"]);
});

test("uses only explicit fallback order and records the pivot", () => {
  const registry = new KernelRegistry();
  registry.register({ id: "portable", key: fallback, source: "fallback" });

  const selected = registry.select({
    key: exact,
    fallbackProfiles: ["portable-f32"],
    pivotReason: "physical-device-evidence",
  });

  assert.equal(selected.kernel.id, "portable");
  assert.deepEqual(selected.pivot, {
    fromProfile: "apple-f16",
    toProfile: "portable-f32",
    reason: "physical-device-evidence",
  });
  assert.deepEqual(registry.pivotRecords(), [selected.pivot]);
  assert.throws(
    () => registry.select({ key: exact, fallbackProfiles: [] }),
    /no kernel.*apple-f16/i,
  );
});

test("records structured compilation success and errors without silent retry", async () => {
  let now = 10;
  const registry = new KernelRegistry(() => now);
  registry.register({ id: "exact", key: exact, source: "exact" });
  registry.register({ id: "portable", key: fallback, source: "fallback" });

  const compiled = await registry.compile(
    { key: exact, fallbackProfiles: ["portable-f32"] },
    async (kernel) => {
      now = 16;
      return `compiled:${kernel.id}`;
    },
  );
  assert.equal(compiled.value, "compiled:exact");

  now = 20;
  await assert.rejects(
    registry.compile(
      { key: exact, fallbackProfiles: ["portable-f32"] },
      async () => {
        now = 27;
        throw new Error("shader validation failed");
      },
    ),
    /shader validation failed/i,
  );

  assert.deepEqual(registry.compilationMetrics(), [
    {
      kernelId: "exact",
      key: exact,
      status: "success",
      durationMs: 6,
      pivot: null,
      error: null,
    },
    {
      kernelId: "exact",
      key: exact,
      status: "error",
      durationMs: 7,
      pivot: null,
      error: {
        name: "Error",
        message: "shader validation failed",
      },
    },
  ]);
});

test("rejects duplicate registration keys", () => {
  const registry = new KernelRegistry();
  registry.register({ id: "one", key: exact, source: "one" });
  assert.throws(
    () => registry.register({ id: "two", key: exact, source: "two" }),
    /duplicate kernel key/i,
  );
});
