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
    /no kernel.*explicit profiles/i,
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
        throw new Error(
          "shader validation failed at local-path:<path>/<model-id>.wgsl for https://example.invalid/<path>",
        );
      },
    ),
    (error: unknown) => {
      assert.ok(error instanceof Error);
      assert.match(error.message, /kernel compilation failed/i);
      assert.doesNotMatch(
        error.message,
        /local-path|model-id|example\.invalid|wgsl/i,
      );
      assert.equal(
        (error as Error & { code?: string }).code,
        "KERNEL_COMPILE_FAILED",
      );
      return true;
    },
  );

  const metrics = registry.compilationMetrics();
  assert.deepEqual(metrics, [
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
        code: "KERNEL_COMPILE_FAILED",
        message: "Kernel compilation failed",
      },
    },
  ]);
  assert.equal(Object.isFrozen(metrics), true);
  assert.equal(Object.isFrozen(metrics[0]), true);
});

test("rejects duplicate registration keys", () => {
  const registry = new KernelRegistry();
  registry.register({ id: "one", key: exact, source: "one" });
  assert.throws(
    () => registry.register({ id: "two", key: exact, source: "two" }),
    /duplicate kernel key/i,
  );
});

test("redacts unsafe kernel identifiers from registration diagnostics", () => {
  const registry = new KernelRegistry();
  const privateId = "local-path:<path>/<kernel-id>.wgsl";

  assert.throws(
    () => registry.register({ id: privateId, key: exact, source: "source" }),
    (error: unknown) => {
      assert.ok(error instanceof Error);
      assert.doesNotMatch(error.message, /local-path|kernel-id|wgsl/i);
      return true;
    },
  );
});

test("freezes stored definitions and all selection boundary objects", () => {
  const registry = new KernelRegistry();
  const input = {
    id: "exact",
    key: { ...exact },
    source: "source",
  };
  registry.register(input);
  input.id = "mutated";
  input.key.profile = "mutated";

  const selection = registry.select({ key: exact, fallbackProfiles: [] });

  assert.equal(Object.isFrozen(selection), true);
  assert.equal(Object.isFrozen(selection.kernel), true);
  assert.equal(Object.isFrozen(selection.kernel.key), true);
  assert.equal(Object.isFrozen(selection.attemptedProfiles), true);
  assert.throws(() => {
    (selection.kernel as { id: string }).id = "corrupted";
  }, TypeError);
  assert.throws(() => {
    (selection.kernel.key as { profile: string }).profile = "corrupted";
  }, TypeError);
  assert.throws(() => {
    (selection.attemptedProfiles as string[]).push("corrupted");
  }, TypeError);
  assert.equal(
    registry.select({ key: exact, fallbackProfiles: [] }).kernel.id,
    "exact",
  );
});

test("returns frozen cloned pivot records", () => {
  const registry = new KernelRegistry();
  registry.register({ id: "portable", key: fallback, source: "fallback" });
  registry.select({
    key: exact,
    fallbackProfiles: ["portable-f32"],
    pivotReason: "physical-device-evidence",
  });

  const pivots = registry.pivotRecords();

  assert.equal(Object.isFrozen(pivots), true);
  assert.equal(Object.isFrozen(pivots[0]), true);
  assert.throws(() => {
    (pivots[0] as { reason: string }).reason = "corrupted";
  }, TypeError);
  assert.equal(
    registry.pivotRecords()[0]?.reason,
    "physical-device-evidence",
  );
});
