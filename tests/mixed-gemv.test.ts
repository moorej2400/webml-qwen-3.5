import assert from "node:assert/strict";
import test from "node:test";

import { GgmlType } from "../src/gguf.js";
import {
  LANGUAGE_GEMV_KERNELS,
  gemvCpu,
  languageGemvRegistryDefinitions,
  planGemvDispatch,
  planMatrixShardDispatch,
  type GemvLayout,
} from "../src/mixed-gemv.js";
import { KernelRegistry } from "../src/kernel-registry.js";
import {
  dequantizeQ4KBlock,
  dequantizeQ5KBlock,
  dequantizeQ6KBlock,
  dequantizeQ8_0Block,
  repackNativeQ4K,
  repackNativeQ5K,
  repackNativeQ6K,
  repackNativeQ8_0,
  unpackNativeQ4KBlock,
  unpackNativeQ5KBlock,
  unpackNativeQ6KBlock,
  unpackNativeQ8_0Block,
} from "../src/mixed-quant.js";
import {
  dequantizeQ3KBlock,
  repackNativeQ3K,
  unpackNativeQ3KBlock,
} from "../src/q3k.js";

function native(length: number, multiplier: number): Uint8Array {
  return Uint8Array.from(
    { length },
    (_, index) => (index * multiplier + 23) & 255,
  );
}

function setHalf(bytes: Uint8Array, offset: number, bits = 0x3c00): void {
  new DataView(bytes.buffer, bytes.byteOffset).setUint16(offset, bits, true);
}

function dot(values: Float32Array, activation: Float32Array): number {
  let sum = 0;
  for (let index = 0; index < values.length; index += 1) {
    sum += values[index]! * activation[index]!;
  }
  return Math.fround(sum);
}

function packedFixture(layout: GemvLayout): {
  packed: Uint8Array;
  values: Float32Array;
} {
  switch (layout) {
    case "f32": {
      const values = Float32Array.of(-3, 0.5, 7, -2);
      return {
        packed: new Uint8Array(values.buffer.slice(0)),
        values,
      };
    }
    case "q8-0-36": {
      const bytes = native(34, 17);
      setHalf(bytes, 0, 0x3800);
      return {
        packed: repackNativeQ8_0(bytes),
        values: dequantizeQ8_0Block(unpackNativeQ8_0Block(bytes)),
      };
    }
    case "q3-k-112": {
      const bytes = native(110, 19);
      setHalf(bytes, 108);
      return {
        packed: repackNativeQ3K(bytes),
        values: dequantizeQ3KBlock(unpackNativeQ3KBlock(bytes)),
      };
    }
    case "q4-k-144": {
      const bytes = native(144, 23);
      setHalf(bytes, 0);
      setHalf(bytes, 2, 0x3800);
      return {
        packed: repackNativeQ4K(bytes),
        values: dequantizeQ4KBlock(unpackNativeQ4KBlock(bytes)),
      };
    }
    case "q5-k-176": {
      const bytes = native(176, 29);
      setHalf(bytes, 0);
      setHalf(bytes, 2, 0x3800);
      return {
        packed: repackNativeQ5K(bytes),
        values: dequantizeQ5KBlock(unpackNativeQ5KBlock(bytes)),
      };
    }
    case "q6-k-212": {
      const bytes = native(210, 31);
      setHalf(bytes, 208);
      return {
        packed: repackNativeQ6K(bytes),
        values: dequantizeQ6KBlock(unpackNativeQ6KBlock(bytes)),
      };
    }
  }
}

test("defines one explicit direct-read kernel ABI for every language layout", () => {
  assert.deepEqual(
    LANGUAGE_GEMV_KERNELS.map(({ ggmlType, layout, phase, profile }) => ({
      ggmlType,
      layout,
      phase,
      profile,
    })),
    [
      { ggmlType: GgmlType.F32, layout: "f32", phase: "shared", profile: "portable-f32" },
      { ggmlType: GgmlType.Q8_0, layout: "q8-0-36", phase: "shared", profile: "portable-f32" },
      { ggmlType: GgmlType.Q3_K, layout: "q3-k-112", phase: "shared", profile: "portable-f32" },
      { ggmlType: GgmlType.Q4_K, layout: "q4-k-144", phase: "shared", profile: "portable-f32" },
      { ggmlType: GgmlType.Q5_K, layout: "q5-k-176", phase: "shared", profile: "portable-f32" },
      { ggmlType: GgmlType.Q6_K, layout: "q6-k-212", phase: "shared", profile: "portable-f32" },
    ],
  );
  for (const definition of LANGUAGE_GEMV_KERNELS) {
    assert.equal(definition.abi.layout, definition.layout);
    assert.equal(definition.abi.phase, definition.phase);
    assert.equal(definition.abi.profile, definition.profile);
    assert.match(definition.source, /array<u32>/);
    assert.match(definition.source, /local_rows\s*:\s*u32/);
    assert.match(definition.source, /output_row_offset\s*:\s*u32/);
    assert.match(definition.source, /packed_weights/);
    assert.match(definition.source, /weight_value/);
    assert.doesNotMatch(definition.source, /array<f16>|mat(2|3|4)x/);
  }
});

test("matches packed CPU GEMV for every layout without a float weight matrix", () => {
  for (const kernel of LANGUAGE_GEMV_KERNELS) {
    const fixture = packedFixture(kernel.layout);
    const activation = Float32Array.from(
      { length: fixture.values.length },
      (_, index) => (index % 11) * 0.125 - 0.5,
    );
    const packed = new Uint8Array(fixture.packed.length * 2);
    packed.set(fixture.packed, 0);
    packed.set(fixture.packed, fixture.packed.length);
    assert.deepEqual(
      gemvCpu(kernel.layout, packed, activation, {
        rows: 2,
        columns: activation.length,
      }),
      Float32Array.of(
        dot(fixture.values, activation),
        dot(fixture.values, activation),
      ),
      kernel.layout,
    );
  }
});

test("plans safe row-aware offsets and complete matrix shard coverage", () => {
  for (const kernel of LANGUAGE_GEMV_KERNELS) {
    const columns = kernel.abi.valuesPerBlock * 2;
    const rowBytes = BigInt(kernel.abi.bytesPerBlock * 2);
    const dispatch = planGemvDispatch({
      layout: kernel.layout,
      localRows: 3,
      columns,
      packedByteOffset: 32,
      outputRowOffset: 7,
    });
    assert.deepEqual(dispatch.workgroups, { x: 3, y: 1, z: 1 });
    assert.equal(dispatch.uniforms.weightWordOffset, 8);
    assert.equal(dispatch.uniforms.outputRowOffset, 7);

    const shards = planMatrixShardDispatch({
      layout: kernel.layout,
      rows: 4,
      columns,
      shards: [
        { logicalByteOffset: 0n, logicalByteLength: rowBytes },
        { logicalByteOffset: rowBytes, logicalByteLength: rowBytes * 3n },
      ],
    });
    assert.deepEqual(shards.map((item) => item.uniforms.localRows), [1, 3]);
    assert.deepEqual(shards.map((item) => item.uniforms.outputRowOffset), [0, 1]);
  }
});

test("guards two-dimensional row flattening before u32 arithmetic can wrap", () => {
  const dispatch = planGemvDispatch({
    layout: "f32",
    localRows: 4_000_000_000,
    columns: 1,
    maxWorkgroupsPerDimension: 3_000_000_000,
  });
  assert.deepEqual(dispatch.workgroups, {
    x: 3_000_000_000,
    y: 2,
    z: 1,
  });

  for (const kernel of LANGUAGE_GEMV_KERNELS) {
    const guard = kernel.source.indexOf(
      "if (invocation.y > (0xffffffffu - invocation.x) / grid.x)",
    );
    const flatten = kernel.source.indexOf(
      "let row = invocation.y * grid.x + invocation.x",
    );
    assert.ok(guard >= 0, `${kernel.layout} is missing the overflow guard`);
    assert.ok(
      guard < flatten,
      `${kernel.layout} computes a wrapped row before the guard`,
    );
  }
});

test("reads matrices from a u32-aligned non-block-multiple shard offset", () => {
  for (const kernel of LANGUAGE_GEMV_KERNELS) {
    const fixture = packedFixture(kernel.layout);
    const activation = Float32Array.from(
      { length: fixture.values.length },
      (_, index) => (index % 5) - 2,
    );
    const prefixed = new Uint8Array(32 + fixture.packed.length);
    prefixed.set(fixture.packed, 32);

    assert.deepEqual(
      gemvCpu(kernel.layout, prefixed, activation, {
        rows: 1,
        columns: activation.length,
        packedByteOffset: 32,
      }),
      Float32Array.of(dot(fixture.values, activation)),
      kernel.layout,
    );
  }
});

test("rejects unsupported and mismatched type/layout or unsafe shapes", () => {
  assert.throws(
    () =>
      planGemvDispatch({
        layout: "q2-k-84" as GemvLayout,
        localRows: 1,
        columns: 256,
      }),
    /unsupported.*layout/i,
  );
  assert.throws(
    () =>
      planGemvDispatch({
        layout: "q6-k-212",
        ggmlType: GgmlType.Q5_K,
        localRows: 1,
        columns: 256,
      }),
    /type.*layout/i,
  );
  assert.throws(
    () =>
      planGemvDispatch({
        layout: "q8-0-36",
        localRows: 1,
        columns: 31,
      }),
    /multiple of 32/i,
  );
  assert.throws(
    () =>
      planGemvDispatch({
        layout: "q6-k-212",
        localRows: 1,
        columns: 256,
        packedByteOffset: 2,
      }),
    /u32 aligned/i,
  );
  assert.throws(
    () =>
      planMatrixShardDispatch({
        layout: "q4-k-144",
        rows: 2,
        columns: 256,
        shards: [{ logicalByteOffset: 0n, logicalByteLength: 144n }],
      }),
    /complete matrix/i,
  );
});

test("registers explicit phase and profile keys for all matrix layouts", () => {
  const definitions = languageGemvRegistryDefinitions({
    phase: "decode",
    profile: "portable-f32",
  });
  const registry = new KernelRegistry();
  for (const definition of definitions) registry.register(definition);

  for (const definition of definitions) {
    const selected = registry.select({
      key: definition.key,
      fallbackProfiles: [],
    });
    assert.equal(selected.kernel.id, definition.id);
    assert.equal(definition.abi.layout, definition.key.layout);
    assert.equal(definition.abi.phase, definition.key.phase);
    assert.equal(definition.abi.profile, definition.key.profile);
  }
  assert.throws(
    () =>
      languageGemvRegistryDefinitions({
        phase: "training" as "decode",
        profile: "portable-f32",
      }),
    /unsupported.*phase/i,
  );
  assert.throws(
    () =>
      languageGemvRegistryDefinitions({
        phase: "decode",
        profile: "unknown" as "portable-f32",
      }),
    /unsupported.*profile/i,
  );
});
