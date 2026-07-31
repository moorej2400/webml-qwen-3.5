import assert from "node:assert/strict";
import test from "node:test";

import {
  DEFAULT_BUFFER_SHARD_CAP_BYTES,
  DEFAULT_UPLOAD_LANE_BYTES,
  probeDeviceProfile,
} from "../src/device-profile.js";

const MIB = 1024 * 1024;

function adapterSurface(options?: {
  readonly maxBufferSize?: number;
  readonly maxStorageBufferBindingSize?: number;
  readonly minStorageBufferOffsetAlignment?: number;
  readonly deviceMinStorageBufferOffsetAlignment?: number;
  readonly deviceFeatures?: readonly string[];
  readonly heapLimit?: number;
}) {
  const requests: unknown[] = [];
  const limits = {
    maxBufferSize: options?.maxBufferSize ?? 512 * MIB,
    maxStorageBufferBindingSize:
      options?.maxStorageBufferBindingSize ?? 384 * MIB,
    minStorageBufferOffsetAlignment:
      options?.minStorageBufferOffsetAlignment ?? 256,
  };
  return {
    requests,
    surface: {
      gpu: {
        async requestAdapter() {
          return {
            features: new Set(["shader-f16", "timestamp-query"]),
            limits,
            async requestDevice(descriptor: {
              requiredFeatures?: readonly string[];
              requiredLimits?: Readonly<Record<string, number>>;
            }) {
              requests.push(descriptor);
              return {
                limits: {
                  ...limits,
                  ...descriptor.requiredLimits,
                  ...(options?.deviceMinStorageBufferOffsetAlignment ===
                  undefined
                    ? {}
                    : {
                        minStorageBufferOffsetAlignment:
                          options.deviceMinStorageBufferOffsetAlignment,
                      }),
                },
                features: new Set(
                  options?.deviceFeatures ??
                    descriptor.requiredFeatures ??
                    [],
                ),
              };
            },
          };
        },
      },
      ...(options?.heapLimit === undefined
        ? {}
        : { performance: { memory: { jsHeapSizeLimit: options.heapLimit } } }),
    },
  };
}

test("probes sanitized facts and requests only declared requirements", async () => {
  const fake = adapterSurface({ heapLimit: 768 * MIB });

  const profile = await probeDeviceProfile(fake.surface, {
    requiredFeatures: ["shader-f16"],
    requiredLimits: { maxStorageBufferBindingSize: 128 * MIB },
  });

  assert.deepEqual(fake.requests, [
    {
      requiredFeatures: ["shader-f16"],
      requiredLimits: { maxStorageBufferBindingSize: 128 * MIB },
    },
  ]);
  assert.equal(
    profile.bufferShardCapBytes,
    DEFAULT_BUFFER_SHARD_CAP_BYTES,
  );
  assert.equal(profile.uploadLaneBytes, DEFAULT_UPLOAD_LANE_BYTES);
  assert.equal(profile.facts.jsHeapLimitBytes, 768 * MIB);
  assert.deepEqual(profile.facts.adapter.features, [
    "shader-f16",
    "timestamp-query",
  ]);
  assert.deepEqual(profile.facts.device.features, ["shader-f16"]);
  assert.deepEqual(profile.facts.adapter.limits, {
    maxBufferSize: 512 * MIB,
    maxStorageBufferBindingSize: 384 * MIB,
  });
  assert.deepEqual(profile.facts.device.limits, {
    maxBufferSize: 512 * MIB,
    maxStorageBufferBindingSize: 128 * MIB,
  });
  assert.deepEqual(Object.keys(profile.facts).sort(), [
    "adapter",
    "device",
    "jsHeapLimitBytes",
  ]);
});

test("does not advertise an available but unrequested feature as enabled", async () => {
  const profile = await probeDeviceProfile(adapterSurface().surface);

  assert.deepEqual(profile.facts.adapter.features, [
    "shader-f16",
    "timestamp-query",
  ]);
  assert.deepEqual(profile.facts.device.features, []);
  assert.equal(profile.facts.device.features.includes("timestamp-query"), false);
});

test("reports the feature set returned by the device as enabled", async () => {
  const profile = await probeDeviceProfile(
    adapterSurface({
      deviceFeatures: ["shader-f16", "timestamp-query"],
    }).surface,
    { requiredFeatures: ["shader-f16"] },
  );

  assert.deepEqual(profile.facts.device.features, [
    "shader-f16",
    "timestamp-query",
  ]);
});

test("applies inverse comparison semantics to minimum alignment limits", async () => {
  const betterAdapter = adapterSurface({
    minStorageBufferOffsetAlignment: 128,
  });

  const accepted = await probeDeviceProfile(betterAdapter.surface, {
    requiredLimits: { minStorageBufferOffsetAlignment: 256 },
  });

  assert.equal(
    accepted.facts.device.limits.minStorageBufferOffsetAlignment,
    256,
  );
  await assert.rejects(
    probeDeviceProfile(
      adapterSurface({
        minStorageBufferOffsetAlignment: 256,
      }).surface,
      {
        requiredLimits: { minStorageBufferOffsetAlignment: 128 },
      },
    ),
    /required WebGPU limit is unavailable/i,
  );
  await assert.rejects(
    probeDeviceProfile(
      adapterSurface({
        minStorageBufferOffsetAlignment: 128,
        deviceMinStorageBufferOffsetAlignment: 512,
      }).surface,
      {
        requiredLimits: { minStorageBufferOffsetAlignment: 256 },
      },
    ),
    /returned WebGPU device does not satisfy required limits/i,
  );
});

test("keeps missing Safari heap telemetry null and supports evidence profiles", async () => {
  const constrained = adapterSurface({
    maxBufferSize: 256 * MIB,
    maxStorageBufferBindingSize: 192 * MIB,
  });

  const profile128 = await probeDeviceProfile(constrained.surface, {
    bufferShardPolicy: "evidence-128",
  });
  const profile64 = await probeDeviceProfile(constrained.surface, {
    bufferShardPolicy: "evidence-64",
  });

  assert.equal(profile128.bufferShardCapBytes, 128 * MIB);
  assert.equal(profile64.bufferShardCapBytes, 64 * MIB);
  assert.equal(profile128.facts.jsHeapLimitBytes, null);
  assert.equal(profile128.bufferShardCapIsCapabilityCeiling, false);
});

test("rejects unavailable requirements and either limiting buffer dimension", async () => {
  const fake = adapterSurface({
    maxBufferSize: 96 * MIB,
    maxStorageBufferBindingSize: 32 * MIB,
  });

  await assert.rejects(
    probeDeviceProfile(fake.surface, {
      requiredFeatures: ["subgroups"],
    }),
    /required WebGPU feature is unavailable/i,
  );
  await assert.rejects(
    probeDeviceProfile(fake.surface, {
      requiredLimits: { maxBufferSize: 128 * MIB },
    }),
    /required WebGPU limit is unavailable/i,
  );
  await assert.rejects(
    probeDeviceProfile(fake.surface, {
      requiredLimits: { maxStorageBufferBindingSize: 64 * MIB },
    }),
    /required WebGPU limit is unavailable/i,
  );
});

test("does not echo caller-provided feature identifiers in diagnostics", async () => {
  await assert.rejects(
    probeDeviceProfile(adapterSurface().surface, {
      requiredFeatures: ["https://example.invalid/<feature-id>"],
    }),
    (error: unknown) => {
      assert.ok(error instanceof Error);
      assert.match(error.message, /required WebGPU feature is unavailable/i);
      assert.doesNotMatch(error.message, /example\.invalid|feature-id|https/i);
      return true;
    },
  );
});

test("rejects missing WebGPU adapters", async () => {
  await assert.rejects(probeDeviceProfile({}), /WebGPU is unavailable/i);
  await assert.rejects(
    probeDeviceProfile({ gpu: { async requestAdapter() { return null; } } }),
    /adapter is unavailable/i,
  );
});
