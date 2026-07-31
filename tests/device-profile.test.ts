import assert from "node:assert/strict";
import test from "node:test";

import {
  DEFAULT_INITIAL_ARENA_BYTES,
  DEFAULT_UPLOAD_LANE_BYTES,
  probeDeviceProfile,
} from "../src/device-profile.js";

const MIB = 1024 * 1024;

function adapterSurface(options?: {
  readonly maxBufferSize?: number;
  readonly maxStorageBufferBindingSize?: number;
  readonly heapLimit?: number;
}) {
  const requests: unknown[] = [];
  const limits = {
    maxBufferSize: options?.maxBufferSize ?? 512 * MIB,
    maxStorageBufferBindingSize:
      options?.maxStorageBufferBindingSize ?? 384 * MIB,
  };
  const device = { limits, createBuffer: () => ({ destroy() {} }) };
  return {
    requests,
    surface: {
      gpu: {
        async requestAdapter() {
          return {
            features: new Set(["shader-f16", "timestamp-query"]),
            limits,
            async requestDevice(descriptor: unknown) {
              requests.push(descriptor);
              return device;
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
  assert.equal(profile.arenaCapBytes, DEFAULT_INITIAL_ARENA_BYTES);
  assert.equal(profile.uploadLaneBytes, DEFAULT_UPLOAD_LANE_BYTES);
  assert.equal(profile.facts.jsHeapLimitBytes, 768 * MIB);
  assert.deepEqual(profile.facts.features, [
    "shader-f16",
    "timestamp-query",
  ]);
  assert.deepEqual(profile.facts.limits, {
    maxBufferSize: 512 * MIB,
    maxStorageBufferBindingSize: 384 * MIB,
  });
  assert.deepEqual(Object.keys(profile.facts).sort(), [
    "features",
    "jsHeapLimitBytes",
    "limits",
  ]);
});

test("keeps missing Safari heap telemetry null and supports evidence profiles", async () => {
  const constrained = adapterSurface({
    maxBufferSize: 256 * MIB,
    maxStorageBufferBindingSize: 192 * MIB,
  });

  const profile128 = await probeDeviceProfile(constrained.surface, {
    arenaPolicy: "evidence-128",
  });
  const profile64 = await probeDeviceProfile(constrained.surface, {
    arenaPolicy: "evidence-64",
  });

  assert.equal(profile128.arenaCapBytes, 128 * MIB);
  assert.equal(profile64.arenaCapBytes, 64 * MIB);
  assert.equal(profile128.facts.jsHeapLimitBytes, null);
  assert.equal(profile128.policyIsCapabilityCeiling, false);
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
    /required WebGPU feature.*subgroups/i,
  );
  await assert.rejects(
    probeDeviceProfile(fake.surface, {
      requiredLimits: { maxBufferSize: 128 * MIB },
    }),
    /maxBufferSize/i,
  );
  await assert.rejects(
    probeDeviceProfile(fake.surface, {
      requiredLimits: { maxStorageBufferBindingSize: 64 * MIB },
    }),
    /maxStorageBufferBindingSize/i,
  );
});

test("rejects missing WebGPU adapters", async () => {
  await assert.rejects(probeDeviceProfile({}), /WebGPU is unavailable/i);
  await assert.rejects(
    probeDeviceProfile({ gpu: { async requestAdapter() { return null; } } }),
    /adapter is unavailable/i,
  );
});
