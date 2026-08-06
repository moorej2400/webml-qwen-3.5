import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

async function loadKernelModule(): Promise<Record<string, unknown>> {
  return import("../src/hybrid-kernels.js").catch(() => ({}));
}

test("defines only the six model-specific hybrid decode kernels", async () => {
  const module = await loadKernelModule();
  assert.ok(Array.isArray(module.QWEN35_HYBRID_KERNELS));
  const kernels = module.QWEN35_HYBRID_KERNELS as readonly {
    readonly id: string;
    readonly key: {
      readonly operation: string;
      readonly layout: string;
      readonly phase: string;
      readonly profile: string;
    };
    readonly source: string;
  }[];

  assert.deepEqual(
    kernels.map(({ key }) => key.operation),
    [
      "full-attention-prepare",
      "full-attention-online",
      "deltanet-conv",
      "deltanet-parameters",
      "deltanet-recurrent",
      "deltanet-gated-norm",
    ],
  );
  assert.equal(kernels.every(({ key }) => key.phase === "decode"), true);
  assert.equal(
    kernels.every(({ key }) => key.profile === "portable-f32"),
    true,
  );
  assert.equal(new Set(kernels.map(({ id }) => id)).size, kernels.length);
});

test("prepares interleaved Q/gate and writes the current packed K/V row", async () => {
  const module = await loadKernelModule();
  const kernels = module.QWEN35_HYBRID_KERNELS as readonly {
    readonly key: { readonly operation: string };
    readonly source: string;
  }[];
  const source = kernels.find(
    ({ key }) => key.operation === "full-attention-prepare",
  )!.source;

  assert.match(source, /query_head\s*\*\s*512u/);
  assert.match(source, /query_base\s*\+\s*256u\s*\+\s*lane/);
  assert.match(source, /prepared_query_gate/);
  assert.match(source, /frequency_owner/);
  assert.match(source, /10000000\.0f/);
  assert.match(source, /pack2x16float/);
  assert.match(source, /params\.position\s*\*\s*512u/);
  assert.match(source, /params\.position\s*>=\s*params\.capacity/);
  assert.match(source, /params\.capacity\s*>\s*16384u/);
  assert.equal(
    (source.match(/var<storage/g) ?? []).length <= 8,
    true,
  );
});

test("streams packed FP16 K/V without a full attention score matrix", async () => {
  const module = await loadKernelModule();
  const kernels = module.QWEN35_HYBRID_KERNELS as readonly {
    readonly key: { readonly operation: string };
    readonly source: string;
  }[];
  const source = kernels.find(
    ({ key }) => key.operation === "full-attention-online",
  )!.source;

  assert.match(source, /unpack2x16float/);
  assert.match(source, /query_head\s*\/\s*4u/);
  assert.match(source, /query_head\s*\*\s*512u/);
  assert.match(source, /0\.0625f/);
  assert.match(source, /running_maximum/);
  assert.match(source, /running_denominator/);
  assert.match(source, /var<storage, read_write> online_state/);
  assert.match(source, /params\.page_index\s*==\s*0u/);
  assert.match(source, /params\.page_index\s*\+\s*1u\s*==\s*params\.page_count/);
  assert.match(source, /online_state\[state_base\]\s*=\s*running_maximum/);
  assert.match(source, /online_state\[state_base \+ 1u\]\s*=\s*running_denominator/);
  assert.match(source, /select\(\s*accumulator\[lane\],[\s\S]*final_page/);
  assert.match(source, /for\s*\(var token/);
  assert.doesNotMatch(source, /array\s*<\s*f32\s*,\s*16384/);
  assert.doesNotMatch(source, /\bscores?\b/i);
});

test("encodes exact DeltaNet state order and head mapping in WGSL", async () => {
  const module = await loadKernelModule();
  const kernels = module.QWEN35_HYBRID_KERNELS as readonly {
    readonly key: { readonly operation: string };
    readonly source: string;
  }[];
  const byOperation = new Map(
    kernels.map(({ key, source }) => [key.operation, source]),
  );

  const conv = byOperation.get("deltanet-conv")!;
  assert.match(conv, /state_values\[base\]\s*=\s*state_values\[base \+ 1u\]/);
  assert.match(conv, /state_values\[base \+ 3u\]\s*=\s*raw_qkv\[channel\]/);
  assert.match(conv, /weight_values\[base \+ tap\]/);

  const parameters = byOperation.get("deltanet-parameters")!;
  assert.match(parameters, /beta_values\[head\]\s*=\s*sigmoid/);
  assert.match(parameters, /ssm_a_values\[head\]\s*\*\s*softplus/);
  assert.match(parameters, /decay_values\[head\]\s*=\s*exp/);

  const recurrent = byOperation.get("deltanet-recurrent")!;
  assert.match(recurrent, /value_head\s*%\s*16u/);
  assert.match(recurrent, /state_values\[state_index\]\s*=\s*decayed/);
  assert.match(
    recurrent,
    /beta_values\[value_head\]\s*\*\s*\(target_value - memory\)/,
  );
  assert.match(recurrent, /state_values\[state_index\]\s*=\s*updated/);
  assert.match(recurrent, /query_value\s*\*\s*updated/);
  assert.match(recurrent, /0\.000001f/);
  assert.match(recurrent, /inverseSqrt\(128\.0f\)/);
  assert.doesNotMatch(recurrent, /\blet\s+target\b/);

  const norm = byOperation.get("deltanet-gated-norm")!;
  assert.match(norm, /norm_weight_values\[lane\]/);
  assert.match(norm, /silu\(z_values\[index\]\)/);
});

test("registers every hybrid kernel through the existing explicit registry", async () => {
  const module = await loadKernelModule();
  assert.equal(typeof module.registerQwen35HybridKernels, "function");
  const registered: unknown[] = [];
  (
    module.registerQwen35HybridKernels as (registry: {
      register(definition: unknown): void;
    }) => void
  )({
    register(definition) {
      registered.push(definition);
    },
  });

  assert.equal(registered.length, 6);
  assert.deepEqual(registered, module.QWEN35_HYBRID_KERNELS);
});

test("plans the final legal packed K/V write without crossing capacity", async () => {
  const module = await loadKernelModule();
  assert.equal(typeof module.planQwen35FullAttentionKvWrite, "function");
  const plan = module.planQwen35FullAttentionKvWrite as (
    position: number,
    capacity: number,
  ) => {
    readonly wordOffset: number;
    readonly wordLength: number;
    readonly endWord: number;
  };

  assert.deepEqual(plan(16_383, 16_384), {
    wordOffset: 8_388_096,
    wordLength: 512,
    endWord: 8_388_608,
  });
  assert.throws(() => plan(16_384, 16_384), /position/i);
  assert.throws(() => plan(0, 16_385), /capacity/i);
});

test("bounds every hybrid dispatch and online K/V read", async () => {
  const module = await loadKernelModule();
  assert.equal(typeof module.planQwen35HybridDispatch, "function");
  const plan = module.planQwen35HybridDispatch as (input: {
    readonly operation: string;
    readonly maxComputeWorkgroupsPerDimension: number;
    readonly position?: number;
    readonly capacity?: number;
    readonly tokenCount?: number;
    readonly positions?: readonly [number, number, number];
  }) => {
    readonly workgroups: { readonly x: number; readonly y: 1; readonly z: 1 };
  };

  const common = { maxComputeWorkgroupsPerDimension: 65_535 };
  assert.deepEqual(
    [
      plan({
        operation: "full-attention-prepare",
        ...common,
        position: 16_383,
        capacity: 16_384,
        positions: [16_383, 16_383, 16_383],
      }).workgroups.x,
      plan({
        operation: "full-attention-online",
        ...common,
        tokenCount: 16_384,
        position: 16_383,
        capacity: 16_384,
      }).workgroups.x,
      plan({ operation: "deltanet-conv", ...common }).workgroups.x,
      plan({ operation: "deltanet-parameters", ...common }).workgroups.x,
      plan({ operation: "deltanet-recurrent", ...common }).workgroups.x,
      plan({ operation: "deltanet-gated-norm", ...common }).workgroups.x,
    ],
    [16, 16, 128, 1, 32, 32],
  );

  for (const input of [
    { tokenCount: 0, position: 0, capacity: 1 },
    { tokenCount: 2, position: 0, capacity: 2 },
    { tokenCount: 1, position: 1, capacity: 1 },
    { tokenCount: 1, position: 0, capacity: 16_385 },
  ]) {
    assert.throws(
      () =>
        plan({
          operation: "full-attention-online",
          ...common,
          ...input,
        }),
      /token|position|capacity/i,
    );
  }
  assert.throws(
    () =>
      plan({
        operation: "full-attention-prepare",
        maxComputeWorkgroupsPerDimension: 15,
        position: 0,
        capacity: 1,
        positions: [0, 0, 0],
      }),
    /workgroup/i,
  );
});

test("ships a deterministic Chrome compile harness for all hybrid kernels", async () => {
  const html = await readFile(
    new URL("../tools/webgpu-kernel-harness.html", import.meta.url),
    "utf8",
  ).catch(() => "");
  const script = await readFile(
    new URL("../tools/webgpu-kernel-harness.mjs", import.meta.url),
    "utf8",
  ).catch(() => "");

  assert.match(html, /webgpu-kernel-harness\.mjs/);
  assert.match(script, /QWEN35_HYBRID_KERNELS/);
  assert.match(script, /getCompilationInfo/);
  assert.match(script, /createComputePipelineAsync/);
  assert.match(script, /for \(const kernel of QWEN35_HYBRID_KERNELS\)/);
  assert.match(script, /runHybridKernel/);
  assert.match(script, /runFullAttentionOnline/);
  assert.match(script, /runDeltaNetConv/);
  assert.match(script, /runDeltaNetParameters/);
  assert.match(script, /runDeltaNetRecurrent/);
  assert.match(script, /runDeltaNetGatedNorm/);
  assert.match(script, /stateMutation/);
  assert.match(script, /position\s*=\s*16_383/);
  assert.match(script, /suffixSentinel/);
  assert.match(script, /packFloat16PairCpu/);
  assert.doesNotMatch(script, /Math\.random/);
});
