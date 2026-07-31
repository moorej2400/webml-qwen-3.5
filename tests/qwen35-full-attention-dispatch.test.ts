import assert from "node:assert/strict";
import test from "node:test";

import { GgmlType } from "../src/gguf.js";
import { QWEN35_4B_CONFIG } from "../src/qwen35-config.js";
import {
  planQwen35ActivationWorkspace,
  type Qwen35ActivationResourceKind,
  type Qwen35ActivationResourceView,
} from "../src/qwen35-activation-workspace.js";
import type {
  Qwen35ForwardBufferSlice,
  Qwen35ForwardDeviceLimits,
} from "../src/qwen35-forward-dispatch.js";
import type { Qwen35HybridLayerResources } from "../src/hybrid-state.js";
import {
  buildQwen35Program,
  type Qwen35Invocation,
  type Qwen35TensorDirectoryEntry,
} from "../src/qwen35-program.js";
import type {
  Qwen35TensorWeightView,
  Qwen35WeightDirectoryView,
} from "../src/qwen35-weight-directory.js";

const limits: Qwen35ForwardDeviceLimits = {
  minStorageBufferOffsetAlignment: 256,
  minUniformBufferOffsetAlignment: 256,
  maxStorageBufferBindingSize: 1 << 30,
  maxUniformBufferBindingSize: 65_536,
  maxComputeWorkgroupsPerDimension: 65_535,
};

const storageTypes = new Map([
  [GgmlType.F32, "f32"],
  [GgmlType.Q3_K, "q3-k-112"],
  [GgmlType.Q6_K, "q6-k-212"],
] as const);

function tensor(
  name: string,
  shape: readonly number[],
  ggmlType: number = GgmlType.Q3_K,
): Qwen35TensorDirectoryEntry {
  return { name, shape, ggmlType, storageType: storageTypes.get(ggmlType)! };
}

function programDirectory(): Qwen35TensorDirectoryEntry[] {
  const entries = [
    tensor("output_norm.weight", [2_560], GgmlType.F32),
    tensor("token_embd.weight", [2_560, 248_320], GgmlType.Q6_K),
  ];
  const mlp = [
    ["ffn_down.weight", [9_216, 2_560]],
    ["ffn_gate.weight", [2_560, 9_216]],
    ["ffn_up.weight", [2_560, 9_216]],
    ["post_attention_norm.weight", [2_560]],
  ] as const;
  for (let layer = 0; layer < 32; layer += 1) {
    const full = (layer + 1) % 4 === 0;
    const attention = full
      ? [
          ["attn_k.weight", [2_560, 1_024]],
          ["attn_k_norm.weight", [256]],
          ["attn_norm.weight", [2_560]],
          ["attn_output.weight", [4_096, 2_560]],
          ["attn_q.weight", [2_560, 8_192]],
          ["attn_q_norm.weight", [256]],
          ["attn_v.weight", [2_560, 1_024]],
        ] as const
      : [
          ["attn_gate.weight", [2_560, 4_096]],
          ["attn_norm.weight", [2_560]],
          ["attn_qkv.weight", [2_560, 8_192]],
          ["ssm_a", [32]],
          ["ssm_alpha.weight", [2_560, 32]],
          ["ssm_beta.weight", [2_560, 32]],
          ["ssm_conv1d.weight", [4, 8_192]],
          ["ssm_dt.bias", [32]],
          ["ssm_norm.weight", [128]],
          ["ssm_out.weight", [4_096, 2_560]],
        ] as const;
    for (const [suffix, shape] of [...attention, ...mlp]) {
      entries.push(tensor(
        `blk.${layer}.${suffix}`,
        shape,
        shape.length === 1 || suffix === "ssm_conv1d.weight"
          ? GgmlType.F32
          : GgmlType.Q3_K,
      ));
    }
  }
  return entries;
}

function weightView(
  entry: Qwen35TensorDirectoryEntry,
  buffer: object = {},
): Qwen35TensorWeightView {
  const layout = {
    f32: { values: 1, bytes: 4 },
    "q3-k-112": { values: 256, bytes: 112 },
    "q6-k-212": { values: 256, bytes: 212 },
  }[entry.storageType];
  if (layout === undefined) throw new Error("unsupported fixture layout");
  const rowBytes = (entry.shape[0]! / layout.values) * layout.bytes;
  const rowCount = entry.shape.slice(1).reduce((value, next) => value * next, 1);
  const byteLength = rowBytes * rowCount;
  return Object.freeze({
    ...entry,
    rowBytes,
    rowCount,
    logicalBytes: BigInt(byteLength),
    physicalRows: Object.freeze([Object.freeze({
      buffer,
      firstRow: 0,
      rowCount,
      tensorByteOffset: 0,
      bufferByteOffset: 0,
      byteLength,
    })]),
  });
}

function weightDirectoryFromViews(
  tensors: readonly Qwen35TensorWeightView[],
): Qwen35WeightDirectoryView {
  const byName = new Map(tensors.map((item) => [item.name, item] as const));
  const bytes = tensors.reduce((sum, item) => sum + item.logicalBytes, 0n);
  return Object.freeze({
    size: tensors.length,
    tensors: Object.freeze(tensors),
    logicalBytes: bytes,
    allocatedBytes: bytes,
    get: (name: string) => byName.get(name),
    entries: () => byName.entries(),
    [Symbol.iterator]: () => byName[Symbol.iterator](),
  });
}

function workspace(overrides: ReadonlyMap<string, object> = new Map()) {
  const views = new Map<Qwen35ActivationResourceKind, Qwen35ActivationResourceView>();
  for (const resource of planQwen35ActivationWorkspace().resources) {
    const byteLength = Number(resource.bytes);
    views.set(resource.kind, Object.freeze({
      ...resource,
      byteLength,
      binding: Object.freeze({
        buffer: overrides.get(resource.kind) ?? {},
        offset: 0 as const,
        size: byteLength,
      }),
    }));
  }
  return Object.freeze({
    get(kind: Qwen35ActivationResourceKind) {
      const view = views.get(kind);
      if (view === undefined) throw new Error("missing fixture activation");
      return view;
    },
  });
}

function state(
  capacity = 8,
  pages = 1,
  sharedBuffer?: object,
): Qwen35HybridLayerResources {
  const bytes = BigInt(capacity * 2_048);
  const pageBytes = bytes / BigInt(pages);
  const resource = (kind: "key" | "value") => Object.freeze({
    id: `hybrid-state-layer-3-${kind}`,
    layer: 3,
    kind,
    bytes,
    byteLength: bytes,
    shards: Object.freeze(Array.from({ length: pages }, (_, index) => Object.freeze({
      buffer: sharedBuffer ?? {},
      logicalByteOffset: BigInt(index) * pageBytes,
      logicalByteLength: pageBytes,
      allocatedByteLength: pageBytes,
    }))),
  });
  return Object.freeze({ layer: 3, kind: "full-attention", key: resource("key"), value: resource("value") });
}

function uniforms(count = 14, buffer: object = {}): readonly Qwen35ForwardBufferSlice[] {
  return Object.freeze(Array.from({ length: count }, (_, index) => Object.freeze({
    buffer,
    offset: index * 256,
    byteLength: 32,
  })));
}

function runnableProgram(program: ReturnType<typeof buildQwen35Program>) {
  const { blockedBy: _blockedBy, ...runtimeProgram } =
    program as unknown as Record<string, unknown>;
  return Object.freeze({ ...runtimeProgram, runnable: true }) as unknown as typeof program;
}

function fixture() {
  const tensors = programDirectory();
  const program = runnableProgram(buildQwen35Program({ config: QWEN35_4B_CONFIG, tensors }));
  const invocation = program.invocations.find(
    (item): item is Extract<Qwen35Invocation, { kind: "full-attention" }> =>
      item.kind === "full-attention" && item.layer === 3,
  )!;
  const layerWeights = tensors
    .filter(({ name }) => name.startsWith("blk.3."))
    .map((entry) => weightView(entry));
  return {
    program,
    invocation,
    weights: weightDirectoryFromViews(layerWeights),
    workspace: workspace(),
    state: state(),
    position: 5,
    capacity: 8,
    mropePositions: [5, 2, 3] as const,
    limits,
  };
}

type AssemblyFixture = ReturnType<typeof fixture> & {
  readonly uniforms: readonly Qwen35ForwardBufferSlice[];
};

async function geometryPlanner(): Promise<(input: ReturnType<typeof fixture>) => {
  readonly layer: number;
  readonly fixedUniformCount: 7;
  readonly physicalGemvPieceCount: number;
  readonly uniformCount: number;
}> {
  const module = await import("../src/qwen35-full-attention-dispatch.js").catch(() => ({}));
  assert.equal(typeof module.planQwen35FullAttentionLayerGeometry, "function");
  return module.planQwen35FullAttentionLayerGeometry as never;
}

async function executableFixture(
  base: ReturnType<typeof fixture> = fixture(),
): Promise<AssemblyFixture> {
  const geometry = (await geometryPlanner())(base);
  return { ...base, uniforms: uniforms(geometry.uniformCount) };
}

async function planner(): Promise<(input: AssemblyFixture) => {
  readonly layer: number;
  readonly position: number;
  readonly uniformCount: number;
  readonly commands: readonly {
    readonly stage: string;
    readonly kernel: { readonly id: string; readonly source: string };
    readonly bindings: readonly { readonly buffer: object; readonly size: number }[];
    readonly uniformWords: readonly number[];
    readonly mutatesPersistentState: boolean;
  }[];
  readonly stateSemantics: {
    readonly failureAfterSubmission: string;
    readonly advancePositionAfter: string;
  };
}> {
  const module = await import("../src/qwen35-full-attention-dispatch.js").catch(() => ({}));
  assert.equal(typeof module.planQwen35FullAttentionLayerDispatch, "function");
  return module.planQwen35FullAttentionLayerDispatch as never;
}

test("derives frozen exact full-attention uniform geometry before allocation", async () => {
  const base = fixture();
  const geometry = (await geometryPlanner())(base);

  assert.deepEqual(geometry, {
    layer: 3,
    fixedUniformCount: 7,
    physicalGemvPieceCount: 7,
    uniformCount: 14,
  });
  assert.equal(Object.isFrozen(geometry), true);
});

test("assembles exact one-token full-attention stages and state bindings", async () => {
  const input = await executableFixture();
  const plan = (await planner())(input);

  assert.equal(plan.layer, 3);
  assert.equal(plan.position, 5);
  assert.equal(plan.uniformCount, 14);
  assert.deepEqual(plan.commands.map(({ stage }) => stage), [
    "input-rms",
    "query-projection",
    "key-projection",
    "value-projection",
    "full-attention-prepare",
    "full-attention-online",
    "attention-output-projection",
    "attention-residual",
    "post-attention-rms",
    "ffn-gate-projection",
    "ffn-up-projection",
    "swiglu",
    "ffn-down-projection",
    "mlp-residual",
  ]);
  const prepare = plan.commands[4]!;
  const online = plan.commands[5]!;
  assert.match(prepare.kernel.source, /pack2x16float/);
  assert.match(online.kernel.source, /running_maximum/);
  assert.deepEqual(prepare.uniformWords, [5, 8, 5, 2, 3, 0, 0, 0]);
  assert.deepEqual(online.uniformWords, [6, 5, 8, 0]);
  assert.equal(prepare.bindings[6]!.buffer,
    input.state.kind === "full-attention" ? input.state.key.shards[0]!.buffer : null);
  assert.equal(online.bindings[2]!.buffer,
    input.state.kind === "full-attention" ? input.state.value.shards[0]!.buffer : null);
  assert.deepEqual(
    plan.commands.filter(({ mutatesPersistentState }) => mutatesPersistentState).map(({ stage }) => stage),
    ["full-attention-prepare"],
  );
  assert.equal(plan.stateSemantics.failureAfterSubmission,
    "persistent-state-indeterminate-dispose-required");
  assert.equal(plan.stateSemantics.advancePositionAfter, "successful-queue-retirement");
});

test("returns deeply frozen real dispatch requests", async () => {
  const input = await executableFixture();
  const plan = (await planner())(input);
  assert.equal(Object.isFrozen(plan), true);
  assert.equal(Object.isFrozen(plan.commands), true);
  assert.ok(plan.commands.every((command) =>
    Object.isFrozen(command) && Object.isFrozen(command.kernel) &&
    Object.isFrozen(command.bindings) && command.bindings.every(Object.isFrozen) &&
    Object.isFrozen(command.uniformWords)));
  assert.deepEqual(plan.commands[0]!.uniformWords.slice(0, 2), [2_560, 2_560]);
  assert.equal(plan.commands.at(-1)!.bindings[2]!.buffer,
    input.workspace.get("packed-embedding-output").binding.buffer);
});

test("keeps physical GEMV fragments ordered and reports dynamic uniforms", async () => {
  const valid = fixture();
  const target = valid.weights.get("blk.3.attn_q.weight")!;
  const view = target.physicalRows[0]!;
  const firstRows = target.rowCount / 2;
  const firstBytes = firstRows * target.rowBytes;
  const fragmented = valid.weights.tensors.map((item) => item.name === target.name
    ? Object.freeze({
        ...item,
        physicalRows: Object.freeze([
          Object.freeze({ ...view, rowCount: firstRows, byteLength: firstBytes }),
          Object.freeze({
            ...view,
            firstRow: firstRows,
            rowCount: target.rowCount - firstRows,
            tensorByteOffset: firstBytes,
            bufferByteOffset: firstBytes,
            byteLength: view.byteLength - firstBytes,
          }),
        ]),
      })
    : item);
  const splitBase = {
    ...valid,
    weights: weightDirectoryFromViews(fragmented),
  };
  const geometry = (await geometryPlanner())(splitBase);
  assert.deepEqual(geometry, {
    layer: 3,
    fixedUniformCount: 7,
    physicalGemvPieceCount: 8,
    uniformCount: 15,
  });
  assert.equal(Object.isFrozen(geometry), true);
  const plan = (await planner())({ ...splitBase, uniforms: uniforms(geometry.uniformCount) });
  assert.equal(plan.uniformCount, 15);
  assert.deepEqual(plan.commands.slice(0, 4).map(({ stage }) => stage), [
    "input-rms", "query-projection", "query-projection", "key-projection",
  ]);
});

test("fails specifically when current kernel ABI cannot address multiple K/V pages", async () => {
  const valid = await executableFixture();
  const plan = await planner();
  assert.throws(
    () => plan({ ...valid, state: state(8, 2) }),
    /multiple K\/V pages|page-aware kernel|state-pages-unsupported/i,
  );
  assert.throws(
    () => plan({
      ...valid,
      limits: { ...valid.limits, maxStorageBufferBindingSize: 8_192 },
    }),
    /page-aware kernel|larger state shard/i,
  );
});

test("rejects invalid context, state kind, invocation identity, and direct tensor type", async () => {
  const plan = await planner();
  const valid = await executableFixture();
  assert.throws(
    () => plan({
      ...valid,
      program: Object.freeze({
        ...valid.program,
        runnable: false,
      }) as unknown as typeof valid.program,
    }),
    { code: "full-attention-program-invalid" },
  );
  assert.throws(
    () => plan({
      ...valid,
      program: Object.freeze({
        ...valid.program,
        blockedBy: undefined,
      }) as unknown as typeof valid.program,
    }),
    { code: "full-attention-program-invalid" },
  );
  assert.throws(
    () => plan({
      ...valid,
      program: Object.freeze({
        ...valid.program,
        blockedBy: "weight-orchestration",
      }) as unknown as typeof valid.program,
    }),
    { code: "full-attention-program-invalid" },
  );
  assert.throws(() => plan({ ...valid, position: 8 }), /context|position/i);
  assert.throws(() => plan({ ...valid, mropePositions: [-1, 0, 0] }), /context|position/i);
  assert.throws(() => plan({ ...valid, invocation: Object.freeze({ ...valid.invocation, layer: 7 }) }), /program|invocation/i);

  if (valid.state.kind !== "full-attention") throw new Error("bad fixture state");
  const wrongState = Object.freeze({
    layer: 3,
    kind: "gated-deltanet" as const,
    conv: Object.freeze({ ...valid.state.key, kind: "conv" as const }),
    recurrent: Object.freeze({ ...valid.state.value, kind: "recurrent" as const }),
  });
  assert.throws(() => plan({ ...valid, state: wrongState }), /state|full-attention/i);

  const invalidWeights = valid.weights.tensors.map((item) =>
    item.name === "blk.3.attn_q_norm.weight"
      ? Object.freeze({ ...item, ggmlType: GgmlType.Q3_K, storageType: "q3-k-112" as const })
      : item);
  assert.throws(
    () => plan({ ...valid, weights: weightDirectoryFromViews(invalidWeights) }),
    /F32|direct|tensor/i,
  );
});

test("rejects incomplete or aliased uniforms and activation-state overlap", async () => {
  const plan = await planner();
  const valid = await executableFixture();
  assert.throws(() => plan({ ...valid, uniforms: uniforms(13) }), /uniform/i);
  const uniformBuffer = {};
  const overlappingUniforms = uniforms(14, uniformBuffer).map((slot, index) =>
    index === 1 ? Object.freeze({ ...slot, offset: 0 }) : slot);
  assert.throws(() => plan({ ...valid, uniforms: overlappingUniforms }), /uniform|overlap|alias/i);

  const shared = {};
  assert.throws(
    () => plan({
      ...valid,
      workspace: workspace(new Map([["attention-projection-primary", shared]])),
      state: state(8, 1, shared),
    }),
    /buffer|overlap|alias/i,
  );
});

test("rejects non-F32 activation views and overlapping model-owned ranges", async () => {
  const plan = await planner();
  const valid = await executableFixture();
  const invalidWorkspace = Object.freeze({
    get(kind: Qwen35ActivationResourceKind) {
      const view = valid.workspace.get(kind);
      return kind === "full-attention-key"
        ? Object.freeze({ ...view, scalarType: "u32" as const })
        : view;
    },
  });
  assert.throws(() => plan({ ...valid, workspace: invalidWorkspace }), /activation|workspace|F32/i);

  const shared = {};
  const overlappingWeights = valid.weights.tensors.map((item) =>
    item.name === "blk.3.attn_q.weight" || item.name === "blk.3.attn_k.weight"
      ? Object.freeze({
          ...item,
          physicalRows: Object.freeze(item.physicalRows.map((view) =>
            Object.freeze({ ...view, buffer: shared, bufferByteOffset: 0 }))),
        })
      : item);
  assert.throws(
    () => plan({ ...valid, weights: weightDirectoryFromViews(overlappingWeights) }),
    /weight|overlap|alias/i,
  );
});

test("sanitizes caller weight and workspace lookup failures", async () => {
  const plan = await planner();
  const valid = await executableFixture();
  const privateDetail = "private-full-attention-model-path";
  assert.throws(
    () => plan({
      ...valid,
      weights: Object.freeze({
        ...valid.weights,
        get() {
          throw new Error(privateDetail);
        },
      }),
    }),
    (error: unknown) => {
      assert.equal((error as { code?: unknown }).code, "full-attention-weight-invalid");
      assert.doesNotMatch((error as Error).message, new RegExp(privateDetail, "i"));
      return true;
    },
  );
  assert.throws(
    () => plan({
      ...valid,
      workspace: Object.freeze({
        get() {
          throw new Error(privateDetail);
        },
      }),
    }),
    (error: unknown) => {
      assert.equal((error as { code?: unknown }).code, "full-attention-workspace-invalid");
      assert.doesNotMatch((error as Error).message, new RegExp(privateDetail, "i"));
      return true;
    },
  );
});
