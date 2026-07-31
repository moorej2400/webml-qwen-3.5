import assert from "node:assert/strict";
import test from "node:test";

import { GgmlType } from "../src/gguf.js";
import { QWEN35_4B_CONFIG } from "../src/qwen35-config.js";
import type {
  Qwen35ForwardBufferSlice,
  Qwen35ForwardDeviceLimits,
} from "../src/qwen35-forward-dispatch.js";
import type { Qwen35HybridLayerResources } from "../src/hybrid-state.js";
import {
  buildQwen35Program,
  type Qwen35Invocation,
  type Qwen35Program,
  type Qwen35TensorDirectoryEntry,
} from "../src/qwen35-program.js";
import {
  planQwen35ActivationWorkspace,
  type Qwen35ActivationResourceKind,
  type Qwen35ActivationResourceView,
} from "../src/qwen35-activation-workspace.js";
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

function programTensor(
  name: string,
  shape: readonly number[],
  ggmlType: number = GgmlType.Q3_K,
): Qwen35TensorDirectoryEntry {
  return { name, shape, ggmlType, storageType: storageTypes.get(ggmlType)! };
}

function programDirectory(): Qwen35TensorDirectoryEntry[] {
  const entries = [
    programTensor("output_norm.weight", [2_560], GgmlType.F32),
    programTensor("token_embd.weight", [2_560, 248_320], GgmlType.Q6_K),
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
      const direct = shape.length === 1 || suffix === "ssm_conv1d.weight";
      entries.push(programTensor(
        `blk.${layer}.${suffix}`,
        shape,
        direct ? GgmlType.F32 : GgmlType.Q3_K,
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
  if (layout === undefined) throw new Error("unsupported test layout");
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

function weights(
  entries: readonly Qwen35TensorDirectoryEntry[],
): Qwen35WeightDirectoryView {
  return weightDirectoryFromViews(entries.map((entry) => weightView(entry)));
}

function weightDirectoryFromViews(
  tensors: readonly Qwen35TensorWeightView[],
): Qwen35WeightDirectoryView {
  const byName = new Map(tensors.map((tensor) => [tensor.name, tensor] as const));
  const bytes = tensors.reduce((sum, tensor) => sum + tensor.logicalBytes, 0n);
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

function layerZeroWeights(entries: readonly Qwen35TensorDirectoryEntry[]) {
  return weights(entries.filter(({ name }) => name.startsWith("blk.0.")));
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
      if (view === undefined) throw new Error("missing test activation");
      return view;
    },
  });
}

function hybridState(
  layer = 0,
  kind: "gated-deltanet" | "full-attention" = "gated-deltanet",
): Qwen35HybridLayerResources {
  if (kind === "full-attention") {
    const resource = (resourceKind: "key" | "value") => Object.freeze({
      id: `hybrid-state-layer-${layer}-${resourceKind}`,
      layer,
      kind: resourceKind,
      bytes: 2_048n,
      byteLength: 2_048n,
      shards: Object.freeze([Object.freeze({
        buffer: {},
        logicalByteOffset: 0n,
        logicalByteLength: 2_048n,
        allocatedByteLength: 2_048n,
      })]),
    });
    return Object.freeze({ layer, kind, key: resource("key"), value: resource("value") });
  }
  const resource = (resourceKind: "conv" | "recurrent", bytes: bigint) =>
    Object.freeze({
      id: `hybrid-state-layer-${layer}-${resourceKind}`,
      layer,
      kind: resourceKind,
      bytes,
      byteLength: bytes,
      shards: Object.freeze([Object.freeze({
        buffer: {},
        logicalByteOffset: 0n,
        logicalByteLength: bytes,
        allocatedByteLength: bytes,
      })]),
    });
  return Object.freeze({
    layer,
    kind,
    conv: resource("conv", 131_072n),
    recurrent: resource("recurrent", 2_097_152n),
  });
}

function uniforms(
  count = 13,
  buffer: object = {},
): readonly Qwen35ForwardBufferSlice[] {
  return Object.freeze(Array.from({ length: count }, (_, index) => Object.freeze({
    buffer,
    offset: index * 256,
    byteLength: 20,
  })));
}

function fixture() {
  const tensors = programDirectory();
  const program = buildQwen35Program({ config: QWEN35_4B_CONFIG, tensors });
  const invocation = program.invocations.find(
    (item): item is Extract<Qwen35Invocation, { kind: "gated-deltanet" }> =>
      item.kind === "gated-deltanet" && item.layer === 0,
  )!;
  return {
    program,
    invocation,
    weights: layerZeroWeights(tensors),
    workspace: workspace(),
    deltanetParameterLiveness:
      planQwen35ActivationWorkspace().deltanetParameterLiveness,
    state: hybridState(),
    uniforms: uniforms(),
    limits,
  };
}

async function planner(): Promise<(input: ReturnType<typeof fixture>) => {
  readonly layer: number;
  readonly commands: readonly {
    readonly stage: string;
    readonly kernel: { readonly id: string };
    readonly bindings: readonly { readonly buffer: object; readonly size: number }[];
    readonly uniformWords: readonly number[];
    readonly mutatesPersistentState: boolean;
  }[];
  readonly uniformCount: number;
  readonly stateSemantics: {
    readonly failureAfterSubmission: string;
    readonly advancePositionAfter: string;
  };
}> {
  const module = await import("../src/qwen35-deltanet-dispatch.js").catch(() => ({}));
  assert.equal(typeof module.planQwen35DeltaNetLayerDispatch, "function");
  return module.planQwen35DeltaNetLayerDispatch as never;
}

test("assembles the exact one-token DeltaNet command order and live buffers", async () => {
  const input = fixture();
  const plan = (await planner())(input);

  assert.equal(plan.layer, 0);
  assert.equal(plan.uniformCount, 13);
  assert.deepEqual(plan.commands.map(({ stage }) => stage), [
    "input-rms",
    "attention-gate-projection",
    "attention-qkv-projection",
    "deltanet-alpha-projection",
    "deltanet-beta-projection",
    "deltanet-parameters",
    "deltanet-conv",
    "deltanet-recurrent",
    "deltanet-gated-norm",
    "deltanet-output-projection",
    "attention-residual",
    "post-attention-rms",
    "ffn-gate-projection",
    "ffn-up-projection",
    "swiglu",
    "ffn-down-projection",
    "mlp-residual",
  ]);
  assert.deepEqual(
    plan.commands.filter(({ mutatesPersistentState }) => mutatesPersistentState)
      .map(({ stage }) => stage),
    ["deltanet-conv", "deltanet-recurrent"],
  );
  assert.equal(plan.commands[5]!.bindings[0]!.buffer,
    input.workspace.get("deltanet-beta").binding.buffer);
  assert.equal(plan.commands[5]!.bindings[4]!.buffer,
    input.workspace.get("full-attention-key").binding.buffer);
  assert.equal(plan.commands[5]!.bindings[5]!.buffer,
    input.workspace.get("full-attention-value").binding.buffer);
  assert.equal(plan.commands[6]!.bindings[2]!.buffer,
    input.state.kind === "gated-deltanet" ? input.state.conv.shards[0]!.buffer : null);
  assert.equal(plan.commands[7]!.bindings[3]!.buffer,
    input.state.kind === "gated-deltanet" ? input.state.recurrent.shards[0]!.buffer : null);
  assert.equal(plan.commands.at(-1)!.bindings[2]!.buffer,
    input.workspace.get("packed-embedding-output").binding.buffer);
  assert.equal(
    plan.stateSemantics.failureAfterSubmission,
    "persistent-state-indeterminate-dispose-required",
  );
  assert.equal(plan.stateSemantics.advancePositionAfter, "successful-queue-retirement");
});

test("preserves DeltaNet stage order across multiple physical matrix row views", async () => {
  const valid = fixture();
  const tensorName = "blk.0.attn_gate.weight";
  const target = valid.weights.get(tensorName)!;
  const sourceView = target.physicalRows[0]!;
  const firstRowCount = target.rowCount / 2;
  const firstByteLength = firstRowCount * target.rowBytes;
  const fragmented = valid.weights.tensors.map((tensor) =>
    tensor.name === tensorName
      ? Object.freeze({
          ...tensor,
          physicalRows: Object.freeze([
            Object.freeze({
              ...sourceView,
              rowCount: firstRowCount,
              byteLength: firstByteLength,
            }),
            Object.freeze({
              ...sourceView,
              firstRow: firstRowCount,
              rowCount: target.rowCount - firstRowCount,
              tensorByteOffset: firstByteLength,
              bufferByteOffset: sourceView.bufferByteOffset + firstByteLength,
              byteLength: sourceView.byteLength - firstByteLength,
            }),
          ]),
        })
      : tensor);

  const plan = (await planner())({
    ...valid,
    weights: weightDirectoryFromViews(fragmented),
    uniforms: uniforms(14),
  });
  const stages = plan.commands.map(({ stage }) => stage);
  const mathematicalStages = stages.filter(
    (stage, index) => index === 0 || stage !== stages[index - 1],
  );

  assert.equal(plan.uniformCount, 14);
  assert.equal(plan.commands.length, 18);
  assert.deepEqual(stages.slice(0, 4), [
    "input-rms",
    "attention-gate-projection",
    "attention-gate-projection",
    "attention-qkv-projection",
  ]);
  assert.deepEqual(
    plan.commands
      .filter(({ stage }) => stage === "attention-gate-projection")
      .map(({ uniformWords }) => uniformWords),
    [
      [firstRowCount, 2_560, 10, 0, 0],
      [target.rowCount - firstRowCount, 2_560, 10, 0, firstRowCount],
    ],
  );
  assert.deepEqual(mathematicalStages, [
    "input-rms",
    "attention-gate-projection",
    "attention-qkv-projection",
    "deltanet-alpha-projection",
    "deltanet-beta-projection",
    "deltanet-parameters",
    "deltanet-conv",
    "deltanet-recurrent",
    "deltanet-gated-norm",
    "deltanet-output-projection",
    "attention-residual",
    "post-attention-rms",
    "ffn-gate-projection",
    "ffn-up-projection",
    "swiglu",
    "ffn-down-projection",
    "mlp-residual",
  ]);
});

test("returns deeply frozen executable requests and encoded uniform words", async () => {
  const plan = (await planner())(fixture());

  assert.equal(Object.isFrozen(plan), true);
  assert.equal(Object.isFrozen(plan.commands), true);
  assert.ok(plan.commands.every((command) =>
    Object.isFrozen(command) &&
    Object.isFrozen(command.kernel) &&
    Object.isFrozen(command.bindings) &&
    command.bindings.every((binding) => Object.isFrozen(binding)) &&
    Object.isFrozen(command.uniformWords) &&
    Object.isFrozen(command.workgroups)));
  assert.deepEqual(plan.commands[0]!.uniformWords.slice(0, 2), [2_560, 2_560]);
  assert.deepEqual(plan.commands[10]!.uniformWords, [2_560, 0, 0, 0]);
  assert.deepEqual(plan.commands[14]!.uniformWords, [9_216, 0, 0, 0]);
});

test("rejects wrong invocation identity, layer kind, and direct tensor storage", async () => {
  const plan = await planner();
  const valid = fixture();
  const forgedInvocation = Object.freeze({ ...valid.invocation, layer: 1 });
  assert.throws(
    () => plan({ ...valid, invocation: forgedInvocation as typeof valid.invocation }),
    /program|invocation|layer/i,
  );
  assert.throws(
    () => plan({ ...valid, state: hybridState(0, "full-attention") }),
    /gated|layer|state/i,
  );

  const directName = "blk.0.ssm_conv1d.weight";
  const badTensors = valid.weights.tensors.map((tensor) =>
    tensor.name === directName
      ? Object.freeze({ ...tensor, ggmlType: GgmlType.Q3_K, storageType: "q3-k-112" })
      : tensor);
  assert.throws(
    () => plan({ ...valid, weights: weights(badTensors) }),
    /F32|direct|tensor/i,
  );
});

test("rejects insufficient, overlapping, and activation-state aliased buffers", async () => {
  const plan = await planner();
  const valid = fixture();
  assert.throws(
    () => plan({ ...valid, uniforms: uniforms(12) }),
    /uniform/i,
  );
  const sharedUniform = {};
  const aliasedUniforms = uniforms(13, sharedUniform).map((slot, index) =>
    index === 1 ? Object.freeze({ ...slot, offset: 0 }) : slot);
  assert.throws(
    () => plan({ ...valid, uniforms: aliasedUniforms }),
    /uniform|alias|overlap/i,
  );

  const sharedStateActivation = {};
  const aliasedWorkspace = workspace(new Map([
    ["attention-projection-primary", sharedStateActivation],
  ]));
  const state = hybridState();
  if (state.kind !== "gated-deltanet") throw new Error("bad fixture");
  const aliasedState = Object.freeze({
    ...state,
    conv: Object.freeze({
      ...state.conv,
      shards: Object.freeze([Object.freeze({
        ...state.conv.shards[0]!,
        buffer: sharedStateActivation,
      })]),
    }),
  });
  assert.throws(
    () => plan({ ...valid, workspace: aliasedWorkspace, state: aliasedState }),
    /alias|overlap|buffer/i,
  );
});

test("rejects overlapping physical ranges owned by distinct model tensors", async () => {
  const plan = await planner();
  const valid = fixture();
  const shared = {};
  const forged = valid.weights.tensors.map((tensor) =>
    tensor.name === "blk.0.attn_gate.weight" ||
      tensor.name === "blk.0.attn_qkv.weight"
      ? Object.freeze({
          ...tensor,
          physicalRows: Object.freeze(tensor.physicalRows.map((view) =>
            Object.freeze({ ...view, buffer: shared, bufferByteOffset: 0 }))),
        })
      : tensor);

  assert.throws(
    () => plan({ ...valid, weights: weightDirectoryFromViews(forged) }),
    /alias|overlap|weight/i,
  );
});

test("rejects activation views whose scalar contract is not F32", async () => {
  const plan = await planner();
  const valid = fixture();
  const invalidWorkspace = Object.freeze({
    get(kind: Qwen35ActivationResourceKind) {
      const view = valid.workspace.get(kind);
      return kind === "normalized-hidden"
        ? Object.freeze({ ...view, scalarType: "u32" as const })
        : view;
    },
  });

  assert.throws(
    () => plan({ ...valid, workspace: invalidWorkspace }),
    /activation|workspace|F32/i,
  );
});
