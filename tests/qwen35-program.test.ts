import assert from "node:assert/strict";
import test from "node:test";

import { GgmlType, type GgufMetadataValue } from "../src/gguf.js";
import {
  QWEN35_4B_CONFIG,
  validateQwen35Config,
} from "../src/qwen35-config.js";
import {
  buildQwen35Program,
  type Qwen35TensorDirectoryEntry,
} from "../src/qwen35-program.js";

function metadata(): Record<string, GgufMetadataValue> {
  return {
    "general.architecture": "qwen35",
    "qwen35.block_count": 32,
    "qwen35.context_length": 262_144,
    "qwen35.embedding_length": 2_560,
    "qwen35.feed_forward_length": 9_216,
    "qwen35.attention.head_count": 16,
    "qwen35.attention.head_count_kv": 4,
    "qwen35.attention.key_length": 256,
    "qwen35.attention.value_length": 256,
    "qwen35.attention.layer_norm_rms_epsilon": 1e-6,
    "qwen35.full_attention_interval": 4,
    "qwen35.rope.dimension_count": 64,
    "qwen35.rope.dimension_sections": [11, 11, 10, 0],
    "qwen35.rope.freq_base": 10_000_000,
    "qwen35.ssm.conv_kernel": 4,
    "qwen35.ssm.state_size": 128,
    "qwen35.ssm.group_count": 16,
    "qwen35.ssm.time_step_rank": 32,
    "qwen35.ssm.inner_size": 4_096,
  };
}

function tensor(
  name: string,
  shape: readonly number[],
  ggmlType: number = GgmlType.Q3_K,
): Qwen35TensorDirectoryEntry {
  const storageTypes = new Map([
    [GgmlType.F32, "f32"],
    [GgmlType.Q8_0, "q8-0-36"],
    [GgmlType.Q3_K, "q3-k-112"],
    [GgmlType.Q4_K, "q4-k-144"],
    [GgmlType.Q5_K, "q5-k-176"],
    [GgmlType.Q6_K, "q6-k-212"],
  ] as const);
  return {
    name,
    shape,
    ggmlType,
    storageType: storageTypes.get(ggmlType)!,
  };
}

function directory(): Qwen35TensorDirectoryEntry[] {
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
          ["ssm_conv1d.weight", [4, 6_144]],
          ["ssm_dt.bias", [32]],
          ["ssm_norm.weight", [128]],
          ["ssm_out.weight", [4_096, 2_560]],
        ] as const;
    for (const [suffix, shape] of [...attention, ...mlp]) {
      const scalar =
        shape.length === 1 ||
        suffix === "ssm_conv1d.weight" ||
        suffix === "ssm_alpha.weight" ||
        suffix === "ssm_beta.weight";
      entries.push(
        tensor(
          `blk.${layer}.${suffix}`,
          shape,
          scalar ? GgmlType.F32 : GgmlType.Q3_K,
        ),
      );
    }
  }
  return entries;
}

test("validates and freezes the exact Qwen3.5 4B GGUF configuration", () => {
  const config = validateQwen35Config(metadata(), {
    productContextLength: 16_384,
    mtpPolicy: "exclude-block-32",
  });

  assert.deepEqual(config, QWEN35_4B_CONFIG);
  assert.equal(config.sourceMaxContextLength, 262_144);
  assert.equal(config.productContextLength, 16_384);
  assert.deepEqual(config.fullAttentionLayers, [3, 7, 11, 15, 19, 23, 27, 31]);
  assert.ok(Object.isFrozen(config));
  assert.ok(Object.isFrozen(config.mropeSections));
});

test("rejects architecture, shape metadata, MTP policy, and product context drift", () => {
  for (const [key, value] of [
    ["general.architecture", "qwen3"],
    ["qwen35.block_count", 33],
    ["qwen35.embedding_length", 4_096],
    ["qwen35.ssm.inner_size", 8_192],
  ] as const) {
    const candidate = metadata();
    candidate[key] = value;
    assert.throws(() => validateQwen35Config(candidate), new RegExp(key.replaceAll(".", "\\.")));
  }
  assert.throws(
    () => validateQwen35Config(metadata(), { mtpPolicy: "include-block-32" as never }),
    /MTP policy/i,
  );
  assert.throws(
    () => validateQwen35Config(metadata(), { productContextLength: 16_385 }),
    /product context.*16384/i,
  );
});

test("builds the exact deterministic 32-layer static program", () => {
  const program = buildQwen35Program({
    config: QWEN35_4B_CONFIG,
    tensors: directory(),
  });

  assert.equal(program.invocations.length, 292);
  assert.deepEqual(
    program.invocations.slice(0, 11).map((invocation) => invocation.kind),
    [
      "embedding",
      "rms-norm",
      "linear-attention-placeholder",
      "residual-add",
      "rms-norm",
      "gemv",
      "gemv",
      "swiglu",
      "gemv",
      "residual-add",
      "rms-norm",
    ],
  );
  const attention = program.invocations.filter((item) =>
    item.kind.endsWith("attention-placeholder"),
  );
  assert.equal(attention.length, 32);
  assert.deepEqual(
    attention
      .filter((item) => item.kind === "full-attention-placeholder")
      .map((item) => item.layer),
    [3, 7, 11, 15, 19, 23, 27, 31],
  );
  assert.equal(program.invocations.at(-3)?.kind, "rms-norm");
  assert.deepEqual(program.invocations.at(-2), {
    kind: "tiled-tied-logits",
    weight: "token_embd.weight",
    tiedWeightOwner: "embedding",
    rows: 248_320,
    columns: 2_560,
  });
  assert.equal(program.invocations.at(-1)?.kind, "top-k-placeholder");
  assert.equal(program.tensorBindings.get("token_embd.weight")?.consumers.length, 2);
});

test("rejects missing, extra, mis-shaped, wrong-interval, and MTP tensors", () => {
  const required = directory();
  const cases: Array<[Qwen35TensorDirectoryEntry[], RegExp]> = [
    [required.slice(1), /missing.*output_norm\.weight/i],
    [[...required, tensor("unused.weight", [256])], /unexpected.*unused\.weight/i],
    [
      required.map((entry) =>
        entry.name === "blk.0.attn_qkv.weight"
          ? { ...entry, shape: [2_560, 4_096] }
          : entry,
      ),
      /shape.*blk\.0\.attn_qkv\.weight/i,
    ],
    [
      required.map((entry) =>
        entry.name === "blk.3.attn_q.weight"
          ? { ...entry, name: "blk.3.attn_qkv.weight" }
          : entry,
      ),
      /blk\.3\./i,
    ],
    [[...required, tensor("blk.32.attn_q.weight", [2_560, 8_192])], /MTP.*blk\.32/i],
  ];
  for (const [tensors, expected] of cases) {
    assert.throws(
      () => buildQwen35Program({ config: QWEN35_4B_CONFIG, tensors }),
      expected,
    );
  }
});

test("accepts all six manifest layouts without coupling shapes to quantization", () => {
  const types = [
    [GgmlType.F32, "f32"],
    [GgmlType.Q8_0, "q8-0-36"],
    [GgmlType.Q3_K, "q3-k-112"],
    [GgmlType.Q4_K, "q4-k-144"],
    [GgmlType.Q5_K, "q5-k-176"],
    [GgmlType.Q6_K, "q6-k-212"],
  ] as const;
  const required = directory();
  for (const [ggmlType, storageType] of types) {
    const candidate = required.map((entry) =>
      entry.shape.length === 2 ? { ...entry, ggmlType, storageType } : entry,
    );
    assert.doesNotThrow(() =>
      buildQwen35Program({ config: QWEN35_4B_CONFIG, tensors: candidate }),
    );
  }
});
