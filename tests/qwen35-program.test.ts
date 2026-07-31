import assert from "node:assert/strict";
import test from "node:test";

import { GgmlType } from "../src/gguf.js";
import {
  QWEN35_4B_CONFIG,
  validateQwen35Config,
  type Qwen35Config,
} from "../src/qwen35-config.js";
import {
  buildQwen35Program,
  type Qwen35TensorDirectoryEntry,
} from "../src/qwen35-program.js";
import { PINNED_QWEN35_GGUF_FIXTURE } from "./fixtures/qwen35-4b-q3-k-l-sanitized.js";

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
          ["ssm_conv1d.weight", [4, 8_192]],
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
  const config = validateQwen35Config(PINNED_QWEN35_GGUF_FIXTURE.metadata, {
    productContextLength: 16_384,
    mtpPolicy: "exclude-block-32",
  });

  assert.deepEqual(config, QWEN35_4B_CONFIG);
  assert.equal(config.sourceMaxContextLength, 262_144);
  assert.equal(config.sourceBlockCount, 33);
  assert.equal(config.nextnPredictLayers, 1);
  assert.equal(config.rmsNormEpsilon, Math.fround(1e-6));
  assert.equal(config.productContextLength, 16_384);
  assert.deepEqual(config.fullAttentionLayers, [3, 7, 11, 15, 19, 23, 27, 31]);
  assert.ok(Object.isFrozen(config));
  assert.ok(Object.isFrozen(config.mropeSections));
});

test("rejects architecture, shape metadata, MTP policy, and product context drift", () => {
  for (const [key, value] of [
    ["general.architecture", "qwen3"],
    ["qwen35.block_count", 32],
    ["qwen35.nextn_predict_layers", 0],
    ["qwen35.embedding_length", 4_096],
    ["qwen35.ssm.inner_size", 8_192],
  ] as const) {
    const candidate = { ...PINNED_QWEN35_GGUF_FIXTURE.metadata };
    candidate[key] = value;
    assert.throws(() => validateQwen35Config(candidate), new RegExp(key.replaceAll(".", "\\.")));
  }
  assert.throws(
    () =>
      validateQwen35Config(PINNED_QWEN35_GGUF_FIXTURE.metadata, {
        mtpPolicy: "include-block-32" as never,
      }),
    /MTP policy/i,
  );
  assert.throws(
    () =>
      validateQwen35Config(PINNED_QWEN35_GGUF_FIXTURE.metadata, {
        productContextLength: 16_385,
      }),
    /product context.*16384/i,
  );
});

test("uses provenance-pinned parsed GGUF metadata and tensor facts", () => {
  assert.equal(PINNED_QWEN35_GGUF_FIXTURE.provenance.revision.length, 40);
  assert.equal(PINNED_QWEN35_GGUF_FIXTURE.provenance.sha256.length, 64);
  assert.equal(
    PINNED_QWEN35_GGUF_FIXTURE.tensors.find(
      (tensor) => tensor.name === "blk.3.attn_q.weight",
    )?.type,
    GgmlType.Q6_K,
  );
  assert.equal(
    PINNED_QWEN35_GGUF_FIXTURE.tensors.find(
      (tensor) => tensor.name === "blk.32.nextn.eh_proj.weight",
    )?.type,
    GgmlType.Q8_0,
  );
  assert.deepEqual(
    PINNED_QWEN35_GGUF_FIXTURE.tensors.find(
      (tensor) => tensor.name === "blk.0.ssm_conv1d.weight",
    )?.dimensions,
    [4n, 8_192n],
  );
  assert.equal(
    PINNED_QWEN35_GGUF_FIXTURE.metadata["qwen35.block_count"],
    33,
  );
  assert.equal(
    PINNED_QWEN35_GGUF_FIXTURE.metadata["qwen35.nextn_predict_layers"],
    1,
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
      "gated-deltanet",
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
    item.kind === "gated-deltanet" || item.kind === "full-attention",
  );
  assert.equal(attention.length, 32);
  assert.deepEqual(
    attention
      .filter((item) => item.kind === "full-attention")
      .map((item) => item.layer),
    [3, 7, 11, 15, 19, 23, 27, 31],
  );
  assert.equal(attention.every((item) => item.runnable), true);
  assert.equal(program.runnable, false);
  assert.equal(program.blockedBy, "weight-orchestration");
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

test("rejects forged Qwen3.5 configuration drift at the program boundary", () => {
  const drift: Readonly<Record<keyof Qwen35Config, unknown>> = {
    architecture: "qwen3",
    baseBlockCount: 0,
    sourceBlockCount: 32,
    nextnPredictLayers: 0,
    mtpBlock: 31,
    mtpPolicy: "include-block-32",
    embeddingLength: 4_096,
    feedForwardLength: 8_192,
    vocabularySize: 1,
    sourceMaxContextLength: 16_384,
    productContextLength: 0,
    attentionHeadCount: 8,
    keyValueHeadCount: 8,
    headDimension: 128,
    keyLength: 128,
    valueLength: 128,
    rmsNormEpsilon: 1e-5,
    fullAttentionInterval: 3,
    fullAttentionLayers: [3, 7, 11, 15, 19, 23, 27, 30],
    linearAttentionLayerCount: 23,
    fullAttentionLayerCount: 9,
    ropeFrequencyBase: 10_000,
    rotaryDimension: 128,
    mropeSections: [10, 11, 11, 0],
    ssmConvKernel: 3,
    ssmStateSize: 64,
    ssmGroupCount: 8,
    ssmTimeStepRank: 16,
    ssmInnerSize: 2_048,
  };
  for (const [key, value] of Object.entries(drift)) {
    const forged = {
      ...QWEN35_4B_CONFIG,
      [key]: value,
    } as Qwen35Config;
    assert.throws(
      () => buildQwen35Program({ config: forged, tensors: directory() }),
      new RegExp(`config.*${key}`, "i"),
    );
  }

  const validatedShortContext = validateQwen35Config(
    PINNED_QWEN35_GGUF_FIXTURE.metadata,
    { productContextLength: 1_024 },
  );
  assert.doesNotThrow(() =>
    buildQwen35Program({
      config: validatedShortContext,
      tensors: directory(),
    }),
  );
});

test("snapshots tensor inputs and exposes mutation-free bindings", () => {
  const tensors = directory();
  const original = tensors[0]!;
  const mutableShape = [...original.shape];
  const mutableEntry = { ...original, shape: mutableShape };
  tensors[0] = mutableEntry;
  const program = buildQwen35Program({
    config: QWEN35_4B_CONFIG,
    tensors,
  });
  const binding = program.tensorBindings.get(original.name)!;

  mutableShape[0] = 1;
  mutableEntry.name = "changed.weight";
  assert.equal(binding.tensor.name, "output_norm.weight");
  assert.deepEqual(binding.tensor.shape, [2_560]);
  assert.ok(Object.isFrozen(binding));
  assert.ok(Object.isFrozen(binding.tensor));
  assert.ok(Object.isFrozen(binding.tensor.shape));
  assert.ok(Object.isFrozen(binding.consumers));
  assert.ok(Object.isFrozen(program.tensorBindings));
  assert.equal(
    (program.tensorBindings as unknown as { set?: unknown }).set,
    undefined,
  );
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
