import {
  GgmlType,
  type GgufMetadataValue,
  type GgufTensorInfo,
} from "../../src/gguf.js";

/**
 * Sanitized directory facts from the pinned public language artifact.
 *
 * Payload offsets and weight values are deliberately absent. The fixture pins
 * the immutable artifact identity so reviewed metadata and shapes cannot be
 * mistaken for generic Qwen defaults.
 */
export const PINNED_QWEN35_GGUF_FIXTURE = Object.freeze({
  provenance: Object.freeze({
    repository: "https://huggingface.co/bartowski/Qwen_Qwen3.5-4B-GGUF",
    revision: "4168f45a16a1290d65a4ec0fa312ae917a4c15d6",
    file: "Qwen_Qwen3.5-4B-Q3_K_L.gguf",
    sha256: "41c3f1bf47e477693dab332e73347c7138d5e9fbfe74c6d2eaba590be1f3d20a",
  }),
  metadata: Object.freeze({
    "general.architecture": "qwen35",
    "qwen35.block_count": 33,
    "qwen35.context_length": 262_144,
    "qwen35.embedding_length": 2_560,
    "qwen35.feed_forward_length": 9_216,
    "qwen35.attention.head_count": 16,
    "qwen35.attention.head_count_kv": 4,
    "qwen35.attention.key_length": 256,
    "qwen35.attention.value_length": 256,
    "qwen35.attention.layer_norm_rms_epsilon": Math.fround(1e-6),
    "qwen35.full_attention_interval": 4,
    "qwen35.rope.dimension_count": 64,
    "qwen35.rope.dimension_sections": Object.freeze([11, 11, 10, 0]),
    "qwen35.rope.freq_base": 10_000_000,
    "qwen35.ssm.conv_kernel": 4,
    "qwen35.ssm.state_size": 128,
    "qwen35.ssm.group_count": 16,
    "qwen35.ssm.time_step_rank": 32,
    "qwen35.ssm.inner_size": 4_096,
    "qwen35.nextn_predict_layers": 1,
  } satisfies Readonly<Record<string, GgufMetadataValue>>),
  tensors: Object.freeze([
    Object.freeze({
      name: "output_norm.weight",
      dimensions: Object.freeze([2_560n]),
      type: GgmlType.F32,
      offset: 0n,
    }),
    Object.freeze({
      name: "token_embd.weight",
      dimensions: Object.freeze([2_560n, 248_320n]),
      type: GgmlType.Q6_K,
      offset: 0n,
    }),
    Object.freeze({
      name: "blk.0.ssm_conv1d.weight",
      dimensions: Object.freeze([4n, 8_192n]),
      type: GgmlType.F32,
      offset: 0n,
    }),
    Object.freeze({
      name: "blk.3.attn_q.weight",
      dimensions: Object.freeze([2_560n, 8_192n]),
      type: GgmlType.Q3_K,
      offset: 0n,
    }),
    Object.freeze({
      name: "blk.32.eh_proj.weight",
      dimensions: Object.freeze([5_120n, 2_560n]),
      type: GgmlType.Q3_K,
      offset: 0n,
    }),
  ] satisfies readonly GgufTensorInfo[]),
  normOracle: Object.freeze({
    sourceSemantics: "zero-centered-plus-one",
    ggufSemantics: "multiplicative",
    sourceWeight: Object.freeze([0, -0.25]),
    ggufWeight: Object.freeze([1, 0.75]),
    conversionReference:
      "https://github.com/ggml-org/llama.cpp/blob/master/conversion/qwen.py#L268-L276",
  }),
});
