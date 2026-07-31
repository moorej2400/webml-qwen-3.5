# WebML Qwen 3.5

This repository is an experimental, model-specific WebGPU runtime for a Qwen
3.5 4B Q3_K_L language model and its vision projector.

The production runtime will use browser-native modules and custom WebGPU
kernels. It will not depend on Transformers.js, ONNX Runtime, llama.cpp, or a
generic inference framework. The current milestone provides the artifact
foundation and mixed-quant compute layer:

- a bounded, random-access GGUF v3 directory parser;
- a versioned and validated browser package manifest;
- deterministic, streaming complete-row shard planning with a pinned MTP
  exclusion policy;
- explicit F32, Q8_0, Q3_K, Q4_K, Q5_K, and Q6_K WebGPU layouts;
- exact block repacking without full-tensor dequantization;
- direct packed-weight GEMV kernels with row-aware shard dispatch;
- an immutable Qwen3.5 4B configuration and exact 32-layer tensor validator;
- a typed static program with runnable attention and DeltaNet operators while
  complete weight orchestration remains explicitly blocked;
- FP32 CPU and WGSL references for reusable vector math, M-RoPE, and top-k;
- exact 16K HybridState planning for eight packed FP16 K/V pairs, 24 FP32
  convolution states, and 24 FP32 recurrent states;
- online full attention with fused Q/gate preparation, Q/K RMSNorm, M-RoPE,
  bounded packed K/V publication, GQA, and stable streamed softmax;
- correctness-first Gated DeltaNet convolution, parameter, recurrent-update,
  gated-normalization, decode, and serial-prefill paths;
- single-row packed embedding decode and tied row-sharded output projection;
- deterministic CPU references and a browser WebGPU parity harness; and
- generated binary fixtures that do not include model data.

[`model-sources.json`](model-sources.json) pins the public source revisions,
artifact hashes, and inspected language tensor inventory used by this runtime.

## Development

Node.js 24 or later is required.

```sh
npm install --ignore-scripts
npm test
npm run typecheck
npm run build
npm run test:converter
```

After `npm run build`, serve the repository over HTTPS or localhost and open
`tools/webgpu-kernel-harness.html` to compile the six GEMV layouts, reusable
Qwen primitives, packed embeddings, and six hybrid attention/DeltaNet kernels.
The harness executes the full-attention preparation stage at position 16,383
and verifies its packed cache row and suffix against CPU references.

See [Architecture](docs/architecture.md) for the artifact and feasibility
rules. See [Third-party notices](THIRD_PARTY_NOTICES.md) for quantization
algorithm attribution.
