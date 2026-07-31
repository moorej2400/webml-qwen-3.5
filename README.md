# WebML Qwen 3.5

This repository is an experimental, model-specific WebGPU runtime for a Qwen
3.5 4B Q3_K_L language model and its vision projector.

The production runtime will use browser-native modules and custom WebGPU
kernels. It will not depend on Transformers.js, ONNX Runtime, llama.cpp, or a
generic inference framework. The current milestone provides the artifact
foundation:

- a bounded, random-access GGUF v3 directory parser;
- a versioned and validated browser package manifest;
- deterministic, streaming shard planning with an explicit MTP exclusion
  policy;
- exact Q3_K 110-byte to WebGPU-aligned 112-byte block conversion; and
- generated binary fixtures that do not include model data.

## Development

Node.js 24 or later is required.

```sh
npm install --ignore-scripts
npm test
npm run typecheck
npm run build
npm run test:converter
```

See [Architecture](docs/architecture.md) for the artifact and feasibility
rules.
