# Third-party notices

## ggml and llama.cpp quantization formats

The Q3_K, Q4_K, Q5_K, Q6_K, and Q8_0 field orders and dequantization formulas
in this repository are clean TypeScript and WGSL reimplementations of the
public ggml quantization formats maintained by the llama.cpp project:

- <https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.h>
- <https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c>

llama.cpp and ggml are available under the MIT License. They are development
references only and are not runtime dependencies of this project.

## Qwen3.5 model configuration

The pinned Qwen3.5 4B dimensions and hybrid-layer schedule are derived from the
public Qwen model configuration:

- <https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/config.json>

Qwen3.5 model materials are available under the Apache License 2.0. The model
is an input to this project and is not a runtime software dependency.
