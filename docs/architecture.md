# Architecture

## Runtime boundary

The browser package is an immutable input to a model-specific runtime. Each
manifest binds source artifacts, tokenizer or processor artifacts, shard byte
ranges, and the runtime ABI to immutable revisions and SHA-256 values. Runtime
code must validate this boundary before it fetches or interprets tensor bytes.

The GGUF converter reads only bounded ranges. It inventories the source tensor
types from the file directory, records excluded MTP tensors with a stable
policy reason, and produces aligned shard segments without splitting
quantization blocks. Q3_K tensors remain quantized: native 110-byte
superblocks are reordered into 112-byte WebGPU blocks. Full tensors are not
expanded to FP16 during conversion.

## Scope

This project targets one Qwen model family and its required multimodal
components. A generic model loader is outside the production design. External
inference frameworks may be used only as independent development references;
they are not production dependencies.

## Experimental feasibility

A failure on one device is evidence about that tested implementation, not
proof that the browser target is impossible. Do not impose an assumed
model-size or browser-memory ceiling. Test materially different allocation,
packing, streaming, kernel, and scheduling strategies before reporting a
boundary.

Document physical-device results with the exact strategy and environment.
Describe repeated limits as tested boundaries, not universal browser rules.
Do not reduce the selected model, quantization, multimodal support, or context
target unless the product requirements change explicitly.
