# Architecture

## Runtime boundary

The browser package is an immutable input to a model-specific runtime. Each
manifest binds source artifacts, tokenizer or processor artifacts, shard byte
ranges, and the runtime ABI to immutable revisions and SHA-256 values. Runtime
code must validate this boundary before it fetches or interprets tensor bytes.

The GGUF converter reads only bounded ranges. It inventories the source tensor
types from the file directory, excludes the pinned `blk.32.*` MTP block and
complete `mtp` or `nextn` name segments with a stable policy reason, and
produces aligned shard segments that contain complete contiguous rows.
Language tensors use one of six explicit layouts:

- F32: native four-byte values;
- Q8_0: native 34-byte blocks reordered to 36 bytes;
- Q3_K: native 110-byte blocks reordered to 112 bytes;
- Q4_K: native, u32-aligned 144-byte blocks;
- Q5_K: native, u32-aligned 176-byte blocks; and
- Q6_K: native 210-byte blocks padded to 212 bytes.

Q8_0 moves its signed quant bytes behind a two-byte pad. Q6_K retains native
field order and adds trailing zero padding. These layouts keep each adjacent
block u32-aligned without changing quantized fields. Full tensors are not
expanded to floating-point storage during conversion or GEMV.

Each correctness-first GEMV invocation owns one local row. The shader reads its
packed block directly, reconstructs scalar values in registers, and writes to
`outputRowOffset + localRow`. Matrix shard plans require contiguous complete
rows. The bound weight buffer uses a u32-aligned shader-local byte offset; the
offset does not need to be a multiple of the quantization block size.

`tools/webgpu-kernel-harness.html` is the deterministic browser validation
surface. It compiles every language shader and compares two GPU rows per layout
against the independent packed CPU path. The parity check rejects non-finite
CPU or GPU output before it applies the numeric tolerance.

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
