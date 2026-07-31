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
The planner applies the manifest shard, segment, and excluded-tensor ceilings
before it appends each entry. Language tensors use one of six explicit layouts:

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
offset does not need to be a multiple of the quantization block size. A shader
guard rejects any 2D invocation whose flattened row index would overflow u32.

`tools/webgpu-kernel-harness.html` is the deterministic browser validation
surface. It executes every language GEMV, reusable primitive, and packed
embedding layout against a CPU reference. Fixtures use distinct row data,
nonzero packed and output offsets, and sentinel slots around each output.
The parity check rejects non-finite CPU or GPU output before it applies the
numeric tolerance.

## Qwen3.5 4B program

The language program is static and model-specific. Its immutable configuration
validates the parsed GGUF metadata for 33 source blocks and one next-token
prediction layer. The program keeps 32 base layers and excludes `blk.32.*`. It
also validates hidden width 2,560, FFN width 9,216, vocabulary 248,320, 16 query
heads, four KV heads, 256-value query/key/value heads, Gated DeltaNet
dimensions, the exact parsed F32 RMSNorm epsilon, partial rotary dimensions,
and GGUF M-RoPE sections `[11, 11, 10, 0]`.

Layers 3, 7, 11, 15, 19, 23, 27, and 31 use full attention. The remaining 24
layers use typed Gated DeltaNet placeholders. Each layer has a fixed invocation
sequence: input RMSNorm, its exact attention placeholder, residual add,
post-attention RMSNorm, gate and up projections, SwiGLU, down projection, and
residual add. Full attention and DeltaNet recurrence remain explicit
non-runnable operators until their own runtime phases are implemented.

The tensor-directory validator requires all 32 base layers with exact names and
shapes, including linear-attention convolution weights shaped `[4, 8192]`. It
rejects unknown base tensors and the `blk.32.*` MTP block. Tensor
storage is selected independently from the six manifest layouts; quantization
does not change a logical tensor shape. Program construction revalidates every
configuration field, snapshots tensor entries and shapes, and exposes bindings
through an immutable lookup facade.

Embedding decodes one packed vocabulary row after checking token and table
bounds. It never expands the 248,320 by 2,560 table. The same
`token_embd.weight` allocation owns output logits: logits are tiled,
row-sharded GEMV consumers, not a duplicate output matrix. The GPU embedding
family shares the GEMV register decoders for all six layouts and dispatches
only the selected row. Its planner bounds every shader-facing value to u32 and
rejects a dispatch that exceeds the live device workgroup limit.

Reusable correctness kernels cover FP32-accumulating RMSNorm, residual add,
SiLU, fused SwiGLU, attention output gating, per-head Q/K RMSNorm, partial
M-RoPE, and stable tiled top-k. GGUF RMSNorm weights are already
multiplicative: the conversion adds one to zero-centered source weights except
for the linear-attention gated norm, whose stored weight is directly
multiplicative. Partial M-RoPE rotates only the first 64 values of each
256-value head. It uses interleaved temporal-height-width frequency ownership
for `[11, 11, 10]` and split-half `rotate_half` lanes, then copies lanes 64
through 255 to the output. CPU and WGSL use the same f32 angle arithmetic and
explicit range reduction at large positions. Top-k rejects non-finite
candidates, breaks equal-score ties by vocabulary index, and reports a valid
result count when fewer than `k` finite candidates exist.

The source GGUF advertises a native maximum context of 262,144 tokens. The
current product contract selects 16,384 tokens and stores the native maximum
separately. This selection is a product limit, not a browser capability
ceiling.

## Scope

This project targets one Qwen model family and its required multimodal
components. A generic model loader is outside the production design. External
inference frameworks may be used only as independent development references;
they are not production dependencies.

This program does not yet implement full-attention score/KV math, Gated
DeltaNet recurrence, tokenization, session lifecycle, OPFS persistence, vision,
UI, or a control server.

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
