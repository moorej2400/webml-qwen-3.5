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
embedding layout against a CPU reference, then compiles all six hybrid
attention and DeltaNet kernels. The full-attention preparation fixture executes
at position 16,383 and verifies normalized/rotated Q, the per-head gate split,
the packed FP16 K/V row, and two untouched suffix words. Fixtures use distinct
row data, nonzero packed and output offsets, and sentinel slots around each
output. The parity check rejects non-finite CPU or GPU output before it applies
the numeric tolerance.

## Qwen3.5 4B program

The language program is static and model-specific. Its immutable configuration
validates the parsed GGUF metadata for 33 source blocks and one next-token
prediction layer. The program keeps 32 base layers and excludes `blk.32.*`. It
also validates hidden width 2,560, FFN width 9,216, vocabulary 248,320, 16 query
heads, four KV heads, 256-value query/key/value heads, Gated DeltaNet
dimensions, the exact parsed F32 RMSNorm epsilon, partial rotary dimensions,
and GGUF M-RoPE sections `[11, 11, 10, 0]`.

Layers 3, 7, 11, 15, 19, 23, 27, and 31 use full attention. The remaining 24
layers use typed Gated DeltaNet operators. Each layer has a fixed invocation
sequence: input RMSNorm, its exact attention operator, residual add,
post-attention RMSNorm, gate and up projections, SwiGLU, down projection, and
residual add. The complete program is runnable through one model-specific
greedy driver. The driver derives its uniform geometry from the physical
weight views before allocation and connects every invocation without a general
tensor framework.

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

HybridState allocates no dummy state. At 16,384 tokens it owns exactly eight
logical packed FP16 K/V pairs, 24 FP32 `[8192,4]` convolution states, and 24
FP32 `[32,128,128]` recurrent states. Their total is 590,348,288 bytes
(563 MiB). Allocation is rollback-safe. A failed partial reset poisons the
state so only disposal remains legal.

Full attention keeps the Qwen GGUF query projection as 16 per-head
`[q256,gate256]` records. A fused preparation kernel splits those records,
applies multiplicative Q/K RMSNorm and exact partial M-RoPE, and writes the
current K/V row as packed FP16 u32 words. The online kernel maps each group of
four query heads to one K/V head and uses a stable running maximum,
denominator, and value accumulator without a full score matrix. Its dispatch
contract validates `1 <= tokenCount <= position + 1 <= capacity <= 16384`.

Gated DeltaNet keeps convolution and recurrent math in FP32. It shifts raw QKV
through oldest-to-current convolution taps, applies SiLU, maps each value head
to Q/K head `h % 16`, computes beta and negative-`ssm_a` decay, and preserves
the required decay, read, beta-delta, rank-one update, then query order. The
serial prefill path is the correctness oracle for later bounded parallel scans.

One text token executes the packed embedding, all 32 layers, and, when a token
is requested, final RMSNorm plus tiled tied logits in one compute batch. The
driver waits for queue retirement before it advances HybridState, then reads
back only the selected u32 token. Uniform uploads follow each command's exact
planner-assigned buffer and offset; they do not assume that command order and
uniform allocation order are the same. Text decode uses M-RoPE positions
`[position, position, position]`.

Correctness-first prefill processes tokens serially and runs logits only for
the final prompt token. Generation keeps the predicted token separate from the
last emitted-but-not-yet-ingested token. This permits a later generation call
to continue without repeating or skipping a token. Cancellation that crosses
submitted generation state poisons the driver and fails closed. Cancellation
while paused at a yielded token preserves the pending continuation.

The source GGUF advertises a native maximum context of 262,144 tokens. The
current product contract selects 16,384 tokens and stores the native maximum
separately. This selection is a product limit, not a browser capability
ceiling.

## Scope

This project targets one Qwen model family and its required multimodal
components. A generic model loader is outside the production design. External
inference frameworks may be used only as independent development references;
they are not production dependencies.

The language program, session lifecycle, origin lock, immutable OPFS cache,
local control protocol, and greedy text driver are connected. Physical-device
validation, vision execution, the final application UI, and public model-shard
publication remain later runtime phases.

## Local iPhone control boundary

The local HTTPS development server injects the control agent only into its own
response. A document automatically requests one short-lived WSS ticket from the
same origin. The server requires the exact configured `Host` and `Origin` for
ticket issuance and WebSocket upgrade, then consumes the ticket before it
accepts the first protocol frame. No pairing code, device token, or reusable
browser secret exists in the page or its storage. The ticket is a same-origin,
trusted-local-network gate rather than cryptographic phone authentication: a
network client able to forge those browser headers is not cryptographically
identified. Short ticket lifetimes, one-use consumption, a pending-ticket cap,
and an issue-rate cap bound reconnect abuse. The independently authenticated
operator API remains bearer-authenticated and bound to `127.0.0.1`.

The server, rather than the browser protocol payload, enriches ignored local
JSONL with durable device/tab/document IDs, a coarse OS family/version parsed
from the TLS handshake user agent, and the direct socket IP. It ignores all
forwarding headers and discards raw user-agent text. This information is solely
for local correlation and is excluded from public builds and tracked artifacts.

## Experimental feasibility

A failure on one device is evidence about that tested implementation, not
proof that the browser target is impossible. Do not impose an assumed
model-size or browser-memory ceiling. Test materially different allocation,
packing, streaming, kernel, and scheduling strategies before reporting a
boundary.

Conventional browser guidance, reported limits, adapter limits, and failed
allocations are inputs to the next experiment. They are not stop conditions.
Change the design and measure the result on the physical target.

Document physical-device results with the exact strategy and environment.
Describe repeated limits as tested boundaries, not universal browser rules.
Do not reduce the selected model, quantization, multimodal support, or context
target unless the product requirements change explicitly.
