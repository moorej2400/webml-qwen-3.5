# Clean-Slate Qwen3.5 4B WebGPU Runtime

## Product objective

Build a custom, model-specific WebGPU runtime for Qwen3.5 4B Q3_K_L. The
production application must not use Transformers.js, ONNX Runtime, llama.cpp,
or another generic inference framework. Official Qwen implementations and
llama.cpp are development oracles only.

The first public release must provide:

- Qwen3.5 4B Q3_K_L text generation;
- one image per user turn;
- a 16,384-token total context, including visual tokens;
- a polished mobile-first chat interface;
- local physical-iPhone telemetry, commands, reload, crash recovery, and
  benchmark automation;
- GitHub Pages deployment; and
- converted browser artifacts in a separate public Hugging Face repository.

Correctness, lifecycle safety, and stability are release gates. Performance
optimization continues after those gates and must use measured physical-device
evidence.

## Feasibility and persistence contract

This project does not treat conventional browser guidance, reported memory
limits, adapter limits, failed allocations, compiler failures, or an early
device crash as proof that the product is impossible. They describe one tested
design only.

Rules of thumb such as a 1 GB browser-model ceiling are not product constraints.
Prior browser implementations have already shown that larger models can run
well when packing, residency, allocation, and scheduling are designed for the
physical device. The working assumption is that this selected model can work
and can achieve unusually strong performance. Experiments must discover the
design that makes this true; they must not search for a reason to lower the
goal.

When a design fails, change one or more of these variables and test again:

- tensor packing and shard layout;
- CPU and GPU allocation ownership;
- buffer size and segmentation;
- download, cache, and upload scheduling;
- kernel fusion, workgroup shape, and shader family;
- prefill and attention algorithms;
- KV representation;
- vision streaming and residency; or
- page lifecycle and cross-tab coordination.

Do not impose an assumed model-size or browser-memory ceiling. Do not reduce
the selected model, Q3_K_L quantization class, multimodal requirement, 16K
context target, public-hosting requirement, or security boundary without an
explicit product decision. A reported boundary requires repeated physical
device evidence across materially different approaches and must be described
as a tested boundary, not a universal browser limit.

The implementation must be persistent and experimental. A slow or failed path
requires a pivot, a new hypothesis, and another measured test. It is not a
reason to stop.

## Pinned model inputs

Pin immutable revisions, byte sizes, and SHA-256 hashes before conversion. The
authoritative machine-readable record is `model-sources.json`.

- Language source: `Qwen_Qwen3.5-4B-Q3_K_L.gguf`, 2,665,441,248 bytes.
- Vision source: `mmproj-Qwen_Qwen3.5-4B-bf16.gguf`, 675,569,216 bytes.
- Official configuration and tokenizer: `Qwen/Qwen3.5-4B` at the revision in
  `model-sources.json`.

The converter must inventory the actual GGML type of every tensor. The Q3 name
does not mean that every tensor uses three bits. The initial language package
must exclude the optional MTP layer and record every exclusion.

## Repository and publication boundary

The historical implementation remains available in Git history. New runtime
work starts from a clean tree and does not import the old inference runtime.

This is a public repository. GitHub stores source, manifests, tests, public
fixtures, checksums, and documentation only. It must never contain model
weights, caches, certificates, credentials, private prompts, telemetry logs,
personal data, company data, LAN details, or machine-specific paths.

Converted weights must be published to a separate public repository such as:

`<hugging-face-user>/Qwen3.5-4B-Q3-K-L-WebGPU`

## Model package conversion

Create a repository-owned streaming Node and TypeScript converter that:

1. parses the exact GGUF tensor directory;
2. records tensor names, shapes, source offsets, and actual GGML types;
3. excludes and records the MTP tensors;
4. converts complete tensor rows into aligned WebGPU layouts without expanding
   the full model to FP16;
5. preserves Q3 values while allowing aligned blocks, transposition, combined
   projections, and vocabulary tiling;
6. converts the BF16 vision projector into independent layer-addressable
   shards; and
7. emits a versioned manifest with source identities, licenses, hashes,
   layouts, shard ranges, tokenizer identity, image-processing settings, and
   runtime ABI.

The converter must stream bounded ranges. It must not load the complete model
into memory.

## Physical-iPhone control system

Build the measurement and recovery system before full-model tuning.

- Serve local development over iPhone-trusted HTTPS.
- Inject the browser agent only in the local development build.
- Automatically connect each local phone document without a pairing code or
  phone token entry. Issue a one-use, short-lived WSS ticket only after exact
  same-origin `Host` and `Origin` checks; do not persist that ticket in the page.
- Gate the phone WSS connection with that single-use ticket and repeat the exact
  origin checks at upgrade. This is a trusted-local-network same-origin boundary,
  not cryptographic client authentication; apply ticket lifetime, one-use, rate,
  and pending-ticket caps.
- Bind the separate operator API only to `127.0.0.1`.
- Use durable device and tab IDs, a new document ID per reload, command IDs,
  benchmark IDs, and monotonic event sequences.
- Add local-only server-derived correlation to control state and ignored JSONL:
  coarse OS family/version and the direct socket IP. Never trust forwarded
  headers or record raw user agents.
- Support `load`, `dispose`, `runPrompt`, `cancelPrompt`, `getState`,
  `warmReload`, and `coldAppReload`.
- Report `accepted`, `started`, and one terminal state: `completed`, `failed`,
  `cancelled`, `timed_out`, or `indeterminate`.
- Complete reload only after the replacement document reconnects and reports
  ready.
- Reconcile reconnects and retries without generating a prompt twice.
- Prefer protocol-driven recovery. Use iPhone Mirroring or remote desktop only
  when the page or socket cannot recover, then return to the protocol.

Write bounded, sanitized JSONL only under ignored `.local/runs/` paths. Omit
unknown fields and free-form strings; do not redact and store raw prompts,
responses, URLs, cookies, forwarded or private addresses, tokens, or unredacted
stacks. The direct socket IP is the sole allowed address
field and is local-only device correlation. Collect load phases, range and cache behavior, shader compilation,
tracked CPU and GPU bytes, image processing, prompt processing, TTFT, token
rate, thermal drift, lifecycle changes, socket loss, errors, and `device.lost`.

## Custom runtime architecture

Use TypeScript, direct WebGPU, hand-written WGSL, and a small static execution
plan. Do not build a general tensor framework.

The stable session API is:

```ts
interface Qwen35Session {
  load(options: LoadOptions): Promise<void>;
  prefill(input: TextOrImageConversation): Promise<SequenceState>;
  generate(options: GenerateOptions): AsyncIterable<GeneratedToken>;
  cancel(): Promise<void>;
  reset(): Promise<void>;
  dispose(): Promise<void>;
  getMetrics(): RuntimeMetrics;
}
```

Required internal boundaries:

- `ModelPackage`: manifest and hash validation;
- `DeviceProfile`: adapter features and measured limits;
- `GpuArena`: packed sharded buffers and allocation ledger;
- `KernelRegistry`: kernel selection by operation, layout, phase, and profile;
- `Qwen35Program`: static 32-layer execution sequence;
- `HybridState`: Gated DeltaNet convolution and recurrent state plus
  full-attention KV pages;
- `Tokenizer`: exact compiled vocabulary, chat template, and incremental UTF-8
  decoding; and
- `VisionEncoder`: preprocessing, streamed projector execution, M-RoPE
  metadata, and projected visual tokens.

Implement in this correctness order:

1. CPU quantization reference and Q3 unpack fixtures.
2. Packed Q3 GEMV for decode and bucketed GEMM for prefill.
3. RMSNorm, residual, SwiGLU, RoPE/M-RoPE, and tiled logits reduction.
4. One online full-attention layer without a full score matrix.
5. One Gated DeltaNet layer with persistent convolution and recurrent state.
6. Full greedy text decode.
7. Chunked prefill and 16K state handling.
8. Vision preprocessing, 24-layer vision path, merger/projector, and visual
   token injection.
9. Sampling, streaming output, cancellation, and conversation reuse.

Keep Gated DeltaNet recurrent calculations in FP32 until numerical evidence
proves that a lower precision is safe.

## Safari memory architecture

- Acquire an exclusive origin-wide Web Lock before adapter creation, model
  access, downloads, or cache work. A blocked tab performs no model work.
- Hold the lock until asynchronous GPU cleanup completes.
- Keep weights packed and GPU-resident in multiple conservative arenas.
- Start with buffers no larger than 256 MiB and change that size when physical
  evidence supports a better strategy.
- Fetch and upload one 32 MiB range at a time. Pivot to 16 or 8 MiB after
  measured failures.
- Never call `arrayBuffer()` on a complete model or clone or tee large
  responses for diagnostics.
- Cache immutable shards in OPFS with authenticated temporary state and
  ready-last atomic publication.
- Keep transient CPU weight buffers below 64 MiB.
- Use packed tiled embedding and output projection with GPU top-k reduction.
- Start with paged FP16 KV. Segment allocation before considering compression.
- If FP16 prevents 16K stability, test per-page INT8 KV with scales against the
  FP16 oracle.
- Stream vision layers from OPFS by default and retain only projected visual
  tokens.

These starting values are experiment settings, not claims about hard browser
limits.

## Application requirements

Build a lightweight TypeScript DOM application without React or another UI
framework. Recreate the proven mobile-first Gemma design language without
copying its inference runtime.

The UI must provide safe-area handling, streaming messages, progress, model
controls, cancellation, live speed data, chat-first loading, image select and
paste, a removable image thumbnail, upload validation, image-processing
progress, context usage, vision-memory state, accessible focus states, reduced
motion, readable contrast, responsive type, and explicit model, cache, and
error states.

Default to one image and about 1,024 visual tokens. The runtime can change image
resolution from measured stability and accuracy evidence but must stay within
the 16K total context.

Apply the repository's `gpt-taste` design preflight during UI implementation,
adapted to a chat application. Never bundle private control endpoints, local
addresses, local WSS tickets, or operator credentials in the public build.

## CI and GitHub Pages

Add a SHA-pinned GitHub Actions Pages workflow. Before deployment, run unit
tests, converter fixture tests, Chromium and WebKit browser tests, public-safety
checks, and a reviewed static build.

Normal CI must not download the multi-gigabyte model. It must validate the
pinned remote manifest and representative range requests. The production Pages
application loads immutable browser-native shards from the public Hugging Face
repository. The production build must contain no local control agent, and
local tests must leave Git status clean.

## Adaptive implementation loop

For every optimization:

1. Record one hypothesis and one changed variable.
2. Run a correctness fixture.
3. Run bracketed physical-iPhone tests: baseline, candidate, baseline.
4. Compare load time, TTFT, prefill, decode, image time, memory, failures, and
   thermal behavior.
5. Keep, revert, or pivot from the evidence.
6. Restore and smoke-test the last known-good configuration.

The implementing agent can change kernel fusion, workgroup sizes, shard sizes,
packed layouts, vision streaming, prefill algorithms, KV representation,
buffer ownership, and scheduling when device evidence disproves the current
design. It must update the architecture decision and continue without asking
for approval.

Primary pivots include:

- slow compact Q3 unpack: test aligned nibble expansion or another
  value-preserving browser pack;
- recurrent drift: return to sequential FP32 and isolate the fused operation;
- slow prefill: test a bounded chunked parallel DeltaNet scan;
- load termination: reduce overlap, staging, shard size, or network lane;
- vision residency failure: stream smaller tiles and retain projected tokens;
- 16K allocation failure: segment KV, then test validated INT8 KV;
- shader regression: split fusion or select a device-specific kernel family;
  and
- faster public baseline: profile the exact difference and adopt its useful
  kernel or schedule without importing its generic runtime.

## Correctness and release gates

Correctness fixtures must validate GGUF metadata, quant blocks, converted
tensors, tokenizer IDs, chat templates, image preprocessing, visual tokens,
per-layer outputs, logits, and deterministic greedy tokens against pinned
official Transformers and llama.cpp development oracles. A speed result is
invalid when output validation fails.

Physical-device acceptance requires:

- text and one-image conversations at 16,384 total tokens;
- three cold loads and five warm loads;
- ten reload, load, and dispose cycles;
- repeated 512-token generations without unbounded allocation growth;
- proof that only one of two Safari tabs allocates or downloads;
- remote load, prompt, cancel, dispose, warm reload, cold reload, reconnect,
  suspected-crash classification, and fallback reopening;
- phone discovery within five seconds, telemetry within four seconds, and
  reconnect within ten seconds after connectivity returns;
- proof that retry or reconnect never generates a prompt twice; and
- tracked image time, warm TTFT, prefill rate, sustained decode rate, peak
  allocations, and five-minute thermal decline.

There is no unsupported hard token-per-second promise. Continue optimization
after the stable gate and accept only repeatable improvements above normal run
variance with correctness preserved. Compare compatible public browser
runtimes on the same device and prompts, while optimizing against this
project's own last known-good baseline.

Public release is complete only when GitHub and Hugging Face repositories are
public, Git history preserves the sanitized old snapshot, no private or local
artifacts are tracked, GitHub Pages passes the complete CI gate, and production
text and image inference load from the pinned public model package.
