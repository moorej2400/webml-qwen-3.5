# Qwen3.5 Runtime Performance Remediation

## Status

The performance remediation baseline is implemented. Desktop `auto` execution
uses resident segmented weights. Apple mobile `auto` execution keeps the
rolling policy that was required for the known-good iPhone path. Explicit
`resident` and `rolling` policies remain available for bracketed device tests.

The Mac Chrome WebGPU harness now executes and validates the language GEMV,
Qwen primitive, hybrid, vision, and staged GPU logits kernels. It also verifies
that streamed logits write candidates on the GPU and return one final token.
No physical iPhone memory-fit, throughput, or thermal claim is made here until
that device is tested.

The earlier physical baseline was approximately 0.0865 generated tokens per
second, or 11.56 seconds per predicted token. It remains historical evidence
for the old rolling/token-major path, not a measurement of the current desktop
resident path.

The implementation must retain the selected model, quantization class,
multimodal requirement, 16K context target, public hosting requirement, and
security boundary. Early failures are evidence about an implementation. They
are not evidence that the browser target is impossible.

## Historical Token Budget

| Stage | Average per predicted token | Share |
| --- | ---: | ---: |
| Rolling GPU allocation | 181 ms | 1.6% |
| OPFS read and GPU upload | 2,319 ms | 20.1% |
| Transformer execution and cleanup | 7,008 ms | 60.6% |
| Final norm and vocabulary scoring | 2,054 ms | 17.8% |
| **Total** | **11,563 ms** | **100%** |

One transformer sweep moves exactly 2,028,898,304 bytes of layer weights. The
disk-backed output projection adds 525,908,400 bytes for each predicted token.
The old rolling path therefore moved about 2.55 GB from OPFS through CPU
staging to the GPU for every generated token.

## Findings And Current State

### 1. All transformer layers are streamed for every token

The rolling policy still makes all 32 transformer layers transient. The
resident policy instead uploads the complete authenticated package into
segmented GPU buffers and keeps it available for decode. Desktop `auto` selects
resident unless an explicit memory ledger limit requires rolling. Apple mobile
`auto` selects rolling until physical bracket tests prove a better policy.

The old rolling policy solved a peak-residency failure by making disk act as
virtual GPU memory. It also made steady-state decode depend on more than 2 GB
of storage traffic per token. The resident path removes that traffic on devices
that can hold the package. Physical-device tests, not assumed browser limits,
must determine whether that path is suitable for iPhone.

Weight residency alone is not sufficient. The runtime now exposes disk reads,
GPU uploads, dispatches, queue submissions, queue retirements, readbacks, and
GPU allocation categories. Performance budgets can subtract a load/prefill
baseline before checking per-generated-token limits.

### 2. Core GPU kernels are scalar correctness kernels

`src/mixed-gemv.ts` now uses a 64-lane workgroup with shared partials and a
tree reduction. Quantized metadata is still decoded per lane for the current
portable ABI; a bucketed GEMM path is a separate future optimization.

`src/hybrid-kernels.ts` now uses cooperative workgroups for the attention and
recurrent reductions. DeltaNet state-cell decay and update work is parallel,
while the recurrent memory and output reductions retain their original FP32
order. Full-attention preparation and online attention use shared reductions.

At short context, the current static plan emits about 552 transformer compute
dispatches per token. Full attention emits more dispatches as KV pages grow.

The current portable GEMV ABI still decodes metadata per lane. The next
optimization would be a bucketed packed GEMM, but it remains intentionally
separate from this correctness-preserving baseline.

### 3. Prefill repeats the single-token decode schedule

The runtime now processes bounded chunks of four prompt or visual tokens in
layer-major order. Each layer's commands are submitted together, and only the
final prompt step performs norm and vocabulary selection. Rolling/mobile
execution loads each streamed layer once per chunk, then fences once before
releasing that layer. If a constrained device cannot allocate the optional
prefill pool, the serial executor remains available as a transactional fallback.

This makes prompt cost scale as:

```text
ceil(prompt tokens / 4) * 2,028,898,304 transformer bytes for rolling prefill
```

The observed short prompt required about 34.5 GB of transformer-layer traffic
before output. A 1,024-token image would require about 2.08 TB. A 16K prefill
would require about 33 TB before accounting for the additional long-context
attention cost.

The current chunk path still uses the existing per-token packed GEMV ABI inside
each layer. It preserves the serial FP32 DeltaNet recurrence order while
parallelizing independent state-cell decay and update work. A bucketed packed
GEMM remains a separate future optimization. The Chrome harness checks the
resulting kernels against their CPU references.

### 4. Vocabulary scoring performs a full disk scan and CPU round trips

The rolling tied-embedding fallback still processes 248,070 decodable rows in
1,024-row tiles. Every tile reads and uploads weights, but tile winners now stay
in a persistent GPU candidate buffer. One final token ID is read back after the
GPU reduction. Output tiles use bounded double buffering and reuse fences only
when a buffer is recycled.

That creates, per predicted token:

- 525,908,400 bytes of output-weight traffic
- 243 scoring tiles in at most 122 GPU submissions
- one final GPU-to-CPU token readback
- bounded output-tile buffers
- queue retirement only for buffer reuse and cleanup

The desktop resident path has no disk-backed tied store. The streamed path is
retained only for devices that need it, and its remaining 243 tile reads are
reported by performance telemetry for physical-device comparison.

### 5. Hot-loop synchronization and ownership are too expensive

The resident path does not destroy rolling layer weights in the token hot loop.
The rolling fallback uses bounded upload-retirement windows and one ownership
fence before each streamed layer is released.

The bind-group cache clear prevents stale groups from retaining destroyed
rolling buffers. It is a safe response to the current ownership model, but it
adds steady-state churn. Resident arenas and generation-scoped bindings should
make this cleanup unnecessary in the token hot path.

Allocation is only 1.6% of the measured time. Allocation tuning must not be
treated as the main fix.

## Test and Telemetry Gaps

The unit suite validates the current execution contract and the performance
counter contracts. The Chrome harness validates shader compilation and CPU
parity. It does not prove physical iPhone performance viability.

The runtime now exposes gates for:

- transformer and vocabulary bytes read per token
- GPU uploads per token
- compute dispatches and queue submissions per token
- GPU readbacks per token
- warm TTFT
- prefill tokens per second
- sustained decode tokens per second
- peak and steady-state allocations
- Long-context throughput decline and thermal decline still require repeated
  physical-device runs.

The session metrics expose permanent, transient, state, disk-read, upload,
dispatch, submission, retirement, and readback values separately from the live
ledger bytes. A baseline-aware budget check prevents load traffic from being
counted as decode traffic.

## Working Runtime Comparison

The working Gemma runtime uses the transferable strategy that this project
needs:

- Model weights are uploaded once and remain GPU-resident.
- Large tensors are segmented without materializing the complete tensor in JS.
- Disk-backed embeddings read only requested rows.
- Decode uses a prebuilt execution sequence.
- Prefill has a separate block program.

The Qwen architecture requires different kernels, but it does not require a
token-major whole-model disk stream.

## Implementation Evidence

1. [x] Performance counters and baseline-aware regression limits.
2. [x] Resident segmented weights with rolling fallback.
3. [x] Workgroup-parallel packed GEMV.
4. [x] Validated cooperative DeltaNet state work with ordered FP32 reductions.
5. [x] Cooperative full-attention and online-attention kernels.
6. [x] GPU candidate reduction with one final readback.
7. [x] Bounded resident layer-major prefill.
8. [x] Mac Chrome shader compilation and CPU parity.
9. [ ] Bracketed physical-device residency, throughput, and thermal tests.
10. [ ] Full text, image, lifecycle, and 16K acceptance run on the iPhone.

Do not accept a speed result when output validation fails. Do not preserve a
slow design only because an earlier test encoded it. Keep the last known-good
correct implementation available while each performance variable is tested.
