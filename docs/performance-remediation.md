# Qwen3.5 Runtime Performance Remediation

## Status

The current runtime proves that the selected Qwen3.5 4B package can execute in
WebGPU on the target mobile browser. It is not a viable inference runtime yet.

Physical-device telemetry measured approximately 0.0865 generated tokens per
second, or 11.56 seconds per predicted token. This is an architecture failure,
not a tuning miss.

The implementation must retain the selected model, quantization class,
multimodal requirement, 16K context target, public hosting requirement, and
security boundary. Early failures are evidence about an implementation. They
are not evidence that the browser target is impossible.

## Measured Token Budget

| Stage | Average per predicted token | Share |
| --- | ---: | ---: |
| Rolling GPU allocation | 181 ms | 1.6% |
| OPFS read and GPU upload | 2,319 ms | 20.1% |
| Transformer execution and cleanup | 7,008 ms | 60.6% |
| Final norm and vocabulary scoring | 2,054 ms | 17.8% |
| **Total** | **11,563 ms** | **100%** |

One transformer sweep moves exactly 2,028,898,304 bytes of layer weights. The
disk-backed output projection adds 525,908,400 bytes for each predicted token.
The current path therefore moves about 2.55 GB from OPFS through CPU staging to
the GPU for every generated token.

## Blocking Findings

### 1. All transformer layers are streamed for every token

`src/qwen35-rolling-layer-weights.ts` deliberately makes all 32 transformer
layers transient. Only the final output norm remains resident. Every token
allocates, reads, uploads, executes, fences, and destroys each layer in program
order.

This policy solved a peak-residency failure by making disk act as virtual GPU
memory. It also made steady-state decode depend on more than 2 GB of storage
traffic per token. The memory strategy must change from whole-model rolling to
resident packed weights split across conservative GPU buffers. Physical-device
tests, not assumed browser limits, must determine buffer and arena sizes.

Weight residency alone is not sufficient. Removing all measured allocation,
upload, and vocabulary costs would leave about seven seconds of transformer
execution per token.

### 2. Core GPU kernels are scalar correctness kernels

`src/mixed-gemv.ts` uses `@workgroup_size(1)`. One invocation owns one output
row and serially loops over every quantization block and input element. The Q3
decoder repeats packed-byte extraction, scale decoding, and delta loading in
the innermost loop.

`src/hybrid-kernels.ts` also uses single-invocation kernels for important model
operations. The DeltaNet recurrent kernel launches one invocation per value
head and performs nested 128 by 128 loops serially. Full-attention preparation
and online attention use one invocation per query head.

At short context, the current static plan emits about 552 transformer compute
dispatches per token. Full attention emits more dispatches as KV pages grow.

Required work:

- Replace scalar packed GEMV with a column-parallel workgroup kernel and a
  stable reduction.
- Decode quantization metadata once per block or cooperative tile instead of
  once per scalar value.
- Parallelize DeltaNet state work while preserving FP32 recurrent semantics and
  exact mutation order.
- Parallelize full-attention head dimensions and online accumulation.
- Build device-specific kernel families when physical results justify them.
- Fuse operations only after the unfused parallel kernels match the reference.

### 3. Prefill repeats the single-token decode schedule

`src/qwen35-greedy-driver.ts` calls the complete token engine once for every
prompt token and every projected visual token. The `prefill` phase label does
not select a batched GEMM or a layer-major chunked execution plan.

This makes prompt cost scale as:

```text
prompt tokens * 2,028,898,304 transformer bytes
```

The observed short prompt required about 34.5 GB of transformer-layer traffic
before output. A 1,024-token image would require about 2.08 TB. A 16K prefill
would require about 33 TB before accounting for the additional long-context
attention cost.

Required work:

- Create a separate prefill program.
- Process bounded token chunks while each layer is resident.
- Use bucketed packed GEMM for projections.
- Implement a bounded parallel DeltaNet scan or another validated chunked
  recurrence strategy.
- Preserve a serial FP32 oracle for correctness comparisons.

### 4. Vocabulary scoring performs a full disk scan and CPU round trips

The disk-backed tied embedding processes 248,070 decodable rows in 1,024-row
tiles. Every predicted token performs 243 tile iterations. Every tile reads and
uploads weights, runs two compute commands, waits, and then performs two serial
four-byte GPU readbacks.

That creates, per predicted token:

- 525,908,400 bytes of output-weight traffic
- 243 scoring batches
- 486 GPU-to-CPU readback submissions
- 486 temporary readback buffers and mappings
- additional queue retirement fences

Required work:

- Keep tile winners and scores on the GPU.
- Reduce all tile candidates on the GPU.
- Read back one final token ID.
- Test resident, segmented-resident, and streamed output-projection policies.
- Retain a disk-backed fallback only when device evidence requires it.

### 5. Hot-loop synchronization and ownership are too expensive

The current path uses a 32 MiB upload lane, which produces about 76 upload
retirement fences per transformer sweep. Each layer then waits after dispatch,
waits again before weight destruction, clears the complete bind-group cache,
and opens an asynchronous validation scope for its batch.

The bind-group cache clear prevents stale groups from retaining destroyed
rolling buffers. It is a safe response to the current ownership model, but it
adds steady-state churn. Resident arenas and generation-scoped bindings should
make this cleanup unnecessary in the token hot path.

Allocation is only 1.6% of the measured time. Allocation tuning must not be
treated as the main fix.

## Test and Telemetry Gaps

The unit suite validates the current execution contract, including all-layer
streaming and serial prefill. It does not validate performance viability.

Add gates for:

- transformer and vocabulary bytes read per token
- GPU uploads per token
- compute dispatches and queue submissions per token
- GPU readbacks per token
- warm TTFT
- prefill tokens per second
- sustained decode tokens per second
- peak and steady-state allocations
- long-context throughput decline
- thermal decline during repeated generation

The UI currently reports live ledger bytes. That value is not false, but it is
incomplete. It can show a small resident footprint while inference depends on
multi-gigabyte disk traffic. Expose permanent, transient, state, disk-read, and
upload metrics separately.

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

## Required Implementation Order

1. Add performance counters and regression limits for bytes, submissions,
   readbacks, and time per token.
2. Establish resident packed transformer weights in Chrome with segmented
   buffers and exact output parity.
3. Replace packed GEMV with a workgroup-parallel kernel.
4. Replace the serial DeltaNet recurrent kernel with a validated parallel
   implementation.
5. Parallelize full-attention preparation and online attention.
6. Keep vocabulary candidates on the GPU and use one final readback.
7. Implement chunked, layer-major prefill.
8. Validate correctness and performance in Chrome.
9. Run bracketed physical-device tests and adapt residency, buffer sizes,
   kernels, and scheduling from the evidence.
10. Validate text, image, lifecycle, and 16K acceptance requirements.

Do not accept a speed result when output validation fails. Do not preserve a
slow design only because an earlier test encoded it. Keep the last known-good
correct implementation available while each performance variable is tested.
