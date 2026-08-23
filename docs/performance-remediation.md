# Qwen3.5 Runtime Performance Remediation

## Status

The first correctness and hot-loop remediation is implemented. `auto` now
tries full segmented residency on every browser and falls back to rolling only
when the measured allocation ledger requires it. A 12-layer resident prefix is
available only through the explicit local `hybrid` experiment. The default
upload window is 64 MiB while authenticated OPFS reads remain bounded at
32 MiB.

The Mac Chrome WebGPU harness now executes and validates the language GEMV,
Qwen primitive, hybrid, vision, and staged GPU logits kernels. It also verifies
that streamed logits write candidates on the GPU and return one final token.
The physical iPhone results below are point-in-time evidence about the tested
runtime configurations. They are not general browser memory, throughput, or
thermal claims.

The earlier physical baseline was approximately 0.0865 generated tokens per
second, or 11.56 seconds per predicted token. It remains historical evidence
for the old rolling/token-major path, not a measurement of the current desktop
resident path.

The current exact Chrome resident path matches all 16 pinned llama.cpp token
IDs. Final exact runs range from 20.74 to 21.54 model target steps per second;
the latest run reaches 21.542 target steps and 22.972 emitted tokens per
second. It tracks 3,622,174,228 GPU bytes for the 3,557,215,360-byte fused
package. An earlier 26-token result did not have this exact full-model oracle
and is not a correctness-qualified speed result. The current result remains
below the 35 target-step-per-second development target.

The retained kernel changes use direct F32 activation reads for Q3 and Q8,
lane-major FP16 packing for Q5, fused residual-plus-RMSNorm, fused DeltaNet
recurrent-plus-gated-norm, and subgroup kernels when the requested WebGPU
device exposes the subgroup feature. A 128-thread twin-Q3 candidate stayed
exact but regressed from 21.246 to 20.799 target steps per second, so the
64-thread geometry remains selected.

The next material work is structural. A current decode reads about 3.05 GB of
packed weights and scores the complete 248,070-row decodable vocabulary for
every token. The measured short-prompt prefill rate reaches about 33 tokens per
second because it performs vocabulary scoring only once. Reaching 35 generated
tokens per second therefore requires both a faster transformer sweep and a
substantially cheaper exact vocabulary-selection path; minor dispatch removal
alone is insufficient.

The implementation must retain the selected model, quantization class,
multimodal requirement, 16K context target, public hosting requirement, and
security boundary. Early failures are evidence about an implementation. They
are not evidence that the browser target is impossible.

## Latest Feedback — 2026-08-14

This section records the current code review and remediation work. It does not
replace the dated physical observations below.

### Measurement contract

The earlier `generatedTokensPerSecond` value is not a clean steady-state decode
measurement. A fresh controlled prompt emits its cached prefill prediction
first, so a command that emits `N` tokens executes `N - 1` decode steps. Its
timer also includes the consumer that renders each fragment. The control-plane
metrics now report `targetStepCount` separately. Until the timing path also
records decode-only, readback, and rendered end-to-end intervals, historical
generated-tokens-per-second values must not be compared directly with the
35-token-per-second target.

### Implemented hot-loop work

- The DeltaNet recurrent kernel now uses one 128-lane workgroup per value head
  and two ordered state traversals: decay with memory accumulation, then
  update with output accumulation. It retains FP32 state and ascending key
  order. This removes two of the former four complete state traversals. The
  theoretical traffic reduction is 100,663,296 bytes per decoded token across
  the 24 DeltaNet layers. Numeric and physical-device validation remain
  required before this is accepted as a speed result.
- The all-layer rolling logits path now submits the final reduction, scalar
  copy, and map in one command submission. It keeps the persistent staged
  bind groups instead of clearing them after every four-tile flush, and it
  reuses the serialized four-byte readback buffer after unmapping it.
- Unit tests cover these ownership and measurement boundaries. The full suite,
  TypeScript checks, and production build passed locally on this date.

### Release and correctness gates still open

The accepted fused ABI-v2 package contains 3,557,215,360 tensor bytes. Its
full-resident Chrome run tracks 3,622,174,228 GPU bytes. Explicit hybrid
prefixes remain useful for iPhone allocation brackets, but no hybrid size is a
mobile default. This is a release-evidence gap, not a browser-memory
conclusion.

Fused quantization factors remain in FP32. Native-GGUF dequantization fixtures,
per-kernel GPU parity, and the 16-token greedy oracle now pass. Longer greedy,
layer-boundary, image, and perplexity comparisons remain open release gates.

The rolling vocabulary path remains a recovery path and still scores every
decodable row. Its byte counts depend on the selected compact or fused layout;
older fixed row-size values below are historical ABI-v1 evidence only.

### Device validation pending

The next iPhone bracket uses the local control system with the new code. It
must capture reset, prefill, target-step count, readback, rendering, peak
tracked memory, socket lifecycle, and output-validation status. Results will
be appended here only after the device run completes.

### Device attempt — 2026-08-14

Direct iPhone Mirroring control reconnected and navigated Safari successfully.
Safari could not establish a TCP connection to either local development test
route, even though the Mac test servers were listening and the Mac firewall was
disabled. No model, kernel, throughput, or memory result was recorded from
this attempt. Restore local phone-to-development-server reachability, then run
the same bounded WebGPU parity harness before recording a physical result.

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

### 1. Whole-model streaming was used for every token

The rolling policy still makes all 32 transformer layers transient. The
resident policy uploads the complete authenticated package into segmented GPU
buffers. `auto` selects resident on desktop and mobile unless an explicit
memory ledger limit requires rolling. Hybrid prefixes are local experiments,
not platform defaults.

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

### 4. Vocabulary scoring performed a full disk scan

The rolling tied-embedding fallback still processes 248,070 decodable rows in
1,024-row tiles. Every tile reads and uploads weights, but tile winners now stay
in a persistent GPU candidate buffer. One final token ID is read back after the
GPU reduction. Output tiles use four bounded staging buffers and reuse fences
only when the oldest buffer is recycled.

That creates, per predicted token:

- 525,908,400 bytes of output-weight traffic
- 243 scoring tiles in at most 61 GPU submissions
- one final GPU-to-CPU token readback
- bounded output-tile buffers
- queue retirement only for buffer reuse and cleanup

The desktop resident path and every hybrid path now keep the tied table on the
GPU. They do not create the disk-backed output store. The all-layer rolling
fallback retains the 243 tile reads, which remain visible in telemetry.

The current controlled configuration keeps four staged output tiles in flight. It
uses the same 1,024-row Q6_K kernel and GPU candidate ABI, adds about 4 MiB of
bounded scratch memory, and reduces the maximum output-scoring submissions
from 122 to 61. A regression test verifies that a buffer is not fenced before
the fourth tile forces reuse. This is a code-level improvement pending an
equivalent iPhone bracket test; it is not yet a throughput claim.

### 5. Hot-loop synchronization and ownership were too expensive

The resident path does not destroy weights in the token hot loop. Hybrid and
rolling execution now submit the embedding and adjacent resident layers as one
command batch. Earlier hybrid brackets needed fewer submissions as their
resident prefix grew; those measurements describe explicit experiments, not a
mobile default. The previous 24-layer path needed about 97 submissions because
each resident layer and each output batch was submitted and retired separately.

Each transient layer now uses one ownership fence after execution. Upload no
longer performs a redundant fence before execution. Bind-group cleanup removes
only groups that reference the transient layer buffers, so permanent groups
remain cached across tokens.

The upload window is independent from the OPFS read ceiling. Physical iPhone
tests found that a 64 MiB window can hold two 32 MiB reads before a queue fence.
For the same 15-token prefill and eight-token decode, queue retirements fell
from 668 to 305, total command time fell from 19.13 to 17.97 seconds, and decode
rose from 0.741 to 0.781 tokens per second. Tracked GPU peak remained
1,420,222,228 bytes. A later 32-token command completed without a disconnect at
0.638 tokens per second and the same tracked peak.

OPFS range reads cache the immutable `File` snapshot instead of reopening the
directory, handle, and snapshot for every tile or layer. A range contained in
one package segment is passed directly to `GPUQueue.writeBuffer`; it is no
longer copied into a second JavaScript array first.

Allocation is only 1.6% of the measured time. Allocation tuning must not be
treated as the main fix.

## Test and Telemetry Gaps

The unit suite validates the current execution contract and the performance
counter contracts. The Chrome harness validates shader compilation and CPU
parity. It does not prove physical iPhone performance viability.

The local control server now validates every declared language shard before it
starts. An incomplete local package fails at startup instead of reaching Safari
as a delayed `404` or range-status failure during a model load.

One iPhone rolling-path smoke run completed a 16-token prompt prefill in
17.835 seconds (0.897 tokens per second) and four generated tokens in 13.166
seconds (0.304 tokens per second). This is current physical evidence that the
rolling path is not release-quality. It is not an accuracy or thermal result.

The first resident-path attempt reached the browser lock wait while an older
local Qwen test tab still owned the origin lock. It did not reach a resident
allocation attempt, so it is not evidence of an iPhone memory boundary.

A later clean resident attempt showed two separate facts. A browser document
that reconnects after a local-server policy change retains its original policy;
it must reload before a new local residency setting takes effect. After dispose
and reload, the new document selected the explicit resident path, allocated
2,617,212,928 tracked GPU bytes, and began uploading the complete
2,555,346,944-byte package. Upload reached 1,824,496,640 bytes, then the
control socket was lost. Safari returned to an idle page without a `device.lost`
event, allocation diagnostic, or ready event. The bounded load lease later
ended as `timed_out`.

This is evidence that the current all-resident allocation and upload schedule
is unstable at that observed point on this device. It is not a model-size
conclusion. The next experiment must hold a measured subset of transformer
layers resident and stream only the remainder, then compare that candidate
against the rolling baseline with the same prompt and device temperature.

The first partial-residency implementation keeps a leading layer prefix in
segmented GPU buffers and streams only the remaining suffix. A 24-layer prefix
uses 1,506,283,008 resident model bytes before state and driver allocations.
On the iPhone it completed the 1,506,283,008-byte weight upload and reached
ready at 1,575,097,620 tracked GPU bytes. Its first short controlled run
prefilled 15 tokens in 7.284 seconds (2.059 tokens per second) and generated
four tokens in 6.391 seconds (0.626 tokens per second). This single run is
promising evidence, not a release benchmark: it needs output validation,
same-prompt bracket tests, repeated reloads, and thermal measurements.

With the same hybrid prefix and a four-tile output scorer, a later four-token
run completed in 5.617 seconds (0.712 tokens per second) after a 6.968-second
prefill. That is a small single-run change, not a benchmark. The command
reported four generated token IDs, but visual validation of decoded text was
not complete, so the result cannot yet be accepted as a correctness-qualified
speed result.

The next diagnostic hybrid load reached 270,528,800 uploaded weight bytes
before its originating browser document disappeared and a replacement document
reconnected. There was no `device.lost` event or allocation diagnostic. This
is evidence about the tested lifecycle and upload schedule, not a browser or
model-size limit. The control plane now terminals an in-flight non-reload
command as `timed_out` with `origin_document_replaced` when its original
document is replaced, so the replacement page cannot silently duplicate work.

A later same-device bracket compared the stable 8-layer default with 12 and 16
resident layers. The 8-layer run completed a 15-token prefill and eight-token
decode at 0.693 generated tokens per second, moving 16,659,112,448 bytes from
OPFS during the complete command. The 12-layer run completed the same command
at 0.741 generated tokens per second, moved 13,954,082,560 bytes, and peaked at
1,420,222,228 tracked GPU bytes. A repeated 32-token run also completed without
a disconnect at the same peak. The 16-layer profile loaded at 1,605,852,436
tracked GPU bytes, but Safari replaced the document 1.964 seconds after prompt
execution began. The 12-layer profile is therefore the current measured default;
the 16-layer result is a tested boundary for this device and runtime, not a
universal Safari memory limit.

Per-command telemetry now also records only the decoded text code-unit count
and the number of available performance snapshots. These numeric fields locate
an empty-rendering or missing-counter boundary without storing a prompt,
response, stack, URL, or device identity.

The local browser agent now resets only stale transport history when a local
control server restarts. It republishes its ready event without replaying an
old prompt. Per-command TTFT telemetry includes prompt prefill plus the first
decode token so it represents user-visible latency. It also records per-command
disk-read, GPU-upload, dispatch, submission, retirement, and readback deltas
without storing prompt or response content.

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

## Latest feedback — 2026-08-22

A full-model review traced the unrelated `Qwen` continuation to real inference
drift, not to a UI role label. Several independent defects contributed to the
bad result and the misleading performance evidence:

- DeltaNet mapped value head `h` to Q/K head `floor(h / 2)`. The pinned GGML
  repeat operation uses `h mod 16`, which maps 32 value heads as `0..15,
  0..15`. The CPU reference, WGSL kernel, fixtures, and architecture contract
  now use the same modulo mapping.
- Chrome exposed the `subgroups` adapter feature but omitted the provisional
  subgroup-size limit fields. The runtime interpreted the absent fields as a
  failed capability check and silently selected portable kernels. It now uses
  the subgroup path when the feature is present and the size fields are absent;
  if a browser exposes those fields, both must report 32.
- Uniform geometry was planned from raw WebIDL limits while execution selected
  kernels from a separate capability snapshot. This could allocate portable
  logits geometry and then execute subgroup logits. Planning and execution now
  share one named-field device-limit snapshot.
- The development server serves compiled `dev-dist` modules. Rebuilding only
  the normal `dist` directory made a page reload execute stale code. The repo
  now has an explicit `build:dev` command, the control startup uses it, and the
  local validation instructions require it before benchmark reloads.
- Q3 and Q8 kernels now consume F32 activations directly. Q5 keeps lane-major
  FP16 packing because the direct candidate reduced measured bandwidth from
  about 46.9 GB/s to 32.8 GB/s and slowed full decode.
- The reported generation rate previously mixed the cached first token with
  executed target steps. Telemetry now reports target-step and emitted-token
  rates separately and records exact reference-token mismatch counts.
- The polished chat left Qwen thinking enabled, so a short-answer request spent
  its token budget on a visible reasoning transcript. The user chat now fixes
  `enableThinking` to false while the tokenizer keeps its exact no-thinking
  template behavior.
- The maximum-token input updated runtime state only on a `change` event. A
  programmatic fill or an unblurred edit could show a new value while generation
  still used the old limit. The control now updates on every valid `input`
  event.
- Generation decoded `<|im_end|>` as an invisible special token but continued
  asking the driver for predictions. The model then generated a synthetic next
  user and assistant turn. The session now closes the driver at the pinned EOS
  token and never consumes the following prediction.

For `Write one short sentence about WebGPU.`, pinned llama.cpp produces the
first 16 token IDs `5793, 48213, 369, 264, 6278, 13775, 5165, 5995, 310, 3300,
1496, 55549, 11, 3238, 11258, 2528`. The full Chrome runtime now matches all
16 IDs with zero mismatches. The accepted exact bracket improved from 6.32 to
21.542 model target steps per second in the latest run. A 128-thread twin-Q3
candidate remained exact but fell to 20.799 target steps per second and was
reverted. A real 64-token UI prompt now stops naturally after the single
sentence `WebGPU is a modern graphics API designed to provide high-performance,
low-level access to GPU compute capabilities directly from JavaScript.`

Chrome full-model language validation now passes. Physical iPhone timing,
long-context stability, and vision validation remain separate release gates.

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
9. [x] Local controller preflight rejects incomplete language packages.
10. [x] One iPhone rolling-path text smoke test with safe performance telemetry.
11. [x] Resident tied output, selective bind-group ownership, OPFS snapshot
    reuse, direct single-segment staging, and resident-prefix command batching.
12. [x] Full unit, type, public-build, and Mac Chrome WebGPU parity gates.
13. [ ] Correctness-qualified iPhone hybrid bracket tests after this remediation.
14. [ ] Full text, image, lifecycle, and 16K acceptance run on the iPhone.

Do not accept a speed result when output validation fails. Do not preserve a
slow design only because an earlier test encoded it. Keep the last known-good
correct implementation available while each performance variable is tested.
