# Qwen3.5 browser session foundation

`Qwen35Session` is the browser ownership boundary for one Qwen3.5 4B model.
It acquires the exclusive origin Web Lock before any adapter, manifest, cache,
download, tokenizer, or model-byte work. The lock remains held until
`dispose()` finishes the driver queue, HybridState, packed weight, device, and
allocation-ledger cleanup.

The public lifecycle is `load`, `prefill`, `generate`, `cancel`, `reset`,
`dispose`, and `getMetrics`. Calls outside their legal state fail with stable
diagnostic codes. One generation iterator can own the driver at a time.
Cancellation is operation-scoped and does not release the origin lock.
Concurrent `cancel()` and `dispose()` calls await one cancellation promise.
The required driver `cancel()` method resolves only after active driver work is
quiescent. After it resolves, the session retires the inner generation iterator
and waits for its `finally` blocks before the session becomes reusable.
Cancelled prefill waits for driver cancellation, then resets partial model
state before the session returns to `ready`. A cancellation failure produces
the stable `driver-cancel-failed` diagnostic and leaves the session failed.
After a complete prefill, the caller must use `reset()` before another full
prefill. This prevents silent replacement of live model state.

The browser loader validates the language manifest and runtime ABI before any
cache or network access. The application must supply the SHA-256 of the exact
published converted manifest and its exact Hugging Face package base URL under
`/resolve/<40-hex-commit>/`. This repository does not invent a package hash
before that converted package is published. The loader also resolves
credential-free HTTPS shard URLs, authenticates immutable OPFS cache content,
loads the exact compiled tokenizer, builds the fixed 32-layer program, and
owns packed GPU allocations. It passes the driver a frozen package directory
that maps each tensor segment to its shard and exact offsets. Cached shards
pass to the driver in awaited upload lanes no larger than the device profile
setting; the loader never creates a complete CPU shard buffer.

The GPU allocation ledger is accounting, not a device-memory capability claim.
Without an explicit experimental budget, its limit is only the largest exact
metrics representation. An explicit `gpuLedgerLimitBytes` enforces that
experiment's chosen budget. Runtime metrics report the ledger's live current
and peak owned bytes, including driver scratch allocations.

## Current milestone boundary

This milestone does not install the complete Qwen layer execution scheduler.
Applications must provide a model-specific `Qwen35ExecutionDriverFactory`.
Production loading fails with `qwen-execution-driver-not-installed` when that
factory is absent. The runtime does not emit placeholder tokens or claim full
inference support.

Text conversations use the authenticated tokenizer and exact Qwen chat
template. The session enforces the 16,384-token total context without
truncation. Image input remains type-compatible but fails with
`vision-not-loaded` until the vision execution milestone.
