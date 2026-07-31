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
Cancelled prefill always resets partial model state before the session returns
to `ready`.

The browser loader validates the language manifest and runtime ABI, resolves
credential-free HTTPS shard URLs, authenticates immutable OPFS cache content,
loads the exact compiled tokenizer, builds the fixed 32-layer program, and
owns packed GPU allocations. Cached shards pass to the driver in awaited upload
lanes no larger than the device profile setting; the loader never creates a
complete CPU shard buffer.

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
