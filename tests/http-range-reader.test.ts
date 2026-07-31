import assert from "node:assert/strict";
import test from "node:test";

import {
  DEFAULT_RANGE_BYTES,
  HttpRangeReader,
  RangeReadError,
  type RangeFetch,
} from "../src/http-range-reader.js";

function parseRequestRange(header: string): { start: number; end: number } {
  const match = /^bytes=(\d+)-(\d+)$/.exec(header);
  assert.ok(match);
  return { start: Number(match[1]), end: Number(match[2]) };
}

function rangedResponse(
  bytes: Uint8Array,
  start: number,
  total: number,
  options: { status?: number; etag?: string; contentRange?: string } = {},
): Response {
  const end = start + bytes.byteLength - 1;
  const headers = new Headers({
    "content-length": String(bytes.byteLength),
    "content-range":
      options.contentRange ?? `bytes ${start}-${end}/${total}`,
  });
  if (options.etag !== undefined) {
    headers.set("etag", options.etag);
  }
  return new Response(bytes, {
    status: options.status ?? 206,
    headers,
  });
}

test("defaults to 32 MiB ranges and serial chunk ownership", async () => {
  assert.equal(DEFAULT_RANGE_BYTES, 32 * 1024 * 1024);
  const sourceBytes = Uint8Array.from({ length: 23 }, (_, index) => index);
  const requested: Array<{ start: number; end: number }> = [];
  const fetcher: RangeFetch = async (_url, init) => {
    const range = parseRequestRange(init.range);
    requested.push(range);
    return rangedResponse(
      sourceBytes.subarray(range.start, range.end + 1),
      range.start,
      sourceBytes.byteLength,
      { etag: '"fixture-v1"' },
    );
  };
  const reader = new HttpRangeReader(fetcher, { rangeStrategies: [8, 4, 2] });
  const output: number[] = [];
  let activeConsumers = 0;
  let peakConsumers = 0;

  await reader.stream(
    {
      locator: "https://public.example/immutable/shard.bin",
      byteLength: sourceBytes.byteLength,
      immutableUrl: false,
      expectedEtag: '"fixture-v1"',
    },
    async (chunk) => {
      activeConsumers += 1;
      peakConsumers = Math.max(peakConsumers, activeConsumers);
      await Promise.resolve();
      output.push(...chunk);
      activeConsumers -= 1;
    },
  );

  assert.deepEqual(output, [...sourceBytes]);
  assert.deepEqual(requested, [
    { start: 0, end: 7 },
    { start: 8, end: 15 },
    { start: 16, end: 22 },
  ]);
  assert.equal(peakConsumers, 1);
  assert.equal(reader.metrics.peakTransientBytes, 8);
});

test("pivots from 32 to 16 to 8 strategy after retryable failures", async () => {
  const requestedLengths: number[] = [];
  const fetcher: RangeFetch = async (_url, init) => {
    const { start, end } = parseRequestRange(init.range);
    const length = end - start + 1;
    requestedLengths.push(length);
    if (length > 2) {
      return new Response(null, { status: 503 });
    }
    return rangedResponse(new Uint8Array(length), start, 6, {
      etag: '"fixture-v1"',
    });
  };
  const reader = new HttpRangeReader(fetcher, {
    rangeStrategies: [8, 4, 2],
    minimumRangeAttempts: 2,
  });

  await reader.stream(
    {
      locator: "https://public.example/immutable/shard.bin",
      byteLength: 6,
      immutableUrl: false,
      expectedEtag: '"fixture-v1"',
    },
    async () => {},
  );

  assert.deepEqual(requestedLengths.slice(0, 3), [6, 4, 2]);
  assert.deepEqual(
    reader.metrics.events
      .filter((event) => event.kind === "range-pivot")
      .map((event) => [event.fromBytes, event.toBytes]),
    [
      [8, 4],
      [4, 2],
    ],
  );
});

test("rejects ignored or contradictory range responses", async () => {
  let ignoredBodyCancelled = false;
  const ignoredBody = new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(new Uint8Array([1, 2, 3]));
    },
    cancel() {
      ignoredBodyCancelled = true;
    },
  });
  const ignored = new HttpRangeReader(
    async () => ({
      status: 200,
      headers: new Headers(),
      body: ignoredBody,
    }),
    { rangeStrategies: [8, 4, 2] },
  );
  await assert.rejects(
    ignored.stream(
      {
        locator: "https://public.example/immutable/shard.bin",
        byteLength: 3,
        immutableUrl: true,
      },
      async () => {},
    ),
    (error: unknown) =>
      error instanceof RangeReadError && error.code === "range-status-invalid",
  );
  assert.equal(ignoredBodyCancelled, true);

  const wrongRange = new HttpRangeReader(
    async () =>
      rangedResponse(new Uint8Array(3), 0, 3, {
        contentRange: "bytes 1-3/3",
      }),
    { rangeStrategies: [8, 4, 2] },
  );
  await assert.rejects(
    wrongRange.stream(
      {
        locator: "https://public.example/immutable/shard.bin",
        byteLength: 3,
        immutableUrl: true,
      },
      async () => {},
    ),
    (error: unknown) =>
      error instanceof RangeReadError &&
      error.code === "content-range-invalid",
  );
});

test("requires stable ETag or explicit immutable URL semantics", async () => {
  const noEtag = new HttpRangeReader(
    async (_url, init) => {
      const range = parseRequestRange(init.range);
      return rangedResponse(
        new Uint8Array(range.end - range.start + 1),
        range.start,
        3,
      );
    },
    { rangeStrategies: [2] },
  );
  await assert.rejects(
    noEtag.stream(
      {
        locator: "https://public.example/mutable/shard.bin",
        byteLength: 3,
        immutableUrl: false,
      },
      async () => {},
    ),
    (error: unknown) =>
      error instanceof RangeReadError && error.code === "etag-required",
  );

  const etags = ['"v1"', '"v2"'];
  const changedEtag = new HttpRangeReader(
    async (_url, init) => {
      const range = parseRequestRange(init.range);
      return rangedResponse(
        new Uint8Array(range.end - range.start + 1),
        range.start,
        4,
        { etag: etags.shift() },
      );
    },
    { rangeStrategies: [2] },
  );
  await assert.rejects(
    changedEtag.stream(
      {
        locator: "https://public.example/mutable/shard.bin",
        byteLength: 4,
        immutableUrl: false,
      },
      async () => {},
    ),
    (error: unknown) =>
      error instanceof RangeReadError && error.code === "etag-changed",
  );
});

test("cancellation stops later ranges and reaches the active fetch signal", async () => {
  const sourceBytes = new Uint8Array(8);
  const controller = new AbortController();
  let requestCount = 0;
  let observedSignal: AbortSignal | undefined;
  const reader = new HttpRangeReader(
    async (_url, init) => {
      requestCount += 1;
      observedSignal = init.signal;
      const range = parseRequestRange(init.range);
      return rangedResponse(
        sourceBytes.subarray(range.start, range.end + 1),
        range.start,
        sourceBytes.byteLength,
      );
    },
    { rangeStrategies: [4] },
  );

  await assert.rejects(
    reader.stream(
      {
        locator: "https://public.example/immutable/shard.bin",
        byteLength: sourceBytes.byteLength,
        immutableUrl: true,
      },
      async () => controller.abort(),
      controller.signal,
    ),
    { name: "AbortError" },
  );
  assert.equal(observedSignal, controller.signal);
  assert.equal(requestCount, 1);
});

test("bounds minimum-strategy retries and keeps transient ranges below 64 MiB", async () => {
  let attempts = 0;
  const reader = new HttpRangeReader(
    async () => {
      attempts += 1;
      throw new TypeError("network detail");
    },
    { rangeStrategies: [2], minimumRangeAttempts: 2 },
  );

  await assert.rejects(
    reader.stream(
      {
        locator: "https://public.example/immutable/shard.bin",
        byteLength: 2,
        immutableUrl: true,
      },
      async () => {},
    ),
    (error: unknown) =>
      error instanceof RangeReadError && error.code === "range-retries-exhausted",
  );
  assert.equal(attempts, 2);

  assert.throws(
    () =>
      new HttpRangeReader(async () => new Response(), {
        rangeStrategies: [64 * 1024 * 1024],
      }),
    /below 64 MiB/i,
  );
});

test("metrics contain only structured transfer counts", async () => {
  const reader = new HttpRangeReader(
    async () => rangedResponse(new Uint8Array([1]), 0, 1),
    { rangeStrategies: [1] },
  );
  await reader.stream(
    {
      locator: "https://private.example/secret/shard.bin?token=do-not-log",
      byteLength: 1,
      immutableUrl: true,
    },
    async () => {},
  );

  const serialized = JSON.stringify(reader.metrics);
  assert.doesNotMatch(serialized, /private|secret|token|https|header|stack/i);
  assert.match(serialized, /bytes/);
});
