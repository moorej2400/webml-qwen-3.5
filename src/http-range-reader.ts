export const DEFAULT_RANGE_BYTES = 32 * 1024 * 1024;
export const RANGE_PIVOT_BYTES = [
  DEFAULT_RANGE_BYTES,
  16 * 1024 * 1024,
  8 * 1024 * 1024,
] as const;
export const MAX_TRANSIENT_WEIGHT_BYTES = 64 * 1024 * 1024;

export interface RangeFetchInit {
  range: string;
  signal: AbortSignal;
}

export interface RangeFetchResponse {
  status: number;
  headers: {
    get(name: string): string | null;
  };
  body: ReadableStream<Uint8Array> | null;
}

export type RangeFetch = (
  locator: string,
  init: RangeFetchInit,
) => Promise<RangeFetchResponse>;

export interface ImmutableRangeSource {
  /** Kept inside the transport boundary and never copied into metrics. */
  locator: string;
  byteLength: number;
  immutableUrl: boolean;
  expectedEtag?: string;
}

export interface RangeChunkMetadata {
  absoluteOffset: number;
  rangeOffset: number;
  rangeLength: number;
}

export type RangeChunkConsumer = (
  chunk: Uint8Array,
  metadata: RangeChunkMetadata,
) => Promise<void>;

export type RangeMetricEvent =
  | {
      kind: "range-complete";
      bytes: number;
      strategyBytes: number;
    }
  | {
      kind: "range-pivot";
      fromBytes: number;
      toBytes: number;
    }
  | {
      kind: "range-retry";
      strategyBytes: number;
      attempt: number;
    };

export interface RangeReaderMetrics {
  bytes: number;
  ranges: number;
  retries: number;
  pivots: number;
  peakTransientBytes: number;
  events: readonly RangeMetricEvent[];
}

export interface HttpRangeReaderOptions {
  rangeStrategies?: readonly number[];
  minimumRangeAttempts?: number;
}

export class RangeReadError extends Error {
  constructor(readonly code: string) {
    super(code);
    this.name = "RangeReadError";
  }
}

class RetryableRangeFailure extends Error {}

interface BufferedRange {
  chunks: Uint8Array[];
  byteLength: number;
  etag?: string;
}

/**
 * Reads immutable package bytes without whole-shard buffering.
 *
 * Each validated HTTP range is buffered only until its protocol fields and
 * byte count are known. Chunks are then transferred to one awaited consumer
 * at a time, so OPFS writes and GPU uploads cannot build an unbounded overlap.
 */
export class HttpRangeReader {
  private readonly strategies: readonly number[];
  private readonly minimumRangeAttempts: number;
  private readonly metricEvents: RangeMetricEvent[] = [];
  private byteCount = 0;
  private rangeCount = 0;
  private retryCount = 0;
  private pivotCount = 0;
  private peakTransientBytes = 0;

  constructor(
    private readonly fetchRange: RangeFetch,
    options: HttpRangeReaderOptions = {},
  ) {
    this.strategies = Object.freeze([
      ...(options.rangeStrategies ?? RANGE_PIVOT_BYTES),
    ]);
    this.minimumRangeAttempts = options.minimumRangeAttempts ?? 2;
    this.validateOptions();
  }

  get metrics(): RangeReaderMetrics {
    return {
      bytes: this.byteCount,
      ranges: this.rangeCount,
      retries: this.retryCount,
      pivots: this.pivotCount,
      peakTransientBytes: this.peakTransientBytes,
      events: this.metricEvents.map((event) => ({ ...event })),
    };
  }

  async stream(
    source: ImmutableRangeSource,
    consume: RangeChunkConsumer,
    signal: AbortSignal = new AbortController().signal,
  ): Promise<void> {
    this.validateSource(source);
    let position = 0;
    let strategyIndex = 0;
    let minimumAttempts = 0;
    let observedEtag = source.expectedEtag;

    while (position < source.byteLength) {
      signal.throwIfAborted();
      const strategyBytes = this.strategies[strategyIndex]!;
      const rangeLength = Math.min(strategyBytes, source.byteLength - position);
      let buffered: BufferedRange;
      try {
        buffered = await this.readRange(
          source,
          position,
          rangeLength,
          observedEtag,
          signal,
        );
      } catch (error) {
        if (signal.aborted) {
          signal.throwIfAborted();
        }
        if (!(error instanceof RetryableRangeFailure)) {
          throw error;
        }
        if (strategyIndex < this.strategies.length - 1) {
          const fromBytes = strategyBytes;
          strategyIndex += 1;
          const toBytes = this.strategies[strategyIndex]!;
          this.pivotCount += 1;
          this.metricEvents.push({
            kind: "range-pivot",
            fromBytes,
            toBytes,
          });
          continue;
        }

        minimumAttempts += 1;
        if (minimumAttempts >= this.minimumRangeAttempts) {
          throw new RangeReadError("range-retries-exhausted");
        }
        this.retryCount += 1;
        this.metricEvents.push({
          kind: "range-retry",
          strategyBytes,
          attempt: minimumAttempts + 1,
        });
        continue;
      }

      minimumAttempts = 0;
      observedEtag ??= buffered.etag;
      let rangeOffset = 0;
      for (let index = 0; index < buffered.chunks.length; index += 1) {
        signal.throwIfAborted();
        const chunk = buffered.chunks[index]!;
        // Remove the reader's reference before the awaited handoff. The
        // consumer owns this chunk until its promise settles.
        buffered.chunks[index] = new Uint8Array(0);
        await consume(chunk, {
          absoluteOffset: position + rangeOffset,
          rangeOffset,
          rangeLength: buffered.byteLength,
        });
        rangeOffset += chunk.byteLength;
      }

      position += buffered.byteLength;
      this.byteCount += buffered.byteLength;
      this.rangeCount += 1;
      this.metricEvents.push({
        kind: "range-complete",
        bytes: buffered.byteLength,
        strategyBytes,
      });
    }
  }

  private async readRange(
    source: ImmutableRangeSource,
    start: number,
    length: number,
    observedEtag: string | undefined,
    signal: AbortSignal,
  ): Promise<BufferedRange> {
    const end = start + length - 1;
    let response: RangeFetchResponse;
    try {
      response = await this.fetchRange(source.locator, {
        range: `bytes=${start}-${end}`,
        signal,
      });
    } catch (error) {
      if (signal.aborted || isAbortError(error)) {
        signal.throwIfAborted();
        throw error;
      }
      throw new RetryableRangeFailure();
    }

    if (response.status !== 206) {
      await cancelResponseBody(response.body);
      if (
        response.status === 408 ||
        response.status === 429 ||
        (response.status >= 500 && response.status <= 599)
      ) {
        throw new RetryableRangeFailure();
      }
      throw new RangeReadError("range-status-invalid");
    }

    try {
      this.validateResponseHeaders(
        response,
        source,
        start,
        end,
        length,
        observedEtag,
      );
    } catch (error) {
      await cancelResponseBody(response.body);
      throw error;
    }
    if (response.body === null) {
      throw new RetryableRangeFailure();
    }

    const chunks: Uint8Array[] = [];
    let received = 0;
    const reader = response.body.getReader();
    try {
      while (true) {
        signal.throwIfAborted();
        const result = await reader.read();
        if (result.done) {
          break;
        }
        const chunk = result.value;
        if (received + chunk.byteLength > length) {
          throw new RangeReadError("range-body-too-large");
        }
        chunks.push(chunk);
        received += chunk.byteLength;
        this.peakTransientBytes = Math.max(
          this.peakTransientBytes,
          received,
        );
      }
    } catch (error) {
      try {
        await reader.cancel();
      } catch {
        // The original protocol, network, or cancellation failure is primary.
      }
      if (error instanceof RangeReadError || signal.aborted) {
        throw error;
      }
      throw new RetryableRangeFailure();
    } finally {
      reader.releaseLock();
    }

    if (received !== length) {
      throw new RetryableRangeFailure();
    }
    const etag = response.headers.get("etag") ?? undefined;
    return etag === undefined
      ? { chunks, byteLength: received }
      : { chunks, byteLength: received, etag };
  }

  private validateResponseHeaders(
    response: RangeFetchResponse,
    source: ImmutableRangeSource,
    start: number,
    end: number,
    length: number,
    observedEtag: string | undefined,
  ): void {
    const contentLength = response.headers.get("content-length");
    if (contentLength !== String(length)) {
      throw new RangeReadError("content-length-invalid");
    }

    const contentRange = response.headers.get("content-range");
    const match =
      contentRange === null
        ? null
        : /^bytes ([0-9]+)-([0-9]+)\/([0-9]+)$/.exec(contentRange);
    if (
      match === null ||
      Number(match[1]) !== start ||
      Number(match[2]) !== end ||
      Number(match[3]) !== source.byteLength
    ) {
      throw new RangeReadError("content-range-invalid");
    }

    const etag = response.headers.get("etag");
    if (!source.immutableUrl && observedEtag === undefined && etag === null) {
      throw new RangeReadError("etag-required");
    }
    if (observedEtag !== undefined && etag !== observedEtag) {
      throw new RangeReadError("etag-changed");
    }
  }

  private validateSource(source: ImmutableRangeSource): void {
    if (
      source.locator.length === 0 ||
      !Number.isSafeInteger(source.byteLength) ||
      source.byteLength <= 0
    ) {
      throw new RangeReadError("range-source-invalid");
    }
  }

  private validateOptions(): void {
    if (
      this.strategies.length === 0 ||
      !Number.isSafeInteger(this.minimumRangeAttempts) ||
      this.minimumRangeAttempts < 1
    ) {
      throw new Error("Range reader strategies and retry count are invalid");
    }
    let previous = Number.POSITIVE_INFINITY;
    for (const bytes of this.strategies) {
      if (
        !Number.isSafeInteger(bytes) ||
        bytes <= 0 ||
        bytes >= MAX_TRANSIENT_WEIGHT_BYTES
      ) {
        throw new Error("A range strategy must remain below 64 MiB");
      }
      if (bytes > previous) {
        throw new Error("Range strategies must be ordered from large to small");
      }
      previous = bytes;
    }
  }
}

async function cancelResponseBody(
  body: ReadableStream<Uint8Array> | null,
): Promise<void> {
  try {
    await body?.cancel();
  } catch {
    // Protocol rejection must not be replaced by a secondary cancel failure.
  }
}

function isAbortError(error: unknown): boolean {
  return (
    error instanceof DOMException
      ? error.name === "AbortError"
      : error instanceof Error && error.name === "AbortError"
  );
}

export function browserRangeFetch(
  fetchImplementation: typeof fetch = fetch,
): RangeFetch {
  return async (locator, init) =>
    fetchImplementation(locator, {
      method: "GET",
      headers: { Range: init.range },
      cache: "no-store",
      signal: init.signal,
    });
}
