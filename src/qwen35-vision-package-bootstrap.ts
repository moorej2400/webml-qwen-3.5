import { diagnosticError } from "./diagnostics.js";
import { browserRangeFetch } from "./http-range-reader.js";
import {
  createIntegrityValidatedQwen35VisionPackage,
  createProductionQwen35VisionPackage,
  QWEN35_PRODUCTION_VISION_PACKAGE_PINS,
  type Qwen35IntegrityValidatedVisionPackage,
  type Qwen35VisionPackagePins,
  type Qwen35ProductionVisionPackage,
} from "./qwen35-vision-package-loader.js";

const MAX_METADATA_BYTES = 1024 * 1024;
const POSITIVE_DECIMAL = /^[1-9][0-9]*$/u;
const SAFE_METADATA_FILE = /^[A-Za-z0-9._-]+$/u;

export const QWEN35_PRODUCTION_VISION_PACKAGE = Object.freeze({
  repository: "moorejared97/Qwen3.5-4B-Q3-K-L-WebGPU",
  revision: "c03d9b750d8ff5ba7e48effcdc2ff16db2ad7153",
  baseUrl: QWEN35_PRODUCTION_VISION_PACKAGE_PINS.packageBaseUrl,
  manifest: Object.freeze({
    file: "vision-manifest.json",
    sha256: QWEN35_PRODUCTION_VISION_PACKAGE_PINS.expectedManifestSha256,
  }),
  layerIndex: Object.freeze({
    file: "vision-layer-index.json",
    sha256: QWEN35_PRODUCTION_VISION_PACKAGE_PINS.expectedLayerIndexSha256,
  }),
});

export interface LoadProductionQwen35VisionPackageOptions {
  /** Allows browser-compatible tests to supply a deterministic transport. */
  readonly fetchImplementation?: typeof fetch;
  readonly signal?: AbortSignal;
  /** Explicit local Chrome smoke-test pins; omitted for the fixed public release. */
  readonly pins?: Qwen35VisionPackagePins;
}

function fail(code: string, message: string): never {
  throw diagnosticError(code, message);
}

async function cancelBody(body: ReadableStream<Uint8Array> | null): Promise<void> {
  try {
    await body?.cancel();
  } catch {
    // Rejecting a response must keep the HTTP or size diagnostic as primary.
  }
}

function declaredByteLength(response: Response): number | undefined {
  const contentLength = response.headers.get("content-length");
  if (contentLength === null) {
    return undefined;
  }
  if (!POSITIVE_DECIMAL.test(contentLength)) {
    fail("vision-metadata-size-invalid", "Vision package metadata size is invalid");
  }
  const length = Number(contentLength);
  if (!Number.isSafeInteger(length) || length > MAX_METADATA_BYTES) {
    fail("vision-metadata-size-invalid", "Vision package metadata size is invalid");
  }
  return length;
}

function metadataFile(value: string | undefined, fallback: string): string {
  const selected = value ?? fallback;
  if (!SAFE_METADATA_FILE.test(selected)) {
    fail("vision-metadata-file-invalid", "Vision package metadata filename is invalid");
  }
  return selected;
}

/**
 * Reads only bounded release metadata; shard bytes remain in the loader's
 * range-streaming path so package bootstrap cannot create a model-sized copy.
 */
async function fetchMetadata(
  url: string,
  fetchImplementation: typeof fetch,
  signal: AbortSignal,
): Promise<Uint8Array> {
  signal.throwIfAborted();
  let response: Response;
  try {
    response = await fetchImplementation(url, {
      method: "GET",
      cache: "no-store",
      signal,
    });
  } catch (error) {
    if (signal.aborted) {
      signal.throwIfAborted();
    }
    throw error;
  }
  if (!response.ok) {
    await cancelBody(response.body);
    fail("vision-metadata-http-invalid", "Vision package metadata request failed");
  }

  let expectedLength: number | undefined;
  try {
    expectedLength = declaredByteLength(response);
  } catch (error) {
    await cancelBody(response.body);
    throw error;
  }
  if (response.body === null) {
    fail("vision-metadata-body-invalid", "Vision package metadata response has no body");
  }

  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let received = 0;
  try {
    while (true) {
      signal.throwIfAborted();
      const result = await reader.read();
      if (result.done) {
        break;
      }
      if (
        !(result.value instanceof Uint8Array) ||
        result.value.byteLength === 0 ||
        received + result.value.byteLength > MAX_METADATA_BYTES
      ) {
        fail("vision-metadata-size-invalid", "Vision package metadata size is invalid");
      }
      chunks.push(result.value);
      received += result.value.byteLength;
    }
  } catch (error) {
    try {
      await reader.cancel();
    } catch {
      // A fetch, decode, or cancellation failure remains the primary error.
    }
    if (signal.aborted) {
      signal.throwIfAborted();
    }
    throw error;
  } finally {
    reader.releaseLock();
  }
  if (received === 0 || received !== expectedLength && expectedLength !== undefined) {
    fail("vision-metadata-size-invalid", "Vision package metadata size is invalid");
  }

  const bytes = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return bytes;
}

/**
 * Loads the fixed public release metadata, then delegates both hash checks and
 * streamed-shard authentication to the package loader. Redirect destinations
 * are intentionally not allowlisted: immutable request paths plus SHA-256
 * pins authenticate the bytes while permitting Hugging Face CDN delivery.
 */
export async function loadProductionQwen35VisionPackage(
  options: LoadProductionQwen35VisionPackageOptions = {},
): Promise<Qwen35ProductionVisionPackage | Qwen35IntegrityValidatedVisionPackage> {
  const fetchImplementation = options.fetchImplementation ?? fetch;
  const upstreamSignal = options.signal ?? new AbortController().signal;
  upstreamSignal.throwIfAborted();
  const controller = new AbortController();
  const relayAbort = (): void => controller.abort(upstreamSignal.reason);
  upstreamSignal.addEventListener("abort", relayAbort, { once: true });
  const packageBaseUrl = options.pins?.packageBaseUrl ?? QWEN35_PRODUCTION_VISION_PACKAGE.baseUrl;
  const fetchOne = async (file: string): Promise<Uint8Array> => {
    try {
      return await fetchMetadata(
        `${packageBaseUrl}${file}`,
        fetchImplementation,
        controller.signal,
      );
    } catch (error) {
      if (!controller.signal.aborted) {
        controller.abort(error);
      }
      throw error;
    }
  };
  try {
    const manifestFile = metadataFile(options.pins?.manifestFile, QWEN35_PRODUCTION_VISION_PACKAGE.manifest.file);
    const layerIndexFile = metadataFile(options.pins?.layerIndexFile, QWEN35_PRODUCTION_VISION_PACKAGE.layerIndex.file);
    const [manifestResult, layerIndexResult] = await Promise.allSettled([
      fetchOne(manifestFile),
      fetchOne(layerIndexFile),
    ]);
    // Let the sibling finish body cancellation before returning the first
    // metadata failure; otherwise its bounded response can outlive bootstrap.
    if (manifestResult.status === "rejected") {
      throw manifestResult.reason;
    }
    if (layerIndexResult.status === "rejected") {
      throw layerIndexResult.reason;
    }
    const manifestBytes = manifestResult.value;
    const layerIndexBytes = layerIndexResult.value;
    // Both fetches can finish between their last reader check and this point.
    // Do not construct a trusted package after a late caller cancellation.
    controller.signal.throwIfAborted();
    const package_ = options.pins === undefined
      ? createProductionQwen35VisionPackage({
          manifestBytes,
          layerIndexBytes,
          rangeFetch: browserRangeFetch(fetchImplementation),
        })
      : createIntegrityValidatedQwen35VisionPackage({
          manifestBytes,
          layerIndexBytes,
          pins: options.pins,
          rangeFetch: browserRangeFetch(fetchImplementation),
        });
    // Construction is synchronous today, but keep this boundary explicit if
    // future construction obtains asynchronous resources before return.
    controller.signal.throwIfAborted();
    return package_;
  } finally {
    upstreamSignal.removeEventListener("abort", relayAbort);
  }
}
