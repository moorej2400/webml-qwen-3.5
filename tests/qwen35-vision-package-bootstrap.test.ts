import assert from "node:assert/strict";
import test from "node:test";

import {
  QWEN35_PRODUCTION_VISION_PACKAGE,
  loadProductionQwen35VisionPackage,
} from "../src/qwen35-vision-package-bootstrap.js";

const REVISION = "c03d9b750d8ff5ba7e48effcdc2ff16db2ad7153";
const BASE_URL = `https://huggingface.co/moorejared97/Qwen3.5-4B-Q3-K-L-WebGPU/resolve/${REVISION}/`;

function response(
  body: BodyInit | null,
  options: {
    readonly status?: number;
    readonly contentLength?: string;
    readonly finalUrl?: string;
    readonly redirected?: boolean;
  } = {},
): Response {
  const headers = new Headers();
  if (options.contentLength !== undefined) {
    headers.set("content-length", options.contentLength);
  }
  const result = new Response(body, { status: options.status ?? 200, headers });
  Object.defineProperty(result, "url", { value: options.finalUrl ?? "" });
  Object.defineProperty(result, "redirected", { value: options.redirected ?? false });
  return result;
}

test("pins the production vision repository and immutable metadata identities", () => {
  assert.deepEqual(QWEN35_PRODUCTION_VISION_PACKAGE, {
    repository: "moorejared97/Qwen3.5-4B-Q3-K-L-WebGPU",
    revision: REVISION,
    baseUrl: BASE_URL,
    manifest: {
      file: "vision-manifest.json",
      sha256: "7323d564ea10a1305db61a4af9c226eef7288202e8a97b5b33ec9be136dbf9a5",
    },
    layerIndex: {
      file: "vision-layer-index.json",
      sha256: "e71fa47629be790db5a9afb247901cea411a95d27e8dbad4029270a7cd61ded2",
    },
  });
  assert.ok(Object.isFrozen(QWEN35_PRODUCTION_VISION_PACKAGE));
  assert.ok(Object.isFrozen(QWEN35_PRODUCTION_VISION_PACKAGE.manifest));
});

test("fetches only the two fixed metadata files and rejects tampered bytes", async () => {
  const requests: string[] = [];
  const fetchImplementation: typeof fetch = async (input) => {
    requests.push(String(input));
    return response(Uint8Array.of(1, 2, 3), {
      contentLength: "3",
      finalUrl: String(input),
    });
  };
  await assert.rejects(loadProductionQwen35VisionPackage({ fetchImplementation }), {
    code: "vision-package-manifest-hash-mismatch",
  });
  assert.deepEqual(requests.sort(), [
    `${BASE_URL}vision-layer-index.json`,
    `${BASE_URL}vision-manifest.json`,
  ]);
});

test("rejects oversized metadata before reading the body", async () => {
  let bodyCancelled = 0;
  const fetchImplementation: typeof fetch = async (input) => response(
    new ReadableStream<Uint8Array>({
      cancel() {
        bodyCancelled += 1;
      },
    }),
    {
      contentLength: String((1024 * 1024) + 1),
      finalUrl: String(input),
    },
  );
  await assert.rejects(loadProductionQwen35VisionPackage({ fetchImplementation }), {
    code: "vision-metadata-size-invalid",
  });
  assert.equal(bodyCancelled, 2);
});

test("propagates cancellation without issuing metadata requests", async () => {
  const controller = new AbortController();
  controller.abort();
  let requests = 0;
  const fetchImplementation: typeof fetch = async () => {
    requests += 1;
    return response(Uint8Array.of(1));
  };
  await assert.rejects(loadProductionQwen35VisionPackage({
    fetchImplementation,
    signal: controller.signal,
  }), { name: "AbortError" });
  assert.equal(requests, 0);
});

test("cancellation after metadata requests start does not construct or return a package", async () => {
  const controller = new AbortController();
  let requests = 0;
  const fetchImplementation: typeof fetch = async () => {
    requests += 1;
    if (requests === 2) {
      controller.abort(new DOMException("cancelled", "AbortError"));
    }
    return response(Uint8Array.of(1), { contentLength: "1" });
  };
  await assert.rejects(loadProductionQwen35VisionPackage({
    fetchImplementation,
    signal: controller.signal,
  }), { name: "AbortError" });
  assert.equal(requests, 2);
});

test("fails safely on metadata HTTP errors", async () => {
  const fetchImplementation: typeof fetch = async (input) => response(null, {
    status: 503,
    finalUrl: String(input),
  });
  await assert.rejects(loadProductionQwen35VisionPackage({ fetchImplementation }), {
    code: "vision-metadata-http-invalid",
  });
});

test("uses pinned hashes instead of a redirected response origin", async () => {
  const cdnFetch: typeof fetch = async (input) => response(Uint8Array.of(1), {
    contentLength: "1",
    finalUrl: `https://cdn-lfs.hf.co/public/${String(input).endsWith("manifest.json") ? "manifest" : "index"}`,
    redirected: true,
  });
  await assert.rejects(loadProductionQwen35VisionPackage({ fetchImplementation: cdnFetch }), {
    code: "vision-package-manifest-hash-mismatch",
  });

  const redirectedFetch: typeof fetch = async () => response(Uint8Array.of(1), {
    contentLength: "1",
    finalUrl: "https://downloads.invalid/vision-metadata.json",
    redirected: true,
  });
  await assert.rejects(loadProductionQwen35VisionPackage({ fetchImplementation: redirectedFetch }), {
    code: "vision-package-manifest-hash-mismatch",
  });
});
