import assert from "node:assert/strict";
import test from "node:test";

import { IncrementalSha256 } from "../src/incremental-sha256.js";

const encoder = new TextEncoder();

test("matches the SHA-256 empty and abc reference vectors", () => {
  assert.equal(
    new IncrementalSha256().digestHex(),
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
  );

  const hash = new IncrementalSha256();
  hash.update(encoder.encode("abc"));
  assert.equal(
    hash.digestHex(),
    "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
  );
});

test("produces the same digest across arbitrary streaming chunk boundaries", () => {
  const input = encoder.encode(
    "The quick brown fox jumps over the lazy dog".repeat(10_000),
  );
  const whole = new IncrementalSha256();
  whole.update(input);

  const streamed = new IncrementalSha256();
  let offset = 0;
  const widths = [1, 3, 17, 64, 511, 4_097];
  let widthIndex = 0;
  while (offset < input.byteLength) {
    const end = Math.min(input.byteLength, offset + widths[widthIndex]!);
    streamed.update(input.subarray(offset, end));
    offset = end;
    widthIndex = (widthIndex + 1) % widths.length;
  }

  assert.equal(streamed.digestHex(), whole.digestHex());
});

test("rejects updates and repeated finalization after digest", () => {
  const hash = new IncrementalSha256();
  hash.digestHex();

  assert.throws(() => hash.update(new Uint8Array([1])), /finalized/i);
  assert.throws(() => hash.digestHex(), /finalized/i);
});
