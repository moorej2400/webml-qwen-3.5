import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdir, mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";

import { assertLocalPackageShards } from "../dev/control/local-package.js";

const oneByteHash = createHash("sha256").update(Uint8Array.of(1)).digest("hex");

test("rejects an incomplete local language package before it reaches a browser range request", async () => {
  const directory = await mkdtemp(path.join(tmpdir(), "qwen-local-package-"));
  await mkdir(path.join(directory, "shards"));
  await writeFile(path.join(directory, "manifest.json"), "{}\n");

  await assert.rejects(
    assertLocalPackageShards({
      directory,
      shards: [{ url: "shards/model-00000.bin", length: "1", sha256: oneByteHash }],
    }),
    /Local package is incomplete/u,
  );
});

test("accepts every declared regular shard inside the local package directory", async () => {
  const directory = await mkdtemp(path.join(tmpdir(), "qwen-local-package-"));
  await mkdir(path.join(directory, "shards"));
  await writeFile(path.join(directory, "shards", "model-00000.bin"), new Uint8Array([1]));

  await assert.doesNotReject(
    assertLocalPackageShards({
      directory,
      shards: [{ url: "shards/model-00000.bin", length: "1", sha256: oneByteHash }],
    }),
  );
});

test("rejects a truncated local shard before an expensive browser load", async () => {
  const directory = await mkdtemp(path.join(tmpdir(), "qwen-local-package-"));
  await mkdir(path.join(directory, "shards"));
  await writeFile(path.join(directory, "shards", "model-00000.bin"), new Uint8Array([1]));

  await assert.rejects(
    assertLocalPackageShards({
      directory,
      shards: [{ url: "shards/model-00000.bin", length: "2", sha256: oneByteHash }],
    }),
    /Local package is incomplete/u,
  );
});

test("rejects a same-length corrupted local shard before browser startup", async () => {
  const directory = await mkdtemp(path.join(tmpdir(), "qwen-local-package-"));
  await mkdir(path.join(directory, "shards"));
  await writeFile(path.join(directory, "shards", "model-00000.bin"), Uint8Array.of(2));

  await assert.rejects(
    assertLocalPackageShards({
      directory,
      shards: [{ url: "shards/model-00000.bin", length: "1", sha256: oneByteHash }],
    }),
    /Local package is incomplete/u,
  );
});
