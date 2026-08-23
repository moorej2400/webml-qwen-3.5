import { createReadStream } from "node:fs";
import { lstat, realpath } from "node:fs/promises";
import path from "node:path";
import { createHash } from "node:crypto";

const incomplete = (): never => {
  throw new Error("Local package is incomplete");
};

const resolveShardPath = (packageDirectory: string, shardUrl: string): string => {
  if (
    shardUrl.length === 0 ||
    shardUrl.startsWith("/") ||
    shardUrl.includes("\\") ||
    shardUrl.split("/").some((part) => part.length === 0 || part === "." || part === "..")
  ) {
    return incomplete();
  }
  return path.resolve(packageDirectory, shardUrl);
};

/**
 * A local package URL is served directly by the development HTTPS server.
 * Verify every declared shard before startup so a bad local path cannot turn
 * into a delayed browser-only 404 during an expensive phone load.
 */
export const assertLocalPackageShards = async (input: {
  readonly directory: string;
  readonly shards: readonly {
    readonly url: string;
    readonly length: string;
    readonly sha256: string;
  }[];
}): Promise<void> => {
  let packageRoot: string | undefined;
  try {
    const facts = await lstat(input.directory);
    if (!facts.isDirectory() || facts.isSymbolicLink()) incomplete();
    packageRoot = await realpath(input.directory);
  } catch {
    incomplete();
  }
  const resolvedPackageRoot = packageRoot ?? incomplete();
  const packagePrefix = `${resolvedPackageRoot}${path.sep}`;

  for (const shard of input.shards) {
    if (!/^(?:0|[1-9][0-9]*)$/.test(shard.length)) incomplete();
    if (!/^[a-f0-9]{64}$/.test(shard.sha256)) incomplete();
    const expectedBytes = BigInt(shard.length);
    if (expectedBytes > BigInt(Number.MAX_SAFE_INTEGER)) incomplete();
    const candidate = resolveShardPath(resolvedPackageRoot, shard.url);
    if (!candidate.startsWith(packagePrefix)) incomplete();
    try {
      const facts = await lstat(candidate);
      if (!facts.isFile() || facts.isSymbolicLink()) incomplete();
      if (BigInt(facts.size) !== expectedBytes) incomplete();
      if (!(await realpath(candidate)).startsWith(packagePrefix)) incomplete();
      const hash = createHash("sha256");
      for await (const chunk of createReadStream(candidate)) hash.update(chunk);
      if (hash.digest("hex") !== shard.sha256) incomplete();
    } catch {
      incomplete();
    }
  }
};
