import assert from "node:assert/strict";
import { mkdtemp, readFile, readdir } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";
import test from "node:test";

const collectFiles = async (directory: string): Promise<string[]> => {
  const entries = await readdir(directory, { withFileTypes: true });
  const nested = await Promise.all(
    entries.map((entry) => {
      const absolute = path.join(directory, entry.name);
      return entry.isDirectory() ? collectFiles(absolute) : [absolute];
    }),
  );
  return nested.flat();
};

test("public build contains no local control agent or private connection material", async () => {
  const output = await mkdtemp(path.join(tmpdir(), "qwen-public-build-"));
  const result = spawnSync(
    process.execPath,
    ["tools/build-public.mjs", "--out-dir", output],
    { cwd: process.cwd(), encoding: "utf8" },
  );
  assert.equal(result.status, 0, `${result.stdout}\n${result.stderr}`);

  const files = await collectFiles(output);
  const content = (
    await Promise.all(files.map(async (file) => `${file}\n${await readFile(file, "utf8")}`))
  ).join("\n");
  for (const forbidden of [
    "dev/control",
    "qwen-control.v1",
    "LOCAL_CONTROL",
    "operatorToken",
    "phoneToken",
    "127.0.0.1",
    "localhost",
    "/v1/command",
    "/.local-agent.js",
  ]) {
    assert.doesNotMatch(content, new RegExp(forbidden.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")));
  }
});
