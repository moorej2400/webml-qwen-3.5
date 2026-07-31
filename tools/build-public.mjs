import { spawnSync } from "node:child_process";
import { mkdirSync } from "node:fs";
import path from "node:path";

const outputFlag = process.argv.indexOf("--out-dir");
if (outputFlag < 0 || process.argv[outputFlag + 1] === undefined) {
  throw new Error("usage: build-public.mjs --out-dir <path>");
}
const outputDirectory = path.resolve(process.argv[outputFlag + 1]);
mkdirSync(outputDirectory, { recursive: true });

// Public compilation starts at `src/`; development control code is outside the
// TypeScript root and therefore cannot enter a Pages artifact by reachability.
const result = spawnSync(
  process.execPath,
  [
    path.resolve("node_modules/typescript/bin/tsc"),
    "--project",
    "tsconfig.public.json",
    "--outDir",
    outputDirectory,
  ],
  { stdio: "inherit" },
);
process.exitCode = result.status ?? 1;
