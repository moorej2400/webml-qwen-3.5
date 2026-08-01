import { spawnSync } from "node:child_process";
import {
  copyFileSync,
  mkdirSync,
  readFileSync,
  readdirSync,
} from "node:fs";
import path from "node:path";

const outputFlag = process.argv.indexOf("--out-dir");
if (outputFlag < 0 || process.argv[outputFlag + 1] === undefined) {
  throw new Error("usage: build-public.mjs --out-dir <path>");
}
const outputDirectory = path.resolve(process.argv[outputFlag + 1]);
mkdirSync(outputDirectory, { recursive: true });

// The public TypeScript project has one root: the reviewed browser entrypoint.
// TypeScript follows only its transitive imports, so Node-only converters and
// unrelated development modules cannot become release artifacts.
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
if (result.status !== 0) {
  process.exitCode = result.status ?? 1;
} else {
  copyFileSync(path.resolve("public/index.html"), path.join(outputDirectory, "index.html"));
  copyFileSync(path.resolve("public/app.css"), path.join(outputDirectory, "app.css"));
  scanPublicJavaScript(outputDirectory);
}

function scanPublicJavaScript(directory) {
  const forbidden = [
    /\b(?:import|export)\b[^"']*["']node:/u,
    /\bimport\s*\(\s*["']node:/u,
    /tokenizer-compiler/u,
    /compileTokenizerSource/u,
    /dev\/control/u,
    /qwen-control\.v1/u,
    /LOCAL_CONTROL/u,
    /operatorToken/u,
    /phoneToken/u,
    /127\.0\.0\.1/u,
    /localhost/u,
    /\/v1\/command/u,
    /\/\.local-agent\.js/u,
  ];
  const pending = [directory];
  while (pending.length > 0) {
    const current = pending.pop();
    for (const entry of readdirSync(current, { withFileTypes: true })) {
      const absolute = path.join(current, entry.name);
      if (entry.isDirectory()) {
        pending.push(absolute);
        continue;
      }
      if (!entry.name.endsWith(".js") && !entry.name.endsWith(".map")) {
        continue;
      }
      const source = readFileSync(absolute, "utf8");
      for (const pattern of forbidden) {
        if (pattern.test(source)) {
          throw new Error(
            `Public artifact ${path.relative(directory, absolute)} matches ${pattern}`,
          );
        }
      }
    }
  }
}
