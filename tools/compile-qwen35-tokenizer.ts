import { createReadStream } from "node:fs";
import { mkdir, writeFile } from "node:fs/promises";
import { resolve, join } from "node:path";
import { pathToFileURL } from "node:url";
import { compileQwen35TokenizerSource } from "../src/tokenizer-compiler.js";

export interface TokenizerCompilerArguments {
  readonly tokenizerPath: string;
  readonly configPath: string;
  readonly outputDirectory: string;
}

export function parseTokenizerCompilerArguments(
  arguments_: readonly string[],
): TokenizerCompilerArguments {
  const values = new Map<string, string>();
  for (let index = 0; index < arguments_.length; index += 2) {
    const name = arguments_[index];
    const value = arguments_[index + 1];
    if (
      name === undefined ||
      value === undefined ||
      !["--tokenizer", "--config", "--output"].includes(name) ||
      values.has(name)
    ) {
      throw new Error("Tokenizer compiler received an unknown argument");
    }
    values.set(name, value);
  }
  const tokenizerPath = values.get("--tokenizer");
  const configPath = values.get("--config");
  const outputDirectory = values.get("--output");
  if (
    tokenizerPath === undefined ||
    configPath === undefined ||
    outputDirectory === undefined
  ) {
    throw new Error(
      "Tokenizer compiler requires --tokenizer, --config, and --output",
    );
  }
  return Object.freeze({ tokenizerPath, configPath, outputDirectory });
}

/**
 * Writes only verified, browser-native artifacts; source JSON remains outside
 * Git and is never copied into the output package.
 */
export async function runTokenizerCompiler(
  options: TokenizerCompilerArguments,
): Promise<void> {
  const compiled = await compileQwen35TokenizerSource(
    createReadStream(options.tokenizerPath),
    createReadStream(options.configPath),
  );
  await mkdir(options.outputDirectory, { recursive: true });
  await writeFile(
    join(options.outputDirectory, "tokenizer.bin"),
    compiled.binary,
    { flag: "wx" },
  );
  await writeFile(
    join(options.outputDirectory, "tokenizer.manifest.json"),
    `${JSON.stringify(compiled.manifest, null, 2)}\n`,
    { encoding: "utf8", flag: "wx" },
  );
}

const invokedPath = process.argv[1];
if (
  invokedPath !== undefined &&
  import.meta.url === pathToFileURL(resolve(invokedPath)).href
) {
  await runTokenizerCompiler(
    parseTokenizerCompilerArguments(process.argv.slice(2)),
  );
}
