import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const BROWSER_ENTRYPOINTS = [
  "../src/qwen-tokenizer.ts",
  "../src/qwen-chat-template.ts",
];

test("browser tokenizer modules have no transitive Node imports", async () => {
  const visited = new Set<string>();
  const pending = BROWSER_ENTRYPOINTS.map(
    (path) => new URL(path, import.meta.url),
  );

  while (pending.length > 0) {
    const url = pending.pop()!;
    if (visited.has(url.href)) {
      continue;
    }
    visited.add(url.href);
    const source = await readFile(url, "utf8");
    const imports = [
      ...source.matchAll(
        /(?:from\s+|import\s*\()\s*["']([^"']+)["']/g,
      ),
    ].map((match) => match[1]!);
    for (const specifier of imports) {
      assert.equal(
        specifier.startsWith("node:"),
        false,
        `${url.pathname} imports ${specifier}`,
      );
      if (!specifier.startsWith(".")) {
        continue;
      }
      const typescriptSpecifier = specifier.replace(/\.js$/, ".ts");
      pending.push(new URL(typescriptSpecifier, url));
    }
  }
});
