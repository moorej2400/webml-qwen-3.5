import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { builtinModules } from "node:module";
import test from "node:test";
import ts from "typescript";
import * as browser from "../src/browser.js";

const BROWSER_ENTRYPOINTS = [
  "../src/browser.ts",
];
const NODE_BUILTINS = new Set(
  builtinModules.flatMap((name) => [name, `node:${name}`]),
);

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
    const sourceFile = ts.createSourceFile(
      url.pathname,
      source,
      ts.ScriptTarget.ESNext,
      true,
      ts.ScriptKind.TS,
    );
    const imports: string[] = [];
    const visit = (node: ts.Node): void => {
      if (
        (ts.isImportDeclaration(node) || ts.isExportDeclaration(node)) &&
        node.moduleSpecifier !== undefined &&
        ts.isStringLiteral(node.moduleSpecifier)
      ) {
        imports.push(node.moduleSpecifier.text);
      }
      if (
        ts.isCallExpression(node) &&
        node.expression.kind === ts.SyntaxKind.ImportKeyword &&
        node.arguments.length === 1 &&
        ts.isStringLiteral(node.arguments[0]!)
      ) {
        imports.push(node.arguments[0].text);
      }
      ts.forEachChild(node, visit);
    };
    visit(sourceFile);
    for (const specifier of imports) {
      assert.equal(
        NODE_BUILTINS.has(specifier),
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

test("the browser entrypoint exports authenticated runtime APIs but not Node compiler APIs", () => {
  assert.equal(typeof browser.Qwen35Tokenizer, "function");
  assert.equal(typeof browser.loadPinnedQwen35Tokenizer, "function");
  assert.equal(typeof browser.renderQwen35Chat, "function");
  assert.equal(typeof browser.Qwen35Session, "function");
  assert.equal(typeof browser.createQwen35ActivationWorkspace, "function");
  assert.equal(typeof browser.planQwen35PackedEmbeddingDispatch, "function");
  assert.equal(typeof browser.planQwen35TiedLogitsDispatches, "function");
  assert.equal(typeof browser.planQwen35LogitsTileWinner, "function");
  assert.equal(typeof browser.planQwen35FinalTokenSelection, "function");
  assert.equal(typeof browser.createQwen35UniformArena, "function");
  assert.equal(typeof browser.planQwen35DeltaNetLayerDispatch, "function");
  assert.equal(typeof browser.assembleQwen35TiledLogitsCommands, "function");
  assert.equal(typeof browser.planQwen35FullAttentionLayerGeometry, "function");
  assert.equal(typeof browser.planQwen35FullAttentionLayerDispatch, "function");
  assert.equal(typeof browser.planQwen35FinalNormDispatch, "function");
  assert.equal(typeof browser.createQwen35AllocationClearer, "function");
  assert.equal(typeof browser.planQwen35VisionImage, "function");
  assert.equal(typeof browser.packQwen35VisionRgb, "function");
  assert.equal(typeof browser.resizeQwen35VisionRgbBicubic, "function");
  assert.equal(typeof browser.preprocessQwen35VisionRgb, "function");
  assert.equal(typeof browser.loadProductionQwen35VisionPackage, "function");
  assert.equal(typeof browser.createQwen35VisionProgram, "function");
  assert.equal("createIntegrityValidatedQwen35VisionPackage" in browser, false);
  assert.equal("createProductionQwen35VisionPackage" in browser, false);
  assert.equal("streamQwen35CachedWeights" in browser, false);
  assert.equal("compileTokenizerSource" in browser, false);
  assert.equal("compileQwen35TokenizerSource" in browser, false);
});
