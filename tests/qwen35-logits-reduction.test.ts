import assert from "node:assert/strict";
import test from "node:test";

import {
  QWEN35_LOGITS_REDUCTION_KERNELS,
  QWEN35_NO_SELECTED_TOKEN,
  planQwen35FinalTokenSelection,
  planQwen35LogitsTileWinner,
  selectQwen35FiniteTileWinner,
  selectQwen35IndexedWinner,
} from "../src/qwen35-logits-reduction.js";

test("selects one finite winner per logical tile with stable vocabulary ties", () => {
  assert.deepEqual(
    selectQwen35FiniteTileWinner(
      Float32Array.of(Number.NaN, 7, 7, Number.POSITIVE_INFINITY, -2),
      1_024,
    ),
    { score: 7, tokenId: 1_025 },
  );
  assert.equal(
    selectQwen35FiniteTileWinner(
      Float32Array.of(Number.NaN, Number.NEGATIVE_INFINITY),
      0,
    ),
    null,
  );
});

test("reduces indexed tile winners without confusing slots for token ids", () => {
  assert.deepEqual(
    selectQwen35IndexedWinner(
      Float32Array.of(3, 9, 9, 100),
      Uint32Array.of(70_000, 125_952, 4, QWEN35_NO_SELECTED_TOKEN),
      4,
    ),
    { score: 9, tokenId: 4 },
  );
  assert.equal(
    selectQwen35IndexedWinner(
      Float32Array.of(Number.NaN, 3),
      Uint32Array.of(8, QWEN35_NO_SELECTED_TOKEN),
      2,
    ),
    null,
  );
});

test("plans exactly one candidate per mathematical tile and one final dispatch", () => {
  assert.deepEqual(
    planQwen35LogitsTileWinner({
      vocabularyStart: 125_952,
      vocabularyRows: 1_024,
      candidateSlot: 123,
    }),
    {
      operation: "logits-tile-top-1",
      uniformWords: [1_024, 125_952, 123, 0],
      workgroups: { x: 1, y: 1, z: 1 },
    },
  );
  assert.deepEqual(planQwen35FinalTokenSelection({ candidateCount: 243 }), {
    operation: "indexed-top-1",
    uniformWords: [243, 0, 0, 0],
    workgroups: { x: 1, y: 1, z: 1 },
  });
  assert.throws(
    () => planQwen35LogitsTileWinner({
      vocabularyStart: 248_000,
      vocabularyRows: 321,
      candidateSlot: 242,
    }),
    /tile/i,
  );
  assert.throws(
    () => planQwen35FinalTokenSelection({ candidateCount: 257 }),
    /candidate/i,
  );
  assert.throws(
    () => planQwen35FinalTokenSelection({ candidateCount: 242 }),
    /candidate/i,
  );
  assert.throws(
    () => planQwen35LogitsTileWinner({
      vocabularyStart: 1_500,
      vocabularyRows: 700,
      candidateSlot: 1,
    }),
    /mathematical/i,
  );
  assert.deepEqual(
    planQwen35LogitsTileWinner({
      vocabularyStart: 247_808,
      vocabularyRows: 262,
      candidateSlot: 242,
    }).uniformWords,
    [262, 247_808, 242, 0],
  );
  assert.throws(
    () => planQwen35LogitsTileWinner({
      vocabularyStart: 247_808,
      vocabularyRows: 263,
      candidateSlot: 242,
    }),
    /tile|mathematical/i,
  );
});

test("defines separate tile and indexed reduction kernels with actual token ids", () => {
  assert.deepEqual(
    QWEN35_LOGITS_REDUCTION_KERNELS.map((kernel) => kernel.operation),
    ["logits-tile-top-1", "indexed-top-1"],
  );
  const tile = QWEN35_LOGITS_REDUCTION_KERNELS[0]!;
  const final = QWEN35_LOGITS_REDUCTION_KERNELS[1]!;
  assert.equal(tile.abi.bindings.logitsTile, 0);
  assert.equal(tile.abi.bindings.candidateTokenIds, 2);
  assert.equal(tile.abi.workgroupSize, 64);
  assert.match(tile.source, /@workgroup_size\(64\)/);
  assert.match(tile.source, /workgroupBarrier\(\)/);
  assert.match(tile.source, /params\.vocabulary_start \+ row/);
  assert.match(tile.source, /token >= 248070u/);
  assert.match(tile.source, /candidate_token_ids\[params\.candidate_slot\]/);
  assert.equal(final.abi.bindings.selectedToken, 2);
  assert.equal(final.abi.workgroupSize, 64);
  assert.match(final.source, /@workgroup_size\(64\)/);
  assert.match(final.source, /workgroupBarrier\(\)/);
  assert.match(final.source, /candidate_token_ids\[slot\]/);
  assert.match(final.source, /token >= 248070u/);
  assert.match(final.source, /selected_token\[0\] = 0xffffffffu/);
  assert.doesNotMatch(final.source, /arrayLength|248320/);
  assert.equal(Object.isFrozen(QWEN35_LOGITS_REDUCTION_KERNELS), true);
  assert.equal(Object.isFrozen(tile), true);
  assert.equal(Object.isFrozen(tile.abi.bindings), true);
});

test("rejects ranges that could address outside fixed workspace bounds", () => {
  assert.throws(
    () => selectQwen35FiniteTileWinner(new Float32Array(1_025), 0),
    /tile/i,
  );
  assert.deepEqual(
    selectQwen35IndexedWinner(
      Float32Array.of(100, 1),
      Uint32Array.of(248_070, 248_069),
      2,
    ),
    { score: 1, tokenId: 248_069 },
  );
  assert.throws(
    () => selectQwen35IndexedWinner(
      Float32Array.of(1),
      Uint32Array.of(1),
      2,
    ),
    /candidate/i,
  );
  assert.deepEqual(
    selectQwen35FiniteTileWinner(Float32Array.of(5), 248_069),
    { score: 5, tokenId: 248_069 },
  );
  assert.throws(
    () => selectQwen35FiniteTileWinner(Float32Array.of(5, 6), 248_069),
    /tile/i,
  );
});
