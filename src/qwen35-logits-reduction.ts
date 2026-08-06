const QWEN35_VOCABULARY_SIZE = 248_320;
const QWEN35_DECODABLE_TOKEN_COUNT = 248_070;
const QWEN35_LOGITS_TILE_ROWS = 1_024;
const QWEN35_MATHEMATICAL_TILE_COUNT = Math.ceil(
  QWEN35_DECODABLE_TOKEN_COUNT / QWEN35_LOGITS_TILE_ROWS,
);
const QWEN35_TOP_K_CAPACITY = 256;

export const QWEN35_NO_SELECTED_TOKEN = 0xffff_ffff;

export interface Qwen35GreedyWinner {
  readonly score: number;
  readonly tokenId: number;
}

export type Qwen35LogitsReductionOperation =
  | "logits-tile-top-1"
  | "indexed-top-1";

export interface Qwen35LogitsReductionKernel {
  readonly id: string;
  readonly operation: Qwen35LogitsReductionOperation;
  readonly entryPoint: "main";
  readonly source: string;
  readonly abi: Readonly<{
    readonly bindings: Readonly<Record<string, number>>;
    readonly uniformWords: 4;
    readonly workgroupSize: 64;
  }>;
}

export interface Qwen35LogitsReductionPlan {
  readonly operation: Qwen35LogitsReductionOperation;
  readonly uniformWords: readonly [number, number, number, number];
  readonly workgroups: Readonly<{ readonly x: 1; readonly y: 1; readonly z: 1 }>;
}

function finiteScore(score: number): boolean {
  return Number.isFinite(score);
}

function requireLogicalTile(vocabularyStart: number, vocabularyRows: number): void {
  if (
    !Number.isSafeInteger(vocabularyStart) ||
    vocabularyStart < 0 ||
    !Number.isSafeInteger(vocabularyRows) ||
    vocabularyRows < 1 ||
    vocabularyRows > QWEN35_LOGITS_TILE_ROWS ||
    vocabularyStart + vocabularyRows > QWEN35_DECODABLE_TOKEN_COUNT
  ) {
    throw new Error("Qwen3.5 logical logits tile is invalid");
  }
}

/** CPU oracle for the exact stable finite-score policy used by the tile shader. */
export function selectQwen35FiniteTileWinner(
  scores: Float32Array,
  vocabularyStart: number,
): Qwen35GreedyWinner | null {
  requireLogicalTile(vocabularyStart, scores.length);
  let winner: Qwen35GreedyWinner | null = null;
  for (let row = 0; row < scores.length; row += 1) {
    const score = scores[row]!;
    if (!finiteScore(score)) continue;
    const tokenId = vocabularyStart + row;
    if (
      winner === null ||
      score > winner.score ||
      (score === winner.score && tokenId < winner.tokenId)
    ) {
      winner = { score, tokenId };
    }
  }
  return winner === null ? null : Object.freeze(winner);
}

/** CPU oracle for reducing logical-tile winners by their real vocabulary IDs. */
export function selectQwen35IndexedWinner(
  candidateScores: Float32Array,
  candidateTokenIds: Uint32Array,
  candidateCount: number,
): Qwen35GreedyWinner | null {
  if (
    !Number.isSafeInteger(candidateCount) ||
    candidateCount < 1 ||
    candidateCount > QWEN35_TOP_K_CAPACITY ||
    candidateScores.length < candidateCount ||
    candidateTokenIds.length < candidateCount
  ) {
    throw new Error("Qwen3.5 logits candidate range is invalid");
  }
  let winner: Qwen35GreedyWinner | null = null;
  for (let slot = 0; slot < candidateCount; slot += 1) {
    const tokenId = candidateTokenIds[slot]!;
    if (tokenId === QWEN35_NO_SELECTED_TOKEN) continue;
    if (tokenId >= QWEN35_DECODABLE_TOKEN_COUNT) continue;
    const score = candidateScores[slot]!;
    if (!finiteScore(score)) continue;
    if (
      winner === null ||
      score > winner.score ||
      (score === winner.score && tokenId < winner.tokenId)
    ) {
      winner = { score, tokenId };
    }
  }
  return winner === null ? null : Object.freeze(winner);
}

const LOGITS_TILE_TOP_1_WGSL = /* wgsl */ `
struct Params {
  vocabulary_rows: u32,
  vocabulary_start: u32,
  candidate_slot: u32,
  pad: u32,
}
@group(0) @binding(0) var<storage, read> logits_tile: array<f32>;
@group(0) @binding(1) var<storage, read_write> candidate_scores: array<f32>;
@group(0) @binding(2) var<storage, read_write> candidate_token_ids: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;
var<workgroup> best_scores: array<f32, 64>;
var<workgroup> best_tokens: array<u32, 64>;
var<workgroup> best_found: array<u32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) local: vec3<u32>) {
  let lane = local.x;
  var found = false;
  var best_score = 0.0f;
  var best_token = 0xffffffffu;
  for (var row = lane; row < params.vocabulary_rows; row += 64u) {
    let score = logits_tile[row];
    let exponent = bitcast<u32>(score) & 0x7f800000u;
    if (exponent == 0x7f800000u) { continue; }
    let token = params.vocabulary_start + row;
    if (token >= 248070u) { continue; }
    if (!found || score > best_score ||
        (score == best_score && token < best_token)) {
      found = true;
      best_score = score;
      best_token = token;
    }
  }
  best_scores[lane] = best_score;
  best_tokens[lane] = best_token;
  best_found[lane] = select(0u, 1u, found);
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride /= 2u) {
    if (lane < stride && best_found[lane + stride] != 0u) {
      let other_score = best_scores[lane + stride];
      let other_token = best_tokens[lane + stride];
      if (best_found[lane] == 0u || other_score > best_scores[lane] ||
          (other_score == best_scores[lane] && other_token < best_tokens[lane])) {
        best_scores[lane] = other_score;
        best_tokens[lane] = other_token;
        best_found[lane] = 1u;
      }
    }
    workgroupBarrier();
  }
  if (lane == 0u) {
    candidate_scores[params.candidate_slot] = best_scores[0];
    candidate_token_ids[params.candidate_slot] = best_tokens[0];
  }
}`;

const INDEXED_TOP_1_WGSL = /* wgsl */ `
struct Params { candidate_count: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<storage, read> candidate_scores: array<f32>;
@group(0) @binding(1) var<storage, read> candidate_token_ids: array<u32>;
@group(0) @binding(2) var<storage, read_write> selected_token: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;
var<workgroup> best_scores: array<f32, 64>;
var<workgroup> best_tokens: array<u32, 64>;
var<workgroup> best_found: array<u32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) local: vec3<u32>) {
  let lane = local.x;
  if (lane == 0u) selected_token[0] = 0xffffffffu;
  var found = false;
  var best_score = 0.0f;
  var best_token = 0xffffffffu;
  for (var slot = lane; slot < params.candidate_count; slot += 64u) {
    let token = candidate_token_ids[slot];
    let score = candidate_scores[slot];
    let exponent = bitcast<u32>(score) & 0x7f800000u;
    if (token == 0xffffffffu || token >= 248070u ||
        exponent == 0x7f800000u) { continue; }
    if (!found || score > best_score ||
        (score == best_score && token < best_token)) {
      found = true;
      best_score = score;
      best_token = token;
    }
  }
  best_scores[lane] = best_score;
  best_tokens[lane] = best_token;
  best_found[lane] = select(0u, 1u, found);
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride /= 2u) {
    if (lane < stride && best_found[lane + stride] != 0u) {
      let other_score = best_scores[lane + stride];
      let other_token = best_tokens[lane + stride];
      if (best_found[lane] == 0u || other_score > best_scores[lane] ||
          (other_score == best_scores[lane] && other_token < best_tokens[lane])) {
        best_scores[lane] = other_score;
        best_tokens[lane] = other_token;
        best_found[lane] = 1u;
      }
    }
    workgroupBarrier();
  }
  if (lane == 0u) selected_token[0] = best_tokens[0];
}`;

function kernel(
  operation: Qwen35LogitsReductionOperation,
  source: string,
  bindings: Readonly<Record<string, number>>,
): Qwen35LogitsReductionKernel {
  return Object.freeze({
    id: `${operation}-decode-portable-f32`,
    operation,
    entryPoint: "main",
    source,
    abi: Object.freeze({
      bindings: Object.freeze({ ...bindings }),
      uniformWords: 4,
      workgroupSize: 64,
    }),
  });
}

export const QWEN35_LOGITS_REDUCTION_KERNELS: readonly Qwen35LogitsReductionKernel[] =
  Object.freeze([
    kernel("logits-tile-top-1", LOGITS_TILE_TOP_1_WGSL, {
      logitsTile: 0,
      candidateScores: 1,
      candidateTokenIds: 2,
      uniforms: 3,
    }),
    kernel("indexed-top-1", INDEXED_TOP_1_WGSL, {
      candidateScores: 0,
      candidateTokenIds: 1,
      selectedToken: 2,
      uniforms: 3,
    }),
  ]);

const ONE_WORKGROUP = Object.freeze({ x: 1, y: 1, z: 1 } as const);

export function planQwen35LogitsTileWinner(input: {
  readonly vocabularyStart: number;
  readonly vocabularyRows: number;
  readonly candidateSlot: number;
}): Qwen35LogitsReductionPlan {
  requireLogicalTile(input.vocabularyStart, input.vocabularyRows);
  if (
    !Number.isSafeInteger(input.candidateSlot) ||
    input.candidateSlot < 0 ||
    input.candidateSlot >= QWEN35_MATHEMATICAL_TILE_COUNT ||
    input.vocabularyStart !== input.candidateSlot * QWEN35_LOGITS_TILE_ROWS ||
    input.vocabularyRows !== Math.min(
      QWEN35_LOGITS_TILE_ROWS,
      QWEN35_DECODABLE_TOKEN_COUNT - input.vocabularyStart,
    )
  ) {
    throw new Error("Qwen3.5 mathematical logits tile assignment is invalid");
  }
  return Object.freeze({
    operation: "logits-tile-top-1",
    uniformWords: Object.freeze([
      input.vocabularyRows,
      input.vocabularyStart,
      input.candidateSlot,
      0,
    ] as const),
    workgroups: ONE_WORKGROUP,
  });
}

export function planQwen35FinalTokenSelection(input: {
  readonly candidateCount: number;
}): Qwen35LogitsReductionPlan {
  if (
    !Number.isSafeInteger(input.candidateCount) ||
    input.candidateCount !== QWEN35_MATHEMATICAL_TILE_COUNT
  ) {
    throw new Error("Qwen3.5 logits candidate count is invalid");
  }
  return Object.freeze({
    operation: "indexed-top-1",
    uniformWords: Object.freeze([input.candidateCount, 0, 0, 0] as const),
    workgroups: ONE_WORKGROUP,
  });
}
