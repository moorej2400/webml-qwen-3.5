import { byteLevelStringToBytes } from "./byte-level.js";
import { diagnosticError } from "./diagnostics.js";
import {
  deserializeCompiledTokenizer,
  QWEN35_MODEL_LOGIT_ROWS,
  readAuthenticatedTokenizerArtifact,
  type CompiledTokenizerTables,
} from "./tokenizer-binary.js";

const PRETOKEN_PATTERN =
  /(?:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+/giu;

export const PINNED_QWEN35_COMPILED_TOKENIZER = Object.freeze({
  byteLength: 5_806_953,
  sha256: "7c0e92451601511a7396d8897f6f4ae16d4783197f6467e9d15e4bc6bc7197bd",
  baseVocabSize: 248_044,
  mergeCount: 247_587,
  addedTokenCount: 26,
  decodableTokenCount: 248_070,
  modelLogitRows: 248_320,
  maskedModelRows: 250,
});

export interface Qwen35TokenizerLimits {
  readonly maxInputCodeUnits: number;
  readonly maxInputBytes: number;
  readonly maxPieces: number;
  readonly maxPieceBytes: number;
  readonly maxMergeWork: number;
  readonly maxDecodeTokens: number;
}

const DEFAULT_LIMITS: Readonly<Qwen35TokenizerLimits> = Object.freeze({
  maxInputCodeUnits: 1_000_000,
  maxInputBytes: 4 * 1024 * 1024,
  maxPieces: 262_144,
  maxPieceBytes: 16_384,
  maxMergeWork: 2_000_000,
  maxDecodeTokens: 16_384,
});

export interface EncodeOptions {
  /** Added tokens are rejected for untrusted text and allowed only for templates. */
  readonly addedTokens?: "reject" | "allow";
}

export interface DecodeOptions {
  readonly skipSpecialTokens?: boolean;
}

export interface UnsafeTokenizerReferenceFixture {
  readonly baseVocabSize: number;
  readonly tokenCount: number;
  readonly tokens: readonly {
    readonly id: number;
    readonly value: string;
  }[];
  readonly merges: readonly {
    readonly rank: number;
    readonly left: number;
    readonly right: number;
    readonly result: number;
  }[];
  readonly addedTokens: readonly {
    readonly id: number;
    readonly content: string;
    readonly special: boolean;
  }[];
}

interface Merge {
  readonly rank: number;
  readonly result: number;
}

interface MergeCandidate {
  readonly rank: number;
  readonly result: number;
  readonly left: number;
  readonly right: number;
  readonly leftVersion: number;
  readonly rightVersion: number;
}

class MergeCandidateHeap {
  readonly #values: MergeCandidate[] = [];

  push(candidate: MergeCandidate): void {
    this.#values.push(candidate);
    let index = this.#values.length - 1;
    while (index > 0) {
      const parent = Math.floor((index - 1) / 2);
      if (!this.#before(candidate, this.#values[parent]!)) {
        break;
      }
      this.#values[index] = this.#values[parent]!;
      index = parent;
    }
    this.#values[index] = candidate;
  }

  pop(): MergeCandidate | undefined {
    const first = this.#values[0];
    const last = this.#values.pop();
    if (first === undefined || last === undefined || this.#values.length === 0) {
      return first;
    }
    let index = 0;
    while (true) {
      const left = index * 2 + 1;
      if (left >= this.#values.length) {
        break;
      }
      const right = left + 1;
      const child =
        right < this.#values.length &&
        this.#before(this.#values[right]!, this.#values[left]!)
          ? right
          : left;
      if (!this.#before(this.#values[child]!, last)) {
        break;
      }
      this.#values[index] = this.#values[child]!;
      index = child;
    }
    this.#values[index] = last;
    return first;
  }

  #before(left: MergeCandidate, right: MergeCandidate): boolean {
    return left.rank < right.rank ||
      (left.rank === right.rank && left.left < right.left);
  }
}

interface AddedToken {
  readonly id: number;
  readonly content: string;
  readonly special: boolean;
}

interface InputSegment {
  readonly kind: "text" | "added";
  readonly value: string;
  readonly id?: number;
}

function pairKey(left: number, right: number, tokenCount: number): number {
  return left * tokenCount + right;
}

function mergedLimits(
  overrides: Partial<Qwen35TokenizerLimits> | undefined,
): Readonly<Qwen35TokenizerLimits> {
  const limits = { ...DEFAULT_LIMITS, ...overrides };
  for (const [name, value] of Object.entries(limits)) {
    if (!Number.isSafeInteger(value) || value <= 0) {
      throw new Error(`${name} must be a positive safe integer`);
    }
  }
  return Object.freeze(limits);
}

function prepareAndValidateTables(
  source: CompiledTokenizerTables,
  copy: boolean,
): CompiledTokenizerTables {
  const tables = copy
    ? {
        baseVocabSize: source.baseVocabSize,
        tokenCount: source.tokenCount,
        tokenOffsets: source.tokenOffsets.slice(),
        tokenBytes: source.tokenBytes.slice(),
        merges: source.merges.slice(),
        addedTokenIds: source.addedTokenIds.slice(),
        addedTokenFlags: source.addedTokenFlags.slice(),
      }
    : source;
  if (
    !Number.isSafeInteger(tables.baseVocabSize) ||
    !Number.isSafeInteger(tables.tokenCount) ||
    tables.baseVocabSize <= 0 ||
    tables.tokenCount < tables.baseVocabSize ||
    tables.tokenOffsets.length !== tables.tokenCount + 1 ||
    tables.merges.length % 3 !== 0 ||
    tables.addedTokenIds.length !== tables.tokenCount - tables.baseVocabSize ||
    tables.addedTokenFlags.length !== tables.addedTokenIds.length
  ) {
    throw new Error("Tokenizer tables have invalid counts");
  }
  if (tables.tokenCount > QWEN35_MODEL_LOGIT_ROWS) {
    throw new Error("Tokenizer token count exceeds Qwen3.5 model logit rows");
  }
  let previous = 0;
  for (const offset of tables.tokenOffsets) {
    if (offset < previous || offset > tables.tokenBytes.byteLength) {
      throw new Error("Tokenizer tables have invalid token offsets");
    }
    previous = offset;
  }
  if (previous !== tables.tokenBytes.byteLength) {
    throw new Error("Tokenizer tables do not cover their token bytes");
  }
  for (let index = 0; index < tables.merges.length; index += 1) {
    if (tables.merges[index]! >= tables.baseVocabSize) {
      throw new Error("Tokenizer tables contain an invalid merge id");
    }
  }
  for (const [index, id] of tables.addedTokenIds.entries()) {
    if (
      id !== tables.baseVocabSize + index ||
      tables.addedTokenFlags[index]! > 1
    ) {
      throw new Error("Tokenizer tables contain an invalid added token");
    }
  }
  return tables;
}

/**
 * Executes the fixed Qwen3.5 NFC, regex, ByteLevel, and BPE pipeline.
 *
 * Added-token matching happens before NFC, as required by the pinned
 * `normalized: false` records. Public encoding rejects those records so chat
 * control tokens cannot be injected through message content.
 */
export class Qwen35Tokenizer {
  readonly #tables: CompiledTokenizerTables;
  readonly #limits: Readonly<Qwen35TokenizerLimits>;
  readonly #byteTokenIds = new Int32Array(256);
  readonly #merges = new Map<number, Merge>();
  readonly #added = new Map<number, AddedToken>();
  readonly #addedByFirstCodeUnit = new Map<string, readonly AddedToken[]>();

  private constructor(
    source: CompiledTokenizerTables,
    limits?: Partial<Qwen35TokenizerLimits>,
    copyTables = true,
    mergeRanks?: Uint32Array,
  ) {
    this.#tables = prepareAndValidateTables(source, copyTables);
    this.#limits = mergedLimits(limits);
    if (
      mergeRanks !== undefined &&
      mergeRanks.length !== this.#tables.merges.length / 3
    ) {
      throw new Error("Tokenizer reference merge ranks are invalid");
    }
    this.#byteTokenIds.fill(-1);

    for (let id = 0; id < this.#tables.baseVocabSize; id += 1) {
      const bytes = this.#tokenBytes(id);
      if (bytes.byteLength === 1) {
        const byte = bytes[0]!;
        if (this.#byteTokenIds[byte] !== -1) {
          throw new Error("Tokenizer tables contain duplicate byte tokens");
        }
        this.#byteTokenIds[byte] = id;
      }
    }

    for (let index = 0; index < this.#tables.merges.length; index += 3) {
      const left = this.#tables.merges[index]!;
      const right = this.#tables.merges[index + 1]!;
      const result = this.#tables.merges[index + 2]!;
      const key = pairKey(left, right, this.#tables.tokenCount);
      if (this.#merges.has(key)) {
        throw new Error("Tokenizer tables contain a duplicate merge pair");
      }
      this.#merges.set(key, {
        rank: mergeRanks?.[index / 3] ?? index / 3,
        result,
      });
    }

    const decoder = new TextDecoder("utf-8", { fatal: true });
    const grouped = new Map<string, AddedToken[]>();
    for (const [index, id] of this.#tables.addedTokenIds.entries()) {
      let content: string;
      try {
        content = decoder.decode(this.#tokenBytes(id));
      } catch {
        throw new Error("Tokenizer added token is not valid UTF-8");
      }
      if (content.length === 0) {
        throw new Error("Tokenizer added token must not be empty");
      }
      const token = Object.freeze({
        id,
        content,
        special: this.#tables.addedTokenFlags[index] === 1,
      });
      this.#added.set(id, token);
      const first = content[0]!;
      const candidates = grouped.get(first) ?? [];
      candidates.push(token);
      grouped.set(first, candidates);
    }
    for (const [first, candidates] of grouped) {
      candidates.sort((left, right) => right.content.length - left.content.length);
      this.#addedByFirstCodeUnit.set(first, Object.freeze(candidates));
    }
  }

  static fromUnsafeCompiledArtifactForTests(
    binary: Uint8Array,
    limits?: Partial<Qwen35TokenizerLimits>,
  ): Qwen35Tokenizer {
    return new Qwen35Tokenizer(deserializeCompiledTokenizer(binary), limits);
  }

  static fromUnsafeTablesForTests(
    tables: CompiledTokenizerTables,
    limits?: Partial<Qwen35TokenizerLimits>,
  ): Qwen35Tokenizer {
    return new Qwen35Tokenizer(tables, limits);
  }

  static fromUnsafeReferenceFixtureForTests(
    fixture: UnsafeTokenizerReferenceFixture,
    limits?: Partial<Qwen35TokenizerLimits>,
  ): Qwen35Tokenizer {
    const reference = compileUnsafeReferenceFixture(fixture);
    return new Qwen35Tokenizer(
      reference.tables,
      limits,
      false,
      reference.mergeRanks,
    );
  }

  static async fromPinnedArtifact(
    chunks: AsyncIterable<Uint8Array>,
    declaredByteLength: number,
    limits?: Partial<Qwen35TokenizerLimits>,
  ): Promise<Qwen35Tokenizer> {
    const binary = await readAuthenticatedTokenizerArtifact(
      chunks,
      PINNED_QWEN35_COMPILED_TOKENIZER,
      declaredByteLength,
    );
    const tables = deserializeCompiledTokenizer(binary);
    requirePinnedTables(tables);
    return new Qwen35Tokenizer(tables, limits, false);
  }

  get decodableTokenCount(): number {
    return this.#tables.tokenCount;
  }

  get modelLogitRows(): number {
    return QWEN35_MODEL_LOGIT_ROWS;
  }

  get undecodableLogitRows(): number {
    return QWEN35_MODEL_LOGIT_ROWS - this.#tables.tokenCount;
  }

  /** Samplers must mask every logit row for which this returns false. */
  isDecodableTokenId(id: number): boolean {
    return Number.isInteger(id) && id >= 0 && id < this.#tables.tokenCount;
  }

  encode(text: string, options: EncodeOptions = {}): number[] {
    if (typeof text !== "string") {
      throw diagnosticError(
        "tokenizer-input-invalid",
        "Tokenizer input must be text",
      );
    }
    if (text.length > this.#limits.maxInputCodeUnits) {
      throw diagnosticError(
        "tokenizer-input-limit",
        "Tokenizer input exceeds its configured limit",
      );
    }
    const addedPolicy = options.addedTokens ?? "reject";
    if (addedPolicy !== "reject" && addedPolicy !== "allow") {
      throw diagnosticError(
        "tokenizer-option-invalid",
        "Tokenizer added-token policy is invalid",
      );
    }

    const segments = this.#splitAddedTokens(text);
    if (
      addedPolicy === "reject" &&
      segments.some((segment) => segment.kind === "added")
    ) {
      throw diagnosticError(
        "tokenizer-added-token-in-input",
        "Input contains a reserved tokenizer token",
      );
    }

    const ids: number[] = [];
    let inputBytes = 0;
    let pieceCount = 0;
    let mergeWork = 0;
    for (const segment of segments) {
      if (segment.kind === "added") {
        ids.push(segment.id!);
        continue;
      }
      const normalized = segment.value.normalize("NFC");
      if (normalized.length > this.#limits.maxInputCodeUnits) {
        throw diagnosticError(
          "tokenizer-input-limit",
          "Tokenizer input exceeds its configured limit",
        );
      }
      const pattern = new RegExp(PRETOKEN_PATTERN.source, PRETOKEN_PATTERN.flags);
      let cursor = 0;
      for (const match of normalized.matchAll(pattern)) {
        if (match.index !== cursor || match[0].length === 0) {
          throw diagnosticError(
            "tokenizer-pretokenizer-failed",
            "Tokenizer pre-tokenization failed",
          );
        }
        cursor += match[0].length;
        pieceCount += 1;
        if (pieceCount > this.#limits.maxPieces) {
          throw diagnosticError(
            "tokenizer-piece-count-limit",
            "Tokenizer piece count exceeds its configured limit",
          );
        }
        const bytes = new TextEncoder().encode(match[0]);
        inputBytes += bytes.byteLength;
        if (inputBytes > this.#limits.maxInputBytes) {
          throw diagnosticError(
            "tokenizer-input-byte-limit",
            "Tokenizer input bytes exceed their configured limit",
          );
        }
        if (bytes.byteLength > this.#limits.maxPieceBytes) {
          throw diagnosticError(
            "tokenizer-piece-limit",
            "Tokenizer piece exceeds its configured limit",
          );
        }
        const encoded = this.#encodePiece(bytes, mergeWork);
        mergeWork = encoded.mergeWork;
        ids.push(...encoded.ids);
      }
      if (cursor !== normalized.length) {
        throw diagnosticError(
          "tokenizer-pretokenizer-failed",
          "Tokenizer pre-tokenization failed",
        );
      }
    }
    return ids;
  }

  decode(ids: Iterable<number>, options: DecodeOptions = {}): string {
    const decoder = this.createStreamingDecoder(options);
    let output = "";
    for (const id of ids) {
      output += decoder.push(id);
    }
    return output + decoder.finish();
  }

  createStreamingDecoder(options: DecodeOptions = {}): QwenStreamingDecoder {
    return new QwenStreamingDecoder(
      this.#tables,
      this.#added,
      options.skipSpecialTokens ?? false,
      this.#limits.maxDecodeTokens,
    );
  }

  addedTokenId(content: string): number | undefined {
    const candidates = this.#addedByFirstCodeUnit.get(content[0] ?? "");
    return candidates?.find((candidate) => candidate.content === content)?.id;
  }

  containsAddedToken(text: string): boolean {
    if (
      typeof text !== "string" ||
      text.length > this.#limits.maxInputCodeUnits
    ) {
      throw diagnosticError(
        "tokenizer-input-limit",
        "Tokenizer input exceeds its configured limit",
      );
    }
    return this.#splitAddedTokens(text).some(
      (segment) => segment.kind === "added",
    );
  }

  #tokenBytes(id: number): Uint8Array {
    const start = this.#tables.tokenOffsets[id]!;
    const end = this.#tables.tokenOffsets[id + 1]!;
    return this.#tables.tokenBytes.subarray(start, end);
  }

  #findAddedToken(text: string, offset: number): AddedToken | undefined {
    const candidates = this.#addedByFirstCodeUnit.get(text[offset]!);
    return candidates?.find((token) => text.startsWith(token.content, offset));
  }

  #splitAddedTokens(text: string): InputSegment[] {
    const segments: InputSegment[] = [];
    let textStart = 0;
    let offset = 0;
    while (offset < text.length) {
      const token = this.#findAddedToken(text, offset);
      if (token === undefined) {
        offset += 1;
        continue;
      }
      if (offset > textStart) {
        segments.push({
          kind: "text",
          value: text.slice(textStart, offset),
        });
      }
      segments.push({
        kind: "added",
        value: token.content,
        id: token.id,
      });
      offset += token.content.length;
      textStart = offset;
    }
    if (textStart < text.length || segments.length === 0) {
      segments.push({ kind: "text", value: text.slice(textStart) });
    }
    return segments;
  }

  #encodePiece(
    bytes: Uint8Array,
    initialMergeWork: number,
  ): { ids: number[]; mergeWork: number } {
    const symbols = Array.from(bytes, (byte) => {
      const id = this.#byteTokenIds[byte]!;
      if (id < 0) {
        throw diagnosticError(
          "tokenizer-byte-token-missing",
          "Tokenizer byte vocabulary is incomplete",
        );
      }
      return id;
    });
    let mergeWork = initialMergeWork;
    const consumeWork = (): void => {
      mergeWork += 1;
      if (mergeWork > this.#limits.maxMergeWork) {
        throw diagnosticError(
          "tokenizer-merge-work-limit",
          "Tokenizer merge work exceeds its configured limit",
        );
      }
    };
    if (symbols.length < 2) {
      return { ids: symbols, mergeWork };
    }

    const values = Int32Array.from(symbols);
    const previous = new Int32Array(symbols.length);
    const next = new Int32Array(symbols.length);
    const versions = new Uint32Array(symbols.length);
    const alive = new Uint8Array(symbols.length);
    alive.fill(1);
    for (let index = 0; index < symbols.length; index += 1) {
      previous[index] = index - 1;
      next[index] = index + 1 < symbols.length ? index + 1 : -1;
    }
    const heap = new MergeCandidateHeap();

    const enqueue = (left: number): void => {
      const right = left >= 0 ? next[left]! : -1;
      if (left < 0 || right < 0 || alive[left] !== 1 || alive[right] !== 1) {
        return;
      }
      consumeWork();
      const merge = this.#merges.get(
        pairKey(values[left]!, values[right]!, this.#tables.tokenCount),
      );
      if (merge !== undefined) {
        heap.push({
          rank: merge.rank,
          result: merge.result,
          left,
          right,
          leftVersion: versions[left]!,
          rightVersion: versions[right]!,
        });
      }
    };

    for (let index = 0; index + 1 < symbols.length; index += 1) {
      enqueue(index);
    }
    while (true) {
      const candidate = heap.pop();
      if (candidate === undefined) {
        break;
      }
      consumeWork();
      if (
        alive[candidate.left] !== 1 ||
        alive[candidate.right] !== 1 ||
        next[candidate.left] !== candidate.right ||
        versions[candidate.left] !== candidate.leftVersion ||
        versions[candidate.right] !== candidate.rightVersion
      ) {
        continue;
      }

      const prior = previous[candidate.left]!;
      const following = next[candidate.right]!;
      values[candidate.left] = candidate.result;
      versions[candidate.left] = versions[candidate.left]! + 1;
      alive[candidate.right] = 0;
      versions[candidate.right] = versions[candidate.right]! + 1;
      next[candidate.left] = following;
      if (following >= 0) {
        previous[following] = candidate.left;
      }
      enqueue(prior);
      enqueue(candidate.left);
    }

    const result: number[] = [];
    for (let node = 0; node >= 0; node = next[node]!) {
      if (alive[node] === 1) {
        result.push(values[node]!);
      }
    }
    return { ids: result, mergeWork };
  }
}

function compileUnsafeReferenceFixture(
  fixture: UnsafeTokenizerReferenceFixture,
): {
  tables: CompiledTokenizerTables;
  mergeRanks: Uint32Array;
} {
  if (
    !Number.isSafeInteger(fixture.baseVocabSize) ||
    !Number.isSafeInteger(fixture.tokenCount) ||
    fixture.baseVocabSize <= 0 ||
    fixture.tokenCount > QWEN35_MODEL_LOGIT_ROWS ||
    fixture.addedTokens.length !==
      fixture.tokenCount - fixture.baseVocabSize
  ) {
    throw new Error("Tokenizer reference fixture counts are invalid");
  }

  const bytesById = new Map<number, Uint8Array>();
  for (const token of fixture.tokens) {
    if (
      !Number.isInteger(token.id) ||
      token.id < 0 ||
      token.id >= fixture.baseVocabSize ||
      bytesById.has(token.id)
    ) {
      throw new Error("Tokenizer reference token id is invalid");
    }
    bytesById.set(token.id, byteLevelStringToBytes(token.value));
  }
  const encoder = new TextEncoder();
  const added = [...fixture.addedTokens].sort(
    (left, right) => left.id - right.id,
  );
  for (const [index, token] of added.entries()) {
    if (
      token.id !== fixture.baseVocabSize + index ||
      typeof token.content !== "string" ||
      token.content.length === 0 ||
      typeof token.special !== "boolean" ||
      bytesById.has(token.id)
    ) {
      throw new Error("Tokenizer reference added token is invalid");
    }
    bytesById.set(token.id, encoder.encode(token.content));
  }

  const offsets = new Uint32Array(fixture.tokenCount + 1);
  const parts: Uint8Array[] = [];
  let byteLength = 0;
  for (let id = 0; id < fixture.tokenCount; id += 1) {
    offsets[id] = byteLength;
    const bytes = bytesById.get(id);
    if (bytes !== undefined) {
      parts.push(bytes);
      byteLength += bytes.byteLength;
    }
  }
  offsets[fixture.tokenCount] = byteLength;
  const tokenBytes = new Uint8Array(byteLength);
  let byteOffset = 0;
  for (const part of parts) {
    tokenBytes.set(part, byteOffset);
    byteOffset += part.byteLength;
  }

  const merges = new Uint32Array(fixture.merges.length * 3);
  const mergeRanks = new Uint32Array(fixture.merges.length);
  let previousRank = -1;
  for (const [index, merge] of fixture.merges.entries()) {
    if (
      !Number.isInteger(merge.rank) ||
      merge.rank <= previousRank ||
      merge.left < 0 ||
      merge.left >= fixture.baseVocabSize ||
      merge.right < 0 ||
      merge.right >= fixture.baseVocabSize ||
      merge.result < 0 ||
      merge.result >= fixture.baseVocabSize
    ) {
      throw new Error("Tokenizer reference merge is invalid");
    }
    mergeRanks[index] = merge.rank;
    merges[index * 3] = merge.left;
    merges[index * 3 + 1] = merge.right;
    merges[index * 3 + 2] = merge.result;
    previousRank = merge.rank;
  }
  return {
    tables: {
      baseVocabSize: fixture.baseVocabSize,
      tokenCount: fixture.tokenCount,
      tokenOffsets: offsets,
      tokenBytes,
      merges,
      addedTokenIds: Uint32Array.from(added.map((token) => token.id)),
      addedTokenFlags: Uint8Array.from(
        added.map((token) => (token.special ? 1 : 0)),
      ),
    },
    mergeRanks,
  };
}

function requirePinnedTables(tables: CompiledTokenizerTables): void {
  const pinned = PINNED_QWEN35_COMPILED_TOKENIZER;
  if (
    tables.baseVocabSize !== pinned.baseVocabSize ||
    tables.merges.length / 3 !== pinned.mergeCount ||
    tables.addedTokenIds.length !== pinned.addedTokenCount ||
    tables.tokenCount !== pinned.decodableTokenCount ||
    QWEN35_MODEL_LOGIT_ROWS - tables.tokenCount !== pinned.maskedModelRows
  ) {
    throw new Error("Tokenizer artifact counts do not match the pinned package");
  }
}

/**
 * Creates the production tokenizer only after exact byte and table identity.
 *
 * The authenticated byte buffer transfers into the session. Token byte data
 * remains a view over that owned buffer, so the loader does not clone the
 * complete 5.8 MB artifact after hashing.
 */
export async function loadPinnedQwen35Tokenizer(
  chunks: AsyncIterable<Uint8Array>,
  declaredByteLength: number,
  limits?: Partial<Qwen35TokenizerLimits>,
): Promise<Qwen35Tokenizer> {
  return Qwen35Tokenizer.fromPinnedArtifact(
    chunks,
    declaredByteLength,
    limits,
  );
}

/** Maintains one TextDecoder stream so token boundaries cannot split scalars. */
export class QwenStreamingDecoder {
  readonly #tables: CompiledTokenizerTables;
  readonly #added: ReadonlyMap<number, AddedToken>;
  readonly #skipSpecialTokens: boolean;
  readonly #maxTokens: number;
  #decoder = new TextDecoder();
  #closed = false;
  #tokenCount = 0;

  constructor(
    tables: CompiledTokenizerTables,
    added: ReadonlyMap<number, AddedToken>,
    skipSpecialTokens: boolean,
    maxTokens: number,
  ) {
    this.#tables = tables;
    this.#added = added;
    this.#skipSpecialTokens = skipSpecialTokens;
    this.#maxTokens = maxTokens;
  }

  push(id: number): string {
    if (this.#closed) {
      throw diagnosticError(
        "tokenizer-decoder-closed",
        "Tokenizer decoder is already closed",
      );
    }
    if (!Number.isInteger(id) || id < 0 || id >= this.#tables.tokenCount) {
      throw diagnosticError(
        "tokenizer-token-id-invalid",
        "Tokenizer token id is outside the vocabulary",
      );
    }
    this.#tokenCount += 1;
    if (this.#tokenCount > this.#maxTokens) {
      throw diagnosticError(
        "tokenizer-decode-token-limit",
        "Tokenizer decode token count exceeds its configured limit",
      );
    }
    const bytes = this.#tables.tokenBytes.subarray(
      this.#tables.tokenOffsets[id]!,
      this.#tables.tokenOffsets[id + 1]!,
    );
    const added = this.#added.get(id);
    if (added === undefined) {
      return this.#decoder.decode(bytes, { stream: true });
    }

    // Added-token literals are UTF-8 text, not ByteLevel bytes. Flush first so
    // an incomplete model byte sequence cannot combine across the boundary.
    const prefix = this.#decoder.decode();
    this.#decoder = new TextDecoder();
    if (added.special && this.#skipSpecialTokens) {
      return prefix;
    }
    return prefix + new TextDecoder().decode(bytes);
  }

  finish(): string {
    if (this.#closed) {
      throw diagnosticError(
        "tokenizer-decoder-closed",
        "Tokenizer decoder is already closed",
      );
    }
    this.#closed = true;
    return this.#decoder.decode();
  }
}
