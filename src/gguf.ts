export interface RandomAccessReader {
  readonly size: bigint;
  read(offset: bigint, length: number): Promise<Uint8Array>;
}

export const GgufMetadataType = {
  Uint8: 0,
  Int8: 1,
  Uint16: 2,
  Int16: 3,
  Uint32: 4,
  Int32: 5,
  Float32: 6,
  Bool: 7,
  String: 8,
  Array: 9,
  Uint64: 10,
  Int64: 11,
  Float64: 12,
} as const;

export type GgufMetadataType =
  (typeof GgufMetadataType)[keyof typeof GgufMetadataType];

export const GgmlType = {
  F32: 0,
  F16: 1,
  Q4_0: 2,
  Q4_1: 3,
  Q5_0: 6,
  Q5_1: 7,
  Q8_0: 8,
  Q8_1: 9,
  Q2_K: 10,
  Q3_K: 11,
  Q4_K: 12,
  Q5_K: 13,
  Q6_K: 14,
  Q8_K: 15,
  IQ2_XXS: 16,
  IQ2_XS: 17,
  IQ3_XXS: 18,
  IQ1_S: 19,
  IQ4_NL: 20,
  IQ3_S: 21,
  IQ2_S: 22,
  IQ4_XS: 23,
  I8: 24,
  I16: 25,
  I32: 26,
  I64: 27,
  F64: 28,
  IQ1_M: 29,
  BF16: 30,
  TQ1_0: 34,
  TQ2_0: 35,
  MXFP4: 39,
} as const;

export type GgmlType = (typeof GgmlType)[keyof typeof GgmlType];

export type GgufMetadataScalar = number | bigint | boolean | string;
export type GgufMetadataValue = GgufMetadataScalar | GgufMetadataScalar[];

export interface GgufTensorInfo {
  readonly name: string;
  readonly dimensions: readonly bigint[];
  readonly type: GgmlType;
  readonly offset: bigint;
}

export interface ParsedGguf {
  readonly version: 3;
  readonly metadata: Readonly<Record<string, GgufMetadataValue>>;
  readonly tensors: readonly GgufTensorInfo[];
  readonly alignment: number;
  readonly dataOffset: bigint;
}

export interface GgufParseLimits {
  readonly maxTensorCount?: number;
  readonly maxMetadataCount?: number;
  readonly maxStringBytes?: number;
  readonly maxArrayLength?: number;
  readonly maxTensorRank?: number;
}

const DEFAULT_LIMITS = {
  maxTensorCount: 1_000_000,
  maxMetadataCount: 1_000_000,
  maxStringBytes: 16 * 1024 * 1024,
  maxArrayLength: 1_000_000,
  maxTensorRank: 4,
} as const;

const SUPPORTED_TENSOR_TYPES = new Set<number>(Object.values(GgmlType));
const SUPPORTED_SCALAR_TYPES = new Set<number>([
  GgufMetadataType.Uint8,
  GgufMetadataType.Int8,
  GgufMetadataType.Uint16,
  GgufMetadataType.Int16,
  GgufMetadataType.Uint32,
  GgufMetadataType.Int32,
  GgufMetadataType.Float32,
  GgufMetadataType.Bool,
  GgufMetadataType.String,
  GgufMetadataType.Uint64,
  GgufMetadataType.Int64,
  GgufMetadataType.Float64,
]);

class Cursor {
  offset = 0n;

  constructor(
    private readonly reader: RandomAccessReader,
    private readonly maxStringBytes: number,
  ) {}

  async bytes(length: number): Promise<Uint8Array> {
    if (!Number.isSafeInteger(length) || length < 0) {
      throw new Error(`Invalid GGUF read length: ${length}`);
    }
    if (this.offset + BigInt(length) > this.reader.size) {
      throw new Error("Truncated GGUF input");
    }
    const value = await this.reader.read(this.offset, length);
    if (value.byteLength !== length) {
      throw new Error("Truncated GGUF input: reader returned a short read");
    }
    this.offset += BigInt(length);
    return value;
  }

  async u8(): Promise<number> {
    return (await this.bytes(1))[0]!;
  }

  async i8(): Promise<number> {
    return new DataView((await this.bytes(1)).buffer).getInt8(0);
  }

  async u16(): Promise<number> {
    return view(await this.bytes(2)).getUint16(0, true);
  }

  async i16(): Promise<number> {
    return view(await this.bytes(2)).getInt16(0, true);
  }

  async u32(): Promise<number> {
    return view(await this.bytes(4)).getUint32(0, true);
  }

  async i32(): Promise<number> {
    return view(await this.bytes(4)).getInt32(0, true);
  }

  async u64(): Promise<bigint> {
    return view(await this.bytes(8)).getBigUint64(0, true);
  }

  async i64(): Promise<bigint> {
    return view(await this.bytes(8)).getBigInt64(0, true);
  }

  async f32(): Promise<number> {
    return view(await this.bytes(4)).getFloat32(0, true);
  }

  async f64(): Promise<number> {
    return view(await this.bytes(8)).getFloat64(0, true);
  }

  async string(): Promise<string> {
    const length = await this.u64();
    if (length > BigInt(this.maxStringBytes)) {
      throw new Error(`GGUF string length ${length} exceeds the configured bound`);
    }
    try {
      return new TextDecoder("utf-8", { fatal: true }).decode(
        await this.bytes(Number(length)),
      );
    } catch (error) {
      if (error instanceof TypeError) {
        throw new Error("GGUF string contains invalid UTF-8", { cause: error });
      }
      throw error;
    }
  }
}

function view(bytes: Uint8Array): DataView {
  return new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
}

function boundedCount(value: bigint, maximum: number, label: string): number {
  if (value > BigInt(maximum)) {
    throw new Error(`GGUF ${label} ${value} exceeds the configured bound`);
  }
  return Number(value);
}

async function readScalar(
  cursor: Cursor,
  type: number,
): Promise<GgufMetadataScalar> {
  switch (type) {
    case GgufMetadataType.Uint8:
      return cursor.u8();
    case GgufMetadataType.Int8:
      return cursor.i8();
    case GgufMetadataType.Uint16:
      return cursor.u16();
    case GgufMetadataType.Int16:
      return cursor.i16();
    case GgufMetadataType.Uint32:
      return cursor.u32();
    case GgufMetadataType.Int32:
      return cursor.i32();
    case GgufMetadataType.Float32:
      return cursor.f32();
    case GgufMetadataType.Bool: {
      const value = await cursor.u8();
      if (value > 1) {
        throw new Error(`Invalid GGUF boolean value: ${value}`);
      }
      return value === 1;
    }
    case GgufMetadataType.String:
      return cursor.string();
    case GgufMetadataType.Uint64:
      return cursor.u64();
    case GgufMetadataType.Int64:
      return cursor.i64();
    case GgufMetadataType.Float64:
      return cursor.f64();
    default:
      throw new Error(`Unsupported GGUF metadata scalar type: ${type}`);
  }
}

async function readMetadataValue(
  cursor: Cursor,
  type: number,
  maxArrayLength: number,
): Promise<GgufMetadataValue> {
  if (type !== GgufMetadataType.Array) {
    return readScalar(cursor, type);
  }

  const elementType = await cursor.u32();
  if (!SUPPORTED_SCALAR_TYPES.has(elementType)) {
    throw new Error(`Unsupported GGUF array element type: ${elementType}`);
  }
  const length = boundedCount(
    await cursor.u64(),
    maxArrayLength,
    "array length",
  );
  const values: GgufMetadataScalar[] = [];
  for (let index = 0; index < length; index += 1) {
    values.push(await readScalar(cursor, elementType));
  }
  return values;
}

function requireAlignment(metadata: Record<string, GgufMetadataValue>): number {
  const value = metadata["general.alignment"] ?? 32;
  if (
    typeof value !== "number" ||
    !Number.isSafeInteger(value) ||
    value < 1 ||
    value > 1024 * 1024 ||
    (value & (value - 1)) !== 0
  ) {
    throw new Error("GGUF general.alignment must be a bounded power of two");
  }
  return value;
}

function align(value: bigint, alignment: number): bigint {
  const boundary = BigInt(alignment);
  return ((value + boundary - 1n) / boundary) * boundary;
}

/**
 * Parses only the GGUF header and directory; tensor payload bytes remain in the
 * random-access source so multi-gigabyte files are never materialized.
 */
export async function parseGguf(
  reader: RandomAccessReader,
  limits: GgufParseLimits = {},
): Promise<ParsedGguf> {
  const resolved = { ...DEFAULT_LIMITS, ...limits };
  if (reader.size < 24n) {
    throw new Error("Truncated GGUF input");
  }
  const cursor = new Cursor(reader, resolved.maxStringBytes);
  const magic = await cursor.bytes(4);
  if (
    magic[0] !== 0x47 ||
    magic[1] !== 0x47 ||
    magic[2] !== 0x55 ||
    magic[3] !== 0x46
  ) {
    throw new Error("Invalid GGUF magic");
  }
  const version = await cursor.u32();
  if (version !== 3) {
    throw new Error(`Unsupported GGUF version: ${version}`);
  }

  const tensorCount = boundedCount(
    await cursor.u64(),
    resolved.maxTensorCount,
    "tensor count",
  );
  const metadataCount = boundedCount(
    await cursor.u64(),
    resolved.maxMetadataCount,
    "metadata count",
  );

  const metadata: Record<string, GgufMetadataValue> = Object.create(null);
  for (let index = 0; index < metadataCount; index += 1) {
    const key = await cursor.string();
    if (Object.hasOwn(metadata, key)) {
      throw new Error(`Duplicate GGUF metadata key: ${key}`);
    }
    const type = await cursor.u32();
    if (
      type !== GgufMetadataType.Array &&
      !SUPPORTED_SCALAR_TYPES.has(type)
    ) {
      throw new Error(`Unsupported GGUF metadata type: ${type}`);
    }
    metadata[key] = await readMetadataValue(
      cursor,
      type,
      resolved.maxArrayLength,
    );
  }

  const alignment = requireAlignment(metadata);
  const tensors: GgufTensorInfo[] = [];
  for (let index = 0; index < tensorCount; index += 1) {
    const name = await cursor.string();
    const rank = await cursor.u32();
    if (rank < 1 || rank > resolved.maxTensorRank) {
      throw new Error(`Invalid GGUF tensor rank: ${rank}`);
    }
    const dimensions: bigint[] = [];
    for (let dimension = 0; dimension < rank; dimension += 1) {
      const size = await cursor.u64();
      if (size === 0n) {
        throw new Error(`Invalid zero GGUF tensor dimension for ${name}`);
      }
      dimensions.push(size);
    }
    const type = await cursor.u32();
    if (!SUPPORTED_TENSOR_TYPES.has(type)) {
      throw new Error(`Unsupported GGUF tensor type: ${type}`);
    }
    const offset = await cursor.u64();
    if (offset % BigInt(alignment) !== 0n) {
      throw new Error(`GGUF tensor offset for ${name} violates alignment`);
    }
    tensors.push({
      name,
      dimensions,
      type: type as GgmlType,
      offset,
    });
  }

  const dataOffset = align(cursor.offset, alignment);
  if (dataOffset > reader.size) {
    throw new Error("Truncated GGUF input before tensor data");
  }
  for (const tensor of tensors) {
    if (dataOffset + tensor.offset > reader.size) {
      throw new Error(`GGUF tensor offset for ${tensor.name} exceeds file size`);
    }
  }

  return { version: 3, metadata, tensors, alignment, dataOffset };
}
