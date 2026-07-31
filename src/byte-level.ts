const byteToCodePoint = new Uint32Array(256);
const codePointToByte = new Map<number, number>();

const visibleBytes = [
  ...range(33, 126),
  ...range(161, 172),
  ...range(174, 255),
];
const assigned = new Set(visibleBytes);
let extraCodePoint = 256;

for (const byte of visibleBytes) {
  byteToCodePoint[byte] = byte;
  codePointToByte.set(byte, byte);
}
for (let byte = 0; byte < 256; byte += 1) {
  if (assigned.has(byte)) {
    continue;
  }
  byteToCodePoint[byte] = extraCodePoint;
  codePointToByte.set(extraCodePoint, byte);
  extraCodePoint += 1;
}

function range(start: number, end: number): number[] {
  const values: number[] = [];
  for (let value = start; value <= end; value += 1) {
    values.push(value);
  }
  return values;
}

/** Maps one UTF-8 byte to the reversible alphabet used by ByteLevel BPE. */
export function byteToByteLevelCharacter(byte: number): string {
  if (!Number.isInteger(byte) || byte < 0 || byte > 255) {
    throw new Error("ByteLevel input must be an unsigned byte");
  }
  return String.fromCodePoint(byteToCodePoint[byte]!);
}

/** Converts a ByteLevel vocabulary spelling back to the represented raw bytes. */
export function byteLevelStringToBytes(value: string): Uint8Array {
  const bytes: number[] = [];
  for (const character of value) {
    const byte = codePointToByte.get(character.codePointAt(0)!);
    if (byte === undefined) {
      throw new Error("Tokenizer vocabulary contains a non-ByteLevel character");
    }
    bytes.push(byte);
  }
  return Uint8Array.from(bytes);
}
