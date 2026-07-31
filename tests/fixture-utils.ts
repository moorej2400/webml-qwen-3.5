export class BinaryWriter {
  readonly bytes: number[] = [];

  u8(value: number): this {
    this.bytes.push(value & 0xff);
    return this;
  }

  u16(value: number): this {
    return this.rawUint(value, 2);
  }

  u32(value: number): this {
    return this.rawUint(value, 4);
  }

  u64(value: bigint | number): this {
    let remaining = BigInt(value);
    for (let index = 0; index < 8; index += 1) {
      this.bytes.push(Number(remaining & 0xffn));
      remaining >>= 8n;
    }
    return this;
  }

  f32(value: number): this {
    const bytes = new Uint8Array(4);
    new DataView(bytes.buffer).setFloat32(0, value, true);
    return this.bytesFrom(bytes);
  }

  string(value: string): this {
    const encoded = new TextEncoder().encode(value);
    this.u64(encoded.length);
    return this.bytesFrom(encoded);
  }

  bytesFrom(value: Uint8Array | number[]): this {
    this.bytes.push(...value);
    return this;
  }

  pad(alignment: number): this {
    while (this.bytes.length % alignment !== 0) {
      this.u8(0);
    }
    return this;
  }

  build(): Uint8Array {
    return Uint8Array.from(this.bytes);
  }

  private rawUint(value: number, byteLength: number): this {
    let remaining = value;
    for (let index = 0; index < byteLength; index += 1) {
      this.bytes.push(remaining & 0xff);
      remaining = Math.floor(remaining / 256);
    }
    return this;
  }
}

export function memoryReader(bytes: Uint8Array): {
  size: bigint;
  reads: Array<{ offset: bigint; length: number }>;
  read(offset: bigint, length: number): Promise<Uint8Array>;
} {
  const reads: Array<{ offset: bigint; length: number }> = [];
  return {
    size: BigInt(bytes.length),
    reads,
    async read(offset, length) {
      reads.push({ offset, length });
      const start = Number(offset);
      return bytes.slice(start, start + length);
    },
  };
}
