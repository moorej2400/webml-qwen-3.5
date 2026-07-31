export interface AllocationReservation {
  readonly id: string;
  readonly category: string;
  readonly bytes: bigint;
}

export interface AllocationLedgerSnapshot {
  readonly limitBytes: bigint;
  readonly currentBytes: bigint;
  readonly peakBytes: bigint;
  readonly currentByCategory: Readonly<Record<string, bigint>>;
  readonly allocationCount: number;
}

export class AllocationLedger {
  readonly #limitBytes: bigint;
  readonly #active = new Map<string, AllocationReservation>();
  readonly #usedIds = new Set<string>();
  readonly #currentByCategory = new Map<string, bigint>();
  #currentBytes = 0n;
  #peakBytes = 0n;

  constructor(limitBytes: bigint) {
    if (limitBytes <= 0n) {
      throw new Error("Allocation ledger limit must be greater than zero");
    }
    this.#limitBytes = limitBytes;
  }

  reserve(reservation: AllocationReservation): void {
    if (reservation.id.length === 0) {
      throw new Error("Allocation id must not be empty");
    }
    if (reservation.category.length === 0) {
      throw new Error("Allocation category must not be empty");
    }
    if (reservation.bytes <= 0n) {
      throw new Error("Allocation bytes must be greater than zero");
    }
    if (this.#usedIds.has(reservation.id)) {
      throw new Error(`Duplicate allocation id: ${reservation.id}`);
    }
    const nextBytes = this.#currentBytes + reservation.bytes;
    if (nextBytes > this.#limitBytes) {
      throw new Error(
        `Allocation would exceed ledger limit of ${this.#limitBytes} bytes`,
      );
    }

    // Mutation starts only after all checks pass, so a rejected reservation
    // cannot leave category totals or ownership in a partial state.
    this.#active.set(reservation.id, { ...reservation });
    this.#usedIds.add(reservation.id);
    this.#currentBytes = nextBytes;
    this.#peakBytes =
      this.#peakBytes > this.#currentBytes
        ? this.#peakBytes
        : this.#currentBytes;
    this.#currentByCategory.set(
      reservation.category,
      (this.#currentByCategory.get(reservation.category) ?? 0n) +
        reservation.bytes,
    );
  }

  release(id: string): void {
    const reservation = this.#active.get(id);
    if (reservation === undefined) {
      throw new Error(`Allocation ${id} is not reserved or was already released`);
    }
    this.#active.delete(id);
    this.#currentBytes -= reservation.bytes;
    const categoryBytes =
      this.#currentByCategory.get(reservation.category)! - reservation.bytes;
    if (categoryBytes === 0n) {
      this.#currentByCategory.delete(reservation.category);
    } else {
      this.#currentByCategory.set(reservation.category, categoryBytes);
    }
  }

  snapshot(): AllocationLedgerSnapshot {
    return {
      limitBytes: this.#limitBytes,
      currentBytes: this.#currentBytes,
      peakBytes: this.#peakBytes,
      currentByCategory: Object.fromEntries(
        [...this.#currentByCategory].sort(([left], [right]) =>
          left < right ? -1 : left > right ? 1 : 0,
        ),
      ),
      allocationCount: this.#active.size,
    };
  }

  assertAllReleased(): void {
    if (this.#active.size === 0) {
      return;
    }
    const allocations = [...this.#active.values()]
      .sort((left, right) =>
        left.id < right.id ? -1 : left.id > right.id ? 1 : 0,
      )
      .map(({ id, bytes }) => `${id}: ${bytes} bytes`)
      .join(", ");
    throw new Error(`GPU allocations remain live: ${allocations}`);
  }
}
