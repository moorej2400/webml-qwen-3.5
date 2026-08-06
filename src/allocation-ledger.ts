import { diagnosticError, isSafeDiagnosticCode } from "./diagnostics.js";

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

const ALLOCATION_HANDLE = Symbol("allocation-handle");

// The opaque token binds release authority to one reservation generation.
// Stable string IDs may then be reused without allowing stale releases.
export interface AllocationHandle {
  readonly [ALLOCATION_HANDLE]: true;
}

interface ActiveAllocation {
  readonly reservation: AllocationReservation;
  readonly generation: bigint;
  readonly handle: AllocationHandle;
}

export class AllocationLedger {
  readonly #limitBytes: bigint;
  readonly #active = new Map<string, ActiveAllocation>();
  readonly #handles = new WeakMap<
    AllocationHandle,
    { readonly id: string; readonly generation: bigint }
  >();
  readonly #currentByCategory = new Map<string, bigint>();
  #nextGeneration = 1n;
  #currentBytes = 0n;
  #peakBytes = 0n;

  constructor(limitBytes: bigint) {
    if (limitBytes <= 0n) {
      throw new Error("Allocation ledger limit must be greater than zero");
    }
    this.#limitBytes = limitBytes;
  }

  reserve(reservation: AllocationReservation): AllocationHandle {
    if (reservation.id.length === 0) {
      throw new Error("Allocation id must not be empty");
    }
    if (!isSafeDiagnosticCode(reservation.category)) {
      throw diagnosticError(
        "ALLOCATION_CATEGORY_INVALID",
        "Allocation category is invalid",
      );
    }
    if (reservation.bytes <= 0n) {
      throw new Error("Allocation bytes must be greater than zero");
    }
    if (this.#active.has(reservation.id)) {
      throw diagnosticError(
        "ALLOCATION_DUPLICATE",
        "Duplicate allocation id is already active",
      );
    }
    const nextBytes = this.#currentBytes + reservation.bytes;
    if (nextBytes > this.#limitBytes) {
      throw diagnosticError(
        "ALLOCATION_LIMIT_EXCEEDED",
        "Allocation would exceed ledger limit",
      );
    }

    // Mutation starts only after all checks pass, so a rejected reservation
    // cannot leave category totals or ownership in a partial state.
    const generation = this.#nextGeneration;
    this.#nextGeneration += 1n;
    const handle = Object.freeze({
      [ALLOCATION_HANDLE]: true as const,
    });
    this.#active.set(reservation.id, {
      reservation: { ...reservation },
      generation,
      handle,
    });
    this.#handles.set(handle, { id: reservation.id, generation });
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
    return handle;
  }

  release(handle: AllocationHandle): void {
    const identity = this.#handles.get(handle);
    const active =
      identity === undefined ? undefined : this.#active.get(identity.id);
    if (
      identity === undefined ||
      active === undefined ||
      active.generation !== identity.generation ||
      active.handle !== handle
    ) {
      throw diagnosticError(
        "ALLOCATION_HANDLE_STALE",
        "Allocation handle is stale or already released",
      );
    }
    const { reservation } = active;
    this.#active.delete(identity.id);
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
    const counts = new Map<string, number>();
    for (const { reservation } of this.#active.values()) {
      counts.set(
        reservation.category,
        (counts.get(reservation.category) ?? 0) + 1,
      );
    }
    const categories = [...counts]
      .sort(([left], [right]) =>
        left < right ? -1 : left > right ? 1 : 0,
      )
      .map(([category, count]) => `${category}:${count}`)
      .join(", ");
    throw diagnosticError(
      "GPU_RESOURCE_LEAK",
      `${this.#active.size} active GPU allocation(s); categories ${categories}`,
    );
  }
}
