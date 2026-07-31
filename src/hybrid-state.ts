export type Qwen35HybridResourceKind = "kv-pair" | "conv" | "recurrent";

export interface Qwen35HybridResourcePlan {
  readonly layer: number;
  readonly kind: Qwen35HybridResourceKind;
  readonly bytes: bigint;
}

export interface Qwen35HybridStatePlan {
  readonly capacity: number;
  readonly resources: readonly Qwen35HybridResourcePlan[];
  readonly totalBytes: bigint;
}

export interface Qwen35HybridStateArena {
  allocate(request: GpuAllocationRequest): Promise<GpuAllocation>;
}

export interface Qwen35HybridStateResource
  extends Qwen35HybridResourcePlan {
  readonly id: string;
}

export interface CreateQwen35HybridStateOptions {
  readonly arena: Qwen35HybridStateArena;
  readonly capacity: number;
  readonly clearAllocation: (
    allocation: GpuAllocation,
    resource: Qwen35HybridStateResource,
  ) => Promise<void>;
}

export const QWEN35_FULL_ATTENTION_LAYERS = Object.freeze([
  3, 7, 11, 15, 19, 23, 27, 31,
] as const);

const QWEN35_LAYER_COUNT = 32;
const KV_HEAD_COUNT = 4n;
const ATTENTION_HEAD_DIMENSION = 256n;
const K_AND_V = 2n;
const FP16_BYTES = 2n;
const DELTANET_CONV_BYTES = 8_192n * 4n * 4n;
const DELTANET_RECURRENT_BYTES = 32n * 128n * 128n * 4n;

function requireCapacity(capacity: number): void {
  if (!Number.isSafeInteger(capacity) || capacity < 1 || capacity > 16_384) {
    throw new Error("Hybrid state capacity must be from 1 through 16384");
  }
}

/**
 * Plans only persistent Qwen3.5 state. Full-attention layers own one packed
 * K/V pair; linear layers own FP32 convolution and recurrent resources.
 */
export function planQwen35HybridState(
  capacity: number,
): Qwen35HybridStatePlan {
  requireCapacity(capacity);
  const fullLayers = new Set<number>(QWEN35_FULL_ATTENTION_LAYERS);
  const resources: Qwen35HybridResourcePlan[] = [];
  const kvPairBytes =
    BigInt(capacity) *
    KV_HEAD_COUNT *
    ATTENTION_HEAD_DIMENSION *
    K_AND_V *
    FP16_BYTES;

  for (let layer = 0; layer < QWEN35_LAYER_COUNT; layer += 1) {
    if (fullLayers.has(layer)) {
      resources.push(Object.freeze({
        layer,
        kind: "kv-pair",
        bytes: kvPairBytes,
      }));
      continue;
    }
    resources.push(
      Object.freeze({
        layer,
        kind: "conv",
        bytes: DELTANET_CONV_BYTES,
      }),
      Object.freeze({
        layer,
        kind: "recurrent",
        bytes: DELTANET_RECURRENT_BYTES,
      }),
    );
  }

  return Object.freeze({
    capacity,
    resources: Object.freeze(resources),
    totalBytes: resources.reduce((sum, resource) => sum + resource.bytes, 0n),
  });
}

interface OwnedHybridResource {
  readonly plan: Qwen35HybridStateResource;
  readonly allocation: GpuAllocation;
}

const GPU_STORAGE_AND_COPY_DST = 0x0080 | 0x0008;

function resourceId(resource: Qwen35HybridResourcePlan): string {
  return `hybrid-state-layer-${resource.layer}-${resource.kind}`;
}

export class Qwen35HybridState {
  readonly #capacity: number;
  readonly #byteLength: bigint;
  readonly #resources: readonly OwnedHybridResource[];
  readonly #clearAllocation: CreateQwen35HybridStateOptions["clearAllocation"];
  #position = 0;
  #disposed = false;
  #poisoned = false;

  constructor(
    plan: Qwen35HybridStatePlan,
    resources: readonly OwnedHybridResource[],
    clearAllocation: CreateQwen35HybridStateOptions["clearAllocation"],
  ) {
    this.#capacity = plan.capacity;
    this.#byteLength = plan.totalBytes;
    this.#resources = resources;
    this.#clearAllocation = clearAllocation;
  }

  get capacity(): number {
    return this.#capacity;
  }

  get position(): number {
    return this.#position;
  }

  get byteLength(): bigint {
    return this.#byteLength;
  }

  get resourceCount(): number {
    return this.#resources.length;
  }

  advance(tokens: number): { readonly start: number; readonly end: number } {
    this.#assertLive();
    if (!Number.isSafeInteger(tokens) || tokens < 1) {
      throw new Error("Hybrid state advance must be a positive safe integer");
    }
    const end = this.#position + tokens;
    if (end > this.#capacity) {
      throw new Error("Hybrid state advance exceeds context capacity");
    }
    const range = Object.freeze({ start: this.#position, end });
    this.#position = end;
    return range;
  }

  async reset(): Promise<void> {
    this.#assertLive();
    try {
      for (const resource of this.#resources) {
        await this.#clearAllocation(resource.allocation, resource.plan);
      }
    } catch {
      // Keep the old position when clearing fails: callers must not reuse a
      // partially reset recurrent state as either old or fresh conversation.
      this.#poisoned = true;
      throw new Error("Hybrid state reset failed");
    }
    this.#position = 0;
  }

  dispose(): void {
    if (this.#disposed) {
      return;
    }
    this.#disposed = true;
    let firstError: unknown;
    // Reverse creation order keeps rollback and normal disposal ownership
    // identical if the arena later gains dependent suballocations.
    for (let index = this.#resources.length - 1; index >= 0; index -= 1) {
      try {
        this.#resources[index]!.allocation.destroy();
      } catch (error) {
        firstError ??= error;
      }
    }
    if (firstError !== undefined) {
      throw new Error("Hybrid state disposal failed");
    }
  }

  #assertLive(): void {
    if (this.#disposed) {
      throw new Error("Hybrid state is disposed");
    }
    if (this.#poisoned) {
      throw new Error("Hybrid state is poisoned and must be disposed");
    }
  }
}

/**
 * Allocates each logical state independently so GpuArena can shard any large
 * K/V pair without inventing dummy resources for the other layer family.
 */
export async function createQwen35HybridState(
  options: CreateQwen35HybridStateOptions,
): Promise<Qwen35HybridState> {
  const plan = planQwen35HybridState(options.capacity);
  const owned: OwnedHybridResource[] = [];
  try {
    for (const resource of plan.resources) {
      const plannedResource = Object.freeze({
        ...resource,
        id: resourceId(resource),
      });
      const allocation = await options.arena.allocate({
        id: plannedResource.id,
        category:
          resource.kind === "kv-pair" ? "kv-cache" : "activation",
        byteLength: resource.bytes,
        usage: GPU_STORAGE_AND_COPY_DST,
        alignment: 4,
        requiredShardQuantumBytes: 4n,
      });
      owned.push(Object.freeze({
        plan: plannedResource,
        allocation,
      }));
    }
  } catch {
    for (let index = owned.length - 1; index >= 0; index -= 1) {
      try {
        owned[index]!.allocation.destroy();
      } catch {
        // The public diagnostic remains stable; the arena records any detailed
        // buffer cleanup evidence without exposing allocation identifiers.
      }
    }
    throw new Error("Hybrid state allocation failed");
  }
  return new Qwen35HybridState(plan, Object.freeze(owned), options.clearAllocation);
}
import type {
  GpuAllocation,
  GpuAllocationRequest,
} from "./gpu-arena.js";
