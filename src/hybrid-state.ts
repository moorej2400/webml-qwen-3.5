export type Qwen35HybridResourceKind =
  | "key"
  | "value"
  | "conv"
  | "recurrent";

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

/**
 * A binding view preserves the physical buffer identity but omits allocation
 * ownership methods. Only Qwen35HybridState may destroy the buffer.
 */
export interface Qwen35HybridBufferView {
  /** Opaque GPUBuffer identity; ownership methods are not part of this API. */
  readonly buffer: object;
  /** Offset within this logical state resource, not within the model package. */
  readonly logicalByteOffset: bigint;
  readonly logicalByteLength: bigint;
  /** Physical buffer extent, including any alignment padding. */
  readonly allocatedByteLength: bigint;
}

export interface Qwen35HybridResourceView
  extends Qwen35HybridStateResource {
  readonly byteLength: bigint;
  /**
   * Physical pages for a page-aware scheduler. Existing single-page kernels
   * must not treat this list as one bindable range.
   */
  readonly shards: readonly Qwen35HybridBufferView[];
}

export type Qwen35HybridLayerResources =
  | Readonly<{
      layer: number;
      kind: "full-attention";
      key: Qwen35HybridResourceView;
      value: Qwen35HybridResourceView;
    }>
  | Readonly<{
      layer: number;
      kind: "gated-deltanet";
      conv: Qwen35HybridResourceView;
      recurrent: Qwen35HybridResourceView;
    }>;

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
const FP16_BYTES = 2n;
const KV_TOKEN_ROW_BYTES = KV_HEAD_COUNT * ATTENTION_HEAD_DIMENSION * FP16_BYTES;
const DELTANET_CONV_ROW_BYTES = 4n * 4n;
const DELTANET_RECURRENT_ROW_BYTES = 128n * 4n;
const DELTANET_CONV_BYTES = 8_192n * 4n * 4n;
const DELTANET_RECURRENT_BYTES = 32n * 128n * 128n * 4n;

function requireCapacity(capacity: number): void {
  if (!Number.isSafeInteger(capacity) || capacity < 1 || capacity > 16_384) {
    throw new Error("Hybrid state capacity must be from 1 through 16384");
  }
}

/**
 * Plans only persistent Qwen3.5 state. Full-attention layers own independently
 * bindable K and V allocations; linear layers own FP32 state allocations.
 */
export function planQwen35HybridState(
  capacity: number,
): Qwen35HybridStatePlan {
  requireCapacity(capacity);
  const fullLayers = new Set<number>(QWEN35_FULL_ATTENTION_LAYERS);
  const resources: Qwen35HybridResourcePlan[] = [];
  const kvBytes = BigInt(capacity) * KV_TOKEN_ROW_BYTES;

  for (let layer = 0; layer < QWEN35_LAYER_COUNT; layer += 1) {
    if (fullLayers.has(layer)) {
      resources.push(
        Object.freeze({ layer, kind: "key", bytes: kvBytes }),
        Object.freeze({ layer, kind: "value", bytes: kvBytes }),
      );
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
  readonly view: Qwen35HybridResourceView;
}

const GPU_STORAGE_AND_COPY_DST = 0x0080 | 0x0008;

function resourceId(resource: Qwen35HybridResourcePlan): string {
  return `hybrid-state-layer-${resource.layer}-${resource.kind}`;
}

function resourceKey(layer: number, kind: Qwen35HybridResourceKind): string {
  return `${layer}:${kind}`;
}

function createResourceView(
  plan: Qwen35HybridStateResource,
  allocation: GpuAllocation,
): Qwen35HybridResourceView {
  if (allocation.logicalBytes !== plan.bytes) {
    throw new Error("Hybrid state allocation does not match its resource plan");
  }
  let expectedOffset = 0n;
  let allocatedBytes = 0n;
  const shards = allocation.shards.map((shard) => {
    if (
      shard.logicalByteOffset !== expectedOffset ||
      shard.logicalByteLength <= 0n ||
      shard.allocatedByteLength < shard.logicalByteLength
    ) {
      throw new Error("Hybrid state allocation has an invalid shard layout");
    }
    expectedOffset += shard.logicalByteLength;
    allocatedBytes += shard.allocatedByteLength;
    return Object.freeze({
      buffer: shard.buffer as object,
      logicalByteOffset: shard.logicalByteOffset,
      logicalByteLength: shard.logicalByteLength,
      allocatedByteLength: shard.allocatedByteLength,
    });
  });
  if (
    expectedOffset !== plan.bytes ||
    allocatedBytes !== allocation.allocatedBytes
  ) {
    throw new Error("Hybrid state allocation has incomplete shard coverage");
  }
  return Object.freeze({
    ...plan,
    byteLength: plan.bytes,
    shards: Object.freeze(shards),
  });
}

export class Qwen35HybridState {
  readonly #capacity: number;
  readonly #byteLength: bigint;
  readonly #resources: readonly OwnedHybridResource[];
  readonly #resourceViews: ReadonlyMap<string, Qwen35HybridResourceView>;
  readonly #layerViews: readonly Qwen35HybridLayerResources[];
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
    this.#resourceViews = new Map(
      resources.map(({ plan: resource, view }) => [
        resourceKey(resource.layer, resource.kind),
        view,
      ]),
    );
    const fullLayers = new Set<number>(QWEN35_FULL_ATTENTION_LAYERS);
    this.#layerViews = Object.freeze(
      Array.from({ length: QWEN35_LAYER_COUNT }, (_, layer) =>
        fullLayers.has(layer)
          ? Object.freeze({
              layer,
              kind: "full-attention" as const,
              key: this.#requiredResourceView(layer, "key"),
              value: this.#requiredResourceView(layer, "value"),
            })
          : Object.freeze({
              layer,
              kind: "gated-deltanet" as const,
              conv: this.#requiredResourceView(layer, "conv"),
              recurrent: this.#requiredResourceView(layer, "recurrent"),
            }),
      ),
    );
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

  getResource(
    layer: number,
    kind: Qwen35HybridResourceKind,
  ): Qwen35HybridResourceView {
    this.#assertLive();
    requireLayer(layer);
    const resource = this.#resourceViews.get(resourceKey(layer, kind));
    if (resource === undefined) {
      throw new Error("Hybrid state resource does not exist for this layer");
    }
    return resource;
  }

  getLayerResources(layer: number): Qwen35HybridLayerResources {
    this.#assertLive();
    requireLayer(layer);
    return this.#layerViews[layer]!;
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

  #requiredResourceView(
    layer: number,
    kind: Qwen35HybridResourceKind,
  ): Qwen35HybridResourceView {
    const resource = this.#resourceViews.get(resourceKey(layer, kind));
    if (resource === undefined) {
      throw new Error("Hybrid state plan is incomplete");
    }
    return resource;
  }
}

function requireLayer(layer: number): void {
  if (!Number.isSafeInteger(layer) || layer < 0 || layer >= QWEN35_LAYER_COUNT) {
    throw new Error("Hybrid state layer must be from 0 through 31");
  }
}

/**
 * Allocates each logical state independently so GpuArena can adapt physical
 * shards without coupling K and V or inventing unused layer resources.
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
          resource.kind === "key" || resource.kind === "value"
            ? "kv-cache"
            : "activation",
        byteLength: resource.bytes,
        usage: GPU_STORAGE_AND_COPY_DST,
        alignment: 4,
        requiredShardQuantumBytes:
          resource.kind === "key" || resource.kind === "value"
            ? KV_TOKEN_ROW_BYTES
            : resource.kind === "conv"
              ? DELTANET_CONV_ROW_BYTES
              : DELTANET_RECURRENT_ROW_BYTES,
      });
      let view: Qwen35HybridResourceView;
      try {
        view = createResourceView(plannedResource, allocation);
      } catch {
        try {
          allocation.destroy();
        } catch {
          // The state cannot expose a malformed allocation, and its ownership
          // still ends here even when the device reports a cleanup failure.
        }
        throw new Error("Hybrid state allocation has invalid shard metadata");
      }
      owned.push(Object.freeze({
        plan: plannedResource,
        allocation,
        view,
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
