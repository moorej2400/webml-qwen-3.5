import {
  allocationDiagnosticError,
  diagnosticError,
} from "./diagnostics.js";
import type { GpuAllocation, GpuAllocationRequest } from "./gpu-arena.js";

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
  readonly onProgress?: (completedBytes: number) => void;
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
  readonly allocations: GpuAllocation[];
  readonly shards: Qwen35HybridBufferView[];
}

interface StagedHybridAllocation {
  readonly plan: Qwen35HybridStateResource;
  readonly allocation: GpuAllocation;
  readonly logicalBase: bigint;
  readonly views: readonly Qwen35HybridBufferView[];
}

const GPU_STORAGE_AND_COPY_DST = 0x0080 | 0x0008;
export const QWEN35_HYBRID_STATE_PAGE_TOKENS = 256;

function resourceId(resource: Qwen35HybridResourcePlan): string {
  return `hybrid-state-layer-${resource.layer}-${resource.kind}`;
}

function resourceKey(layer: number, kind: Qwen35HybridResourceKind): string {
  return `${layer}:${kind}`;
}

function createAllocationViews(
  plan: Qwen35HybridStateResource,
  allocation: GpuAllocation,
  logicalBase: bigint,
): readonly Qwen35HybridBufferView[] {
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
      logicalByteOffset: logicalBase + shard.logicalByteOffset,
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
  return Object.freeze(shards);
}

export class Qwen35HybridState {
  readonly #capacity: number;
  readonly #arena: Qwen35HybridStateArena;
  readonly #resources = new Map<string, OwnedHybridResource>();
  readonly #resourceViews = new Map<string, Qwen35HybridResourceView>();
  readonly #layerViews = new Map<number, Qwen35HybridLayerResources>();
  readonly #allocationOrder: GpuAllocation[] = [];
  readonly #clearAllocation: CreateQwen35HybridStateOptions["clearAllocation"];
  readonly #onProgress: CreateQwen35HybridStateOptions["onProgress"];
  #residentCapacity = 0;
  #byteLength = 0n;
  #position = 0;
  #disposed = false;
  #poisoned = false;
  #growthActive = false;

  constructor(
    options: CreateQwen35HybridStateOptions,
  ) {
    requireCapacity(options.capacity);
    this.#capacity = options.capacity;
    this.#arena = options.arena;
    this.#clearAllocation = options.clearAllocation;
    this.#onProgress = options.onProgress;
  }

  get capacity(): number {
    return this.#capacity;
  }

  get position(): number {
    return this.#position;
  }

  get residentCapacity(): number {
    return this.#residentCapacity;
  }

  get byteLength(): bigint {
    return this.#byteLength;
  }

  get resourceCount(): number {
    return this.#resources.size;
  }

  /**
   * Makes the requested logical prefix resident without allocating the unused
   * remainder of the 16K product context. A growth commits only after every
   * page succeeds, so cancellation leaves the prior prefix reusable.
   */
  async ensureCapacity(
    requiredEnd: number,
    signal?: AbortSignal,
  ): Promise<void> {
    this.#assertLive();
    if (
      !Number.isSafeInteger(requiredEnd) ||
      requiredEnd < 1 ||
      requiredEnd > this.#capacity
    ) {
      throw new Error(
        `Hybrid state resident capacity must be from 1 through ${this.#capacity}`,
      );
    }
    if (requiredEnd <= this.#residentCapacity) return;
    if (this.#growthActive) {
      throw new Error("Hybrid state growth is already active");
    }

    const targetCapacity = Math.min(
      this.#capacity,
      Math.ceil(requiredEnd / QWEN35_HYBRID_STATE_PAGE_TOKENS) *
        QWEN35_HYBRID_STATE_PAGE_TOKENS,
    );
    const staged: StagedHybridAllocation[] = [];
    this.#growthActive = true;
    try {
      signal?.throwIfAborted();
      if (this.#residentCapacity === 0) {
        for (const resource of planQwen35HybridState(this.#capacity).resources) {
          if (resource.kind === "key" || resource.kind === "value") continue;
          await this.#stageAllocation(staged, resource, 0n, signal);
        }
      }
      const firstPage = this.#residentCapacity /
        QWEN35_HYBRID_STATE_PAGE_TOKENS;
      const pageCount = targetCapacity / QWEN35_HYBRID_STATE_PAGE_TOKENS;
      for (let page = firstPage; page < pageCount; page += 1) {
        for (const layer of QWEN35_FULL_ATTENTION_LAYERS) {
          for (const kind of ["key", "value"] as const) {
            await this.#stageAllocation(
              staged,
              Object.freeze({
                layer,
                kind,
                bytes: BigInt(QWEN35_HYBRID_STATE_PAGE_TOKENS) *
                  KV_TOKEN_ROW_BYTES,
              }),
              BigInt(page * QWEN35_HYBRID_STATE_PAGE_TOKENS) *
                KV_TOKEN_ROW_BYTES,
              signal,
              page,
            );
          }
        }
      }
      signal?.throwIfAborted();
      this.#commitGrowth(staged, targetCapacity);
    } catch (error) {
      const rollbackFailed = destroyStagedReverse(staged) !== undefined;
      if (rollbackFailed) {
        this.#poisoned = true;
        throw diagnosticError("unknown", "Hybrid state allocation failed");
      }
      if (error instanceof DOMException && error.name === "AbortError") {
        throw error;
      }
      let code: unknown;
      let hasCode = false;
      if (
        (typeof error === "object" && error !== null) ||
        typeof error === "function"
      ) {
        try {
          hasCode = Object.hasOwn(error, "code");
          if (hasCode) code = (error as { readonly code?: unknown }).code;
        } catch {
          hasCode = true;
          code = undefined;
        }
      }
      throw hasCode
        ? allocationDiagnosticError(code)
        : diagnosticError("unknown", "Hybrid state allocation failed");
    } finally {
      this.#growthActive = false;
    }
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
    const view = this.#layerViews.get(layer);
    if (view === undefined) {
      throw new Error("Hybrid state plan is incomplete");
    }
    return view;
  }

  advance(tokens: number): { readonly start: number; readonly end: number } {
    this.#assertLive();
    if (!Number.isSafeInteger(tokens) || tokens < 1) {
      throw new Error("Hybrid state advance must be a positive safe integer");
    }
    const end = this.#position + tokens;
    if (end > this.#residentCapacity) {
      throw new Error(
        "Hybrid state advance exceeds resident capacity; allocate the range first",
      );
    }
    const range = Object.freeze({ start: this.#position, end });
    this.#position = end;
    return range;
  }

  async reset(): Promise<void> {
    this.#assertLive();
    try {
      for (const resource of this.#resources.values()) {
        for (const allocation of resource.allocations) {
          await this.#clearAllocation(allocation, resource.plan);
        }
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
    for (let index = this.#allocationOrder.length - 1; index >= 0; index -= 1) {
      try {
        this.#allocationOrder[index]!.destroy();
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

  #resourceView(resource: OwnedHybridResource): Qwen35HybridResourceView {
    const byteLength = resource.shards.reduce(
      (total, shard) => total + shard.logicalByteLength,
      0n,
    );
    return Object.freeze({
      ...resource.plan,
      bytes: byteLength,
      byteLength,
      shards: Object.freeze([...resource.shards]),
    });
  }

  async #stageAllocation(
    staged: StagedHybridAllocation[],
    resource: Qwen35HybridResourcePlan,
    logicalBase: bigint,
    signal?: AbortSignal,
    page?: number,
  ): Promise<void> {
    signal?.throwIfAborted();
    const publicId = resourceId(resource);
    const plan = Object.freeze({
      ...resource,
      id: page === undefined ? publicId : `${publicId}-page-${page}`,
    });
    const allocation = await this.#arena.allocate({
      id: plan.id,
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
    let views: readonly Qwen35HybridBufferView[];
    try {
      views = createAllocationViews(plan, allocation, logicalBase);
    } catch {
      try {
        allocation.destroy();
      } catch {
        this.#poisoned = true;
      }
      throw allocationDiagnosticError("state_metadata");
    }
    // Snapshot caller-owned arena metadata so publication never depends on a
    // later read of mutable allocation fields.
    staged.push(Object.freeze({ plan, allocation, logicalBase, views }));
    if (this.#disposed) {
      throw new Error("Hybrid state was disposed during growth");
    }
    signal?.throwIfAborted();
    try {
      this.#onProgress?.(Number(
        this.#byteLength + staged.reduce(
          (total, item) => total + item.plan.bytes,
          0n,
        ),
      ));
    } catch {
      // Telemetry is outside state ownership; classify it separately so a
      // callback defect is never mistaken for a WebGPU allocation failure.
      throw allocationDiagnosticError("state_progress");
    }
  }

  #commitGrowth(
    staged: readonly StagedHybridAllocation[],
    targetCapacity: number,
  ): void {
    this.#assertLive();
    // Validate the complete staged transaction before changing any live map.
    // JavaScript cannot interleave caller mutation with the synchronous commit
    // that follows, and publication uses only the already-frozen views.
    for (const item of staged) {
      const current = createAllocationViews(
        item.plan,
        item.allocation,
        item.logicalBase,
      );
      if (
        current.length !== item.views.length ||
        current.some((view, index) => {
          const snapshot = item.views[index];
          return snapshot === undefined || view.buffer !== snapshot.buffer ||
            view.logicalByteOffset !== snapshot.logicalByteOffset ||
            view.logicalByteLength !== snapshot.logicalByteLength ||
            view.allocatedByteLength !== snapshot.allocatedByteLength;
        })
      ) {
        throw new Error("Hybrid state allocation changed before commit");
      }
    }
    const changedKeys = new Set<string>();
    for (const item of staged) {
      const key = resourceKey(item.plan.layer, item.plan.kind);
      let resource = this.#resources.get(key);
      if (resource === undefined) {
        resource = {
          plan: Object.freeze({ ...item.plan, id: resourceId(item.plan) }),
          allocations: [],
          shards: [],
        };
        this.#resources.set(key, resource);
      }
      resource.allocations.push(item.allocation);
      resource.shards.push(...item.views);
      this.#allocationOrder.push(item.allocation);
      changedKeys.add(key);
      this.#byteLength += item.plan.bytes;
    }
    for (const key of changedKeys) {
      this.#resourceViews.set(key, this.#resourceView(this.#resources.get(key)!));
    }
    const fullLayers = new Set<number>(QWEN35_FULL_ATTENTION_LAYERS);
    for (let layer = 0; layer < QWEN35_LAYER_COUNT; layer += 1) {
      const prior = this.#layerViews.get(layer);
      if (
        prior !== undefined &&
        !changedKeys.has(resourceKey(layer, "key")) &&
        !changedKeys.has(resourceKey(layer, "value")) &&
        !changedKeys.has(resourceKey(layer, "conv")) &&
        !changedKeys.has(resourceKey(layer, "recurrent"))
      ) {
        continue;
      }
      this.#layerViews.set(
        layer,
        fullLayers.has(layer)
          ? Object.freeze({
              layer,
              kind: "full-attention",
              key: this.#requiredResourceView(layer, "key"),
              value: this.#requiredResourceView(layer, "value"),
            })
          : Object.freeze({
              layer,
              kind: "gated-deltanet",
              conv: this.#requiredResourceView(layer, "conv"),
              recurrent: this.#requiredResourceView(layer, "recurrent"),
            }),
      );
    }
    this.#residentCapacity = targetCapacity;
  }
}

function destroyStagedReverse(
  staged: readonly StagedHybridAllocation[],
): unknown {
  let firstError: unknown;
  for (let index = staged.length - 1; index >= 0; index -= 1) {
    try {
      staged[index]!.allocation.destroy();
    } catch (error) {
      firstError ??= error;
    }
  }
  return firstError;
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
  return new Qwen35HybridState(options);
}
