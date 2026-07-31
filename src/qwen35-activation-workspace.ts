import { diagnosticError } from "./diagnostics.js";
import type {
  GpuAllocation,
  GpuAllocationRequest,
} from "./gpu-arena.js";

const GPU_BUFFER_USAGE_COPY_SRC = 0x0004;
const GPU_BUFFER_USAGE_COPY_DST = 0x0008;
const GPU_BUFFER_USAGE_STORAGE = 0x0080;
const STORAGE_AND_CLEAR = GPU_BUFFER_USAGE_STORAGE | GPU_BUFFER_USAGE_COPY_DST;
const STORAGE_CLEAR_AND_READBACK =
  STORAGE_AND_CLEAR | GPU_BUFFER_USAGE_COPY_SRC;

const QWEN35_VOCABULARY_SIZE = 248_320;
const LOGITS_TILE_ROWS = 1_024;
const MATHEMATICAL_VOCABULARY_TILE_COUNT = Math.ceil(
  QWEN35_VOCABULARY_SIZE / LOGITS_TILE_ROWS,
);
// The scheduler must coalesce physical shard dispatches into one winner per
// mathematical tile before using these 256 final-reduction candidate slots.
const TOP_K_CANDIDATE_CAPACITY = 256;
const SCALAR_BYTES = 4;

export type Qwen35ActivationResourceKind =
  | "packed-embedding-output"
  | "hidden-secondary"
  | "normalized-hidden"
  | "attention-projection-primary"
  | "attention-projection-secondary"
  | "attention-inner-primary"
  | "attention-inner-secondary"
  | "full-attention-key"
  | "full-attention-value"
  | "deltanet-alpha"
  | "deltanet-beta"
  | "ffn-gate"
  | "ffn-up"
  | "ffn-product"
  | "logits-tile"
  | "top-k-scores"
  | "top-k-indices"
  | "selected-token";

export interface Qwen35ActivationResourcePlan {
  readonly kind: Qwen35ActivationResourceKind;
  readonly scalarType: "f32" | "u32";
  readonly elementCount: number;
  readonly bytes: bigint;
  readonly usage: number;
}

export interface Qwen35ActivationWorkspacePlan {
  readonly resourceCount: number;
  readonly resources: readonly Qwen35ActivationResourcePlan[];
  readonly totalBytes: bigint;
  readonly vocabularySize: 248_320;
  readonly logitsTileRows: 1_024;
  /** Shape arithmetic only; physical weight views may require more dispatches. */
  readonly mathematicalVocabularyTileCount: number;
  /** One candidate per mathematical tile after physical dispatch coalescing. */
  readonly topKCandidateCapacity: 256;
  /** Final top-1 writes a vocabulary id or this no-selection u32 sentinel. */
  readonly selectedTokenInvalidSentinel: 0xffff_ffff;
  readonly deltanetParameterLiveness: Qwen35DeltaNetParameterLiveness;
}

export type Qwen35DeltaNetParameterRole =
  | "rawAlpha"
  | "rawBeta"
  | "transformedBeta"
  | "decay";

export interface Qwen35DeltaNetParameterRange {
  readonly resource:
    | "deltanet-alpha"
    | "deltanet-beta"
    | "full-attention-key"
    | "full-attention-value";
  readonly byteOffset: number;
  readonly byteLength: number;
  readonly access: "read" | "write";
}

export interface Qwen35DeltaNetParameterLiveness {
  readonly activeLayerKind: "gated-deltanet";
  /** These buffers are borrowed only while a full-attention layer is inactive. */
  readonly borrowedFullAttentionResources: readonly [
    "full-attention-key",
    "full-attention-value",
  ];
  readonly ranges: Readonly<
    Record<Qwen35DeltaNetParameterRole, Qwen35DeltaNetParameterRange>
  >;
}

/** A bindable range without the allocation's destroy authority. */
export interface Qwen35ActivationBindingView {
  readonly buffer: object;
  readonly offset: 0;
  readonly size: number;
}

export interface Qwen35ActivationResourceView
  extends Qwen35ActivationResourcePlan {
  readonly byteLength: number;
  readonly binding: Qwen35ActivationBindingView;
}

export interface Qwen35ActivationArena {
  allocate(request: GpuAllocationRequest): Promise<GpuAllocation>;
}

export interface CreateQwen35ActivationWorkspaceOptions {
  readonly arena: Qwen35ActivationArena;
  readonly clearAllocation: (
    allocation: GpuAllocation,
    resource: Qwen35ActivationResourceView,
  ) => Promise<void>;
}

function resource(
  kind: Qwen35ActivationResourceKind,
  elementCount: number,
  scalarType: "f32" | "u32" = "f32",
): Qwen35ActivationResourcePlan {
  // Only the final vocabulary id crosses to CPU; intermediate candidates stay
  // GPU-resident even when no valid finite score survives reduction.
  const readback = kind === "selected-token";
  return Object.freeze({
    kind,
    scalarType,
    elementCount,
    bytes: BigInt(elementCount * SCALAR_BYTES),
    usage: readback ? STORAGE_CLEAR_AND_READBACK : STORAGE_AND_CLEAR,
  });
}

const DELTANET_PARAMETER_LIVENESS: Qwen35DeltaNetParameterLiveness = (() => {
  const ranges = Object.freeze({
    rawAlpha: Object.freeze({
      resource: "deltanet-alpha" as const,
      byteOffset: 0,
      byteLength: 128,
      access: "read" as const,
    }),
    rawBeta: Object.freeze({
      resource: "deltanet-beta" as const,
      byteOffset: 0,
      byteLength: 128,
      access: "read" as const,
    }),
    transformedBeta: Object.freeze({
      resource: "full-attention-key" as const,
      byteOffset: 0,
      byteLength: 128,
      access: "write" as const,
    }),
    decay: Object.freeze({
      resource: "full-attention-value" as const,
      byteOffset: 0,
      byteLength: 128,
      access: "write" as const,
    }),
  });
  return Object.freeze({
    activeLayerKind: "gated-deltanet",
    borrowedFullAttentionResources: Object.freeze([
      "full-attention-key",
      "full-attention-value",
    ] as const),
    ranges,
  });
})();

/**
 * Fixed one-token workspace for the pinned Qwen3.5 4B language program.
 * Logits scan a bounded tile; the tied packed embedding remains model-owned.
 */
const ACTIVATION_PLAN: Qwen35ActivationWorkspacePlan = (() => {
  const resources = Object.freeze([
    resource("packed-embedding-output", 2_560),
    resource("hidden-secondary", 2_560),
    resource("normalized-hidden", 2_560),
    resource("attention-projection-primary", 8_192),
    resource("attention-projection-secondary", 8_192),
    resource("attention-inner-primary", 4_096),
    resource("attention-inner-secondary", 4_096),
    resource("full-attention-key", 1_024),
    resource("full-attention-value", 1_024),
    resource("deltanet-alpha", 32),
    resource("deltanet-beta", 32),
    resource("ffn-gate", 9_216),
    resource("ffn-up", 9_216),
    resource("ffn-product", 9_216),
    resource("logits-tile", LOGITS_TILE_ROWS),
    resource("top-k-scores", TOP_K_CANDIDATE_CAPACITY),
    resource("top-k-indices", TOP_K_CANDIDATE_CAPACITY, "u32"),
    resource("selected-token", 1, "u32"),
  ]);
  return Object.freeze({
    resourceCount: resources.length,
    resources,
    totalBytes: resources.reduce((sum, item) => sum + item.bytes, 0n),
    vocabularySize: QWEN35_VOCABULARY_SIZE,
    logitsTileRows: LOGITS_TILE_ROWS,
    mathematicalVocabularyTileCount: MATHEMATICAL_VOCABULARY_TILE_COUNT,
    topKCandidateCapacity: TOP_K_CANDIDATE_CAPACITY,
    selectedTokenInvalidSentinel: 0xffff_ffff,
    deltanetParameterLiveness: DELTANET_PARAMETER_LIVENESS,
  });
})();

export function planQwen35ActivationWorkspace(): Qwen35ActivationWorkspacePlan {
  return ACTIVATION_PLAN;
}

interface OwnedResource {
  readonly allocation: GpuAllocation;
  readonly view: Qwen35ActivationResourceView;
}

function bindingView(
  plan: Qwen35ActivationResourcePlan,
  allocation: GpuAllocation,
): Qwen35ActivationResourceView {
  const shard = allocation.shards[0];
  if (
    allocation.logicalBytes !== plan.bytes ||
    allocation.shards.length !== 1 ||
    shard === undefined ||
    shard.logicalByteOffset !== 0n ||
    shard.logicalByteLength !== plan.bytes ||
    shard.allocatedByteLength < plan.bytes ||
    plan.bytes > BigInt(Number.MAX_SAFE_INTEGER)
  ) {
    throw diagnosticError(
      "activation-workspace-layout-invalid",
      "Qwen3.5 activation workspace allocation has an invalid layout",
    );
  }
  const byteLength = Number(plan.bytes);
  const binding = Object.freeze({
    buffer: shard.buffer as object,
    offset: 0 as const,
    size: byteLength,
  });
  return Object.freeze({
    ...plan,
    byteLength,
    binding,
  });
}

function destroyReverse(allocations: readonly GpuAllocation[]): unknown {
  let firstError: unknown;
  for (let index = allocations.length - 1; index >= 0; index -= 1) {
    try {
      allocations[index]!.destroy();
    } catch (error) {
      firstError ??= error;
    }
  }
  return firstError;
}

export class Qwen35ActivationWorkspace {
  readonly #owned: readonly OwnedResource[];
  readonly #byKind: ReadonlyMap<
    Qwen35ActivationResourceKind,
    Qwen35ActivationResourceView
  >;
  readonly #clearAllocation: CreateQwen35ActivationWorkspaceOptions["clearAllocation"];
  readonly resources: readonly Qwen35ActivationResourceView[];
  readonly logicalBytes: bigint;
  readonly allocatedBytes: bigint;
  #resetPromise: Promise<void> | null = null;
  #disposePromise: Promise<void> | null = null;
  #disposed = false;
  #poisoned = false;

  constructor(
    owned: readonly OwnedResource[],
    clearAllocation: CreateQwen35ActivationWorkspaceOptions["clearAllocation"],
  ) {
    this.#owned = owned;
    this.#clearAllocation = clearAllocation;
    this.resources = Object.freeze(owned.map(({ view }) => view));
    this.#byKind = new Map(this.resources.map((view) => [view.kind, view]));
    this.logicalBytes = this.resources.reduce(
      (sum, view) => sum + view.bytes,
      0n,
    );
    this.allocatedBytes = owned.reduce(
      (sum, item) => sum + item.allocation.allocatedBytes,
      0n,
    );
  }

  get resourceCount(): number {
    return this.resources.length;
  }

  get(kind: Qwen35ActivationResourceKind): Qwen35ActivationResourceView {
    this.#assertAccessible();
    const resource = this.#byKind.get(kind);
    if (resource === undefined) {
      throw diagnosticError(
        "activation-workspace-resource-missing",
        "Qwen3.5 activation workspace resource does not exist",
      );
    }
    return resource;
  }

  async reset(): Promise<void> {
    this.#assertLive();
    if (this.#resetPromise !== null) {
      return this.#resetPromise;
    }
    const operation = this.#clearResources();
    this.#resetPromise = operation;
    const retire = (): void => {
      if (this.#resetPromise === operation) this.#resetPromise = null;
    };
    operation.then(retire, retire);
    return operation;
  }

  dispose(): Promise<void> {
    if (this.#disposePromise !== null) return this.#disposePromise;
    this.#disposed = true;
    this.#disposePromise = this.#disposeOwned();
    return this.#disposePromise;
  }

  #assertLive(): void {
    if (this.#disposed) {
      throw diagnosticError(
        "activation-workspace-disposed",
        "Qwen3.5 activation workspace is disposed",
      );
    }
    if (this.#poisoned) {
      throw diagnosticError(
        "activation-workspace-poisoned",
        "Qwen3.5 activation workspace is poisoned and must be disposed",
      );
    }
  }

  #assertAccessible(): void {
    this.#assertLive();
    if (this.#resetPromise !== null) {
      throw diagnosticError(
        "activation-workspace-resetting",
        "Qwen3.5 activation workspace reset is in progress",
      );
    }
  }

  async #clearResources(): Promise<void> {
    try {
      for (const resource of this.#owned) {
        await this.#clearAllocation(resource.allocation, resource.view);
      }
    } catch {
      // Partial zeroing cannot be classified as either old or clean state.
      this.#poisoned = true;
      throw diagnosticError(
        "activation-workspace-reset-failed",
        "Qwen3.5 activation workspace reset failed",
      );
    }
  }

  async #disposeOwned(): Promise<void> {
    const reset = this.#resetPromise;
    if (reset !== null) {
      // Buffer ownership outlives an in-flight clear operation.
      try {
        await reset;
      } catch {
        // Reset already poisoned the workspace; disposal must still reclaim it.
      }
    }
    if (destroyReverse(this.#owned.map(({ allocation }) => allocation)) !== undefined) {
      throw diagnosticError(
        "activation-workspace-disposal-failed",
        "Qwen3.5 activation workspace disposal failed",
      );
    }
  }
}

/** Allocates one complete GPU binding for every fixed transient vector. */
export async function createQwen35ActivationWorkspace(
  options: CreateQwen35ActivationWorkspaceOptions,
): Promise<Qwen35ActivationWorkspace> {
  const allocations: GpuAllocation[] = [];
  const owned: OwnedResource[] = [];
  try {
    for (const [index, plan] of ACTIVATION_PLAN.resources.entries()) {
      const allocation = await options.arena.allocate({
        id: `activation-workspace-${index}`,
        category: "activation",
        byteLength: plan.bytes,
        usage: plan.usage,
        alignment: SCALAR_BYTES,
        // Current kernels bind one logical vector; splitting it is not valid.
        requiredShardQuantumBytes: plan.bytes,
      });
      allocations.push(allocation);
      owned.push(Object.freeze({
        allocation,
        view: bindingView(plan, allocation),
      }));
    }
  } catch {
    if (destroyReverse(allocations) !== undefined) {
      throw diagnosticError(
        "activation-workspace-layout-rollback-failed",
        "Qwen3.5 activation workspace layout rollback failed",
      );
    }
    throw diagnosticError(
      "activation-workspace-layout-allocation-failed",
      "Qwen3.5 activation workspace layout allocation failed",
    );
  }
  return new Qwen35ActivationWorkspace(
    Object.freeze(owned),
    options.clearAllocation,
  );
}
