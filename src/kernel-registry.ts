export interface KernelKey {
  readonly operation: string;
  readonly layout: string;
  readonly phase: string;
  readonly profile: string;
}

export interface KernelDefinition {
  readonly id: string;
  readonly key: KernelKey;
  readonly source: string;
}

export interface KernelPivot {
  readonly fromProfile: string;
  readonly toProfile: string;
  readonly reason: string;
}

export interface KernelSelection {
  readonly kernel: KernelDefinition;
  readonly pivot: KernelPivot | null;
  readonly attemptedProfiles: readonly string[];
}

export interface KernelSelectionRequest {
  readonly key: KernelKey;
  readonly fallbackProfiles: readonly string[];
  readonly pivotReason?: string;
}

export interface CompilationMetric {
  readonly kernelId: string;
  readonly key: KernelKey;
  readonly status: "success" | "error";
  readonly durationMs: number;
  readonly pivot: KernelPivot | null;
  readonly error: {
    readonly name: string;
    readonly message: string;
  } | null;
}

function keyText(key: KernelKey): string {
  return JSON.stringify([
    key.operation,
    key.layout,
    key.phase,
    key.profile,
  ]);
}

function sanitizeError(error: unknown): { name: string; message: string } {
  if (error instanceof Error) {
    return { name: error.name, message: error.message };
  }
  return { name: "Error", message: String(error) };
}

export class KernelRegistry {
  readonly #kernels = new Map<string, KernelDefinition>();
  readonly #metrics: CompilationMetric[] = [];
  readonly #pivots: KernelPivot[] = [];
  readonly #clock: () => number;

  constructor(clock: () => number = () => performance.now()) {
    this.#clock = clock;
  }

  register(kernel: KernelDefinition): void {
    const text = keyText(kernel.key);
    if (this.#kernels.has(text)) {
      throw new Error(`Duplicate kernel key: ${text}`);
    }
    this.#kernels.set(text, {
      ...kernel,
      key: { ...kernel.key },
    });
  }

  select(request: KernelSelectionRequest): KernelSelection {
    const attemptedProfiles: string[] = [];
    for (const profile of [
      request.key.profile,
      ...request.fallbackProfiles,
    ]) {
      attemptedProfiles.push(profile);
      const key = { ...request.key, profile };
      const kernel = this.#kernels.get(keyText(key));
      if (kernel === undefined) {
        continue;
      }
      if (profile === request.key.profile) {
        return { kernel, pivot: null, attemptedProfiles };
      }
      if (request.pivotReason === undefined || request.pivotReason.length === 0) {
        throw new Error(
          "An explicit pivot reason is required for kernel fallback",
        );
      }
      const pivot = {
        fromProfile: request.key.profile,
        toProfile: profile,
        reason: request.pivotReason,
      };
      this.#pivots.push(pivot);
      return {
        kernel,
        pivot,
        attemptedProfiles,
      };
    }
    throw new Error(
      `No kernel for ${request.key.operation}/${request.key.layout}/${request.key.phase}; attempted profiles: ${attemptedProfiles.join(", ")}`,
    );
  }

  async compile<T>(
    request: KernelSelectionRequest,
    compiler: (kernel: KernelDefinition) => Promise<T>,
  ): Promise<{ value: T; selection: KernelSelection }> {
    const selection = this.select(request);
    const started = this.#clock();
    try {
      const value = await compiler(selection.kernel);
      this.#metrics.push({
        kernelId: selection.kernel.id,
        key: { ...selection.kernel.key },
        status: "success",
        durationMs: Math.max(0, this.#clock() - started),
        pivot: selection.pivot,
        error: null,
      });
      return { value, selection };
    } catch (error) {
      // Compilation failure is evidence for a future explicit pivot; compile()
      // never retries another profile without a new selection request.
      this.#metrics.push({
        kernelId: selection.kernel.id,
        key: { ...selection.kernel.key },
        status: "error",
        durationMs: Math.max(0, this.#clock() - started),
        pivot: selection.pivot,
        error: sanitizeError(error),
      });
      throw error;
    }
  }

  pivotRecords(): readonly KernelPivot[] {
    return this.#pivots.map((pivot) => ({ ...pivot }));
  }

  compilationMetrics(): readonly CompilationMetric[] {
    return this.#metrics.map((metric) => ({
      ...metric,
      key: { ...metric.key },
      pivot: metric.pivot === null ? null : { ...metric.pivot },
      error: metric.error === null ? null : { ...metric.error },
    }));
  }
}
