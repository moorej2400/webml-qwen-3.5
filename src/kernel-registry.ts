import {
  diagnosticError,
  isSafeDiagnosticCode,
} from "./diagnostics.js";

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
    readonly code: "KERNEL_COMPILE_FAILED";
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

function requireSafeKernelCode(value: string): void {
  if (!isSafeDiagnosticCode(value)) {
    throw diagnosticError(
      "KERNEL_REGISTRATION_INVALID",
      "Kernel diagnostic code is invalid",
    );
  }
}

function frozenKey(key: KernelKey): KernelKey {
  return Object.freeze({ ...key });
}

function frozenKernel(kernel: KernelDefinition): KernelDefinition {
  return Object.freeze({
    ...kernel,
    key: frozenKey(kernel.key),
  });
}

function frozenPivot(pivot: KernelPivot): KernelPivot {
  return Object.freeze({ ...pivot });
}

function frozenSelection(
  kernel: KernelDefinition,
  pivot: KernelPivot | null,
  attemptedProfiles: readonly string[],
): KernelSelection {
  return Object.freeze({
    kernel: frozenKernel(kernel),
    pivot: pivot === null ? null : frozenPivot(pivot),
    attemptedProfiles: Object.freeze([...attemptedProfiles]),
  });
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
    requireSafeKernelCode(kernel.id);
    requireSafeKernelCode(kernel.key.operation);
    requireSafeKernelCode(kernel.key.layout);
    requireSafeKernelCode(kernel.key.phase);
    requireSafeKernelCode(kernel.key.profile);
    const text = keyText(kernel.key);
    if (this.#kernels.has(text)) {
      throw diagnosticError(
        "KERNEL_REGISTRATION_DUPLICATE",
        "Duplicate kernel key",
      );
    }
    this.#kernels.set(text, frozenKernel(kernel));
  }

  select(request: KernelSelectionRequest): KernelSelection {
    requireSafeKernelCode(request.key.operation);
    requireSafeKernelCode(request.key.layout);
    requireSafeKernelCode(request.key.phase);
    requireSafeKernelCode(request.key.profile);
    for (const profile of request.fallbackProfiles) {
      requireSafeKernelCode(profile);
    }
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
        return frozenSelection(kernel, null, attemptedProfiles);
      }
      if (
        request.pivotReason === undefined ||
        !isSafeDiagnosticCode(request.pivotReason)
      ) {
        throw diagnosticError(
          "KERNEL_FALLBACK_REASON_INVALID",
          "Kernel fallback requires a safe reason code",
        );
      }
      const pivot = frozenPivot({
        fromProfile: request.key.profile,
        toProfile: profile,
        reason: request.pivotReason,
      });
      this.#pivots.push(pivot);
      return frozenSelection(kernel, pivot, attemptedProfiles);
    }
    throw diagnosticError(
      "KERNEL_NOT_FOUND",
      "No kernel matched the requested operation and explicit profiles",
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
      this.#metrics.push(Object.freeze({
        kernelId: selection.kernel.id,
        key: frozenKey(selection.kernel.key),
        status: "success",
        durationMs: Math.max(0, this.#clock() - started),
        pivot:
          selection.pivot === null ? null : frozenPivot(selection.pivot),
        error: null,
      }));
      return { value, selection };
    } catch {
      // Compilation failure is evidence for a future explicit pivot; compile()
      // never retries another profile without a new selection request.
      this.#metrics.push(Object.freeze({
        kernelId: selection.kernel.id,
        key: frozenKey(selection.kernel.key),
        status: "error",
        durationMs: Math.max(0, this.#clock() - started),
        pivot:
          selection.pivot === null ? null : frozenPivot(selection.pivot),
        error: Object.freeze({
          code: "KERNEL_COMPILE_FAILED" as const,
          message: "Kernel compilation failed",
        }),
      }));
      throw diagnosticError(
        "KERNEL_COMPILE_FAILED",
        "Kernel compilation failed",
      );
    }
  }

  pivotRecords(): readonly KernelPivot[] {
    return Object.freeze(
      this.#pivots.map((pivot) => frozenPivot(pivot)),
    );
  }

  compilationMetrics(): readonly CompilationMetric[] {
    return Object.freeze(
      this.#metrics.map((metric) =>
        Object.freeze({
          ...metric,
          key: frozenKey(metric.key),
          pivot:
            metric.pivot === null ? null : frozenPivot(metric.pivot),
          error:
            metric.error === null
              ? null
              : Object.freeze({ ...metric.error }),
        }),
      ),
    );
  }
}
