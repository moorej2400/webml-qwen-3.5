/**
 * Public runtime diagnostics carry stable codes and pre-sanitized messages.
 * Callers must not pass browser, compiler, model, or allocation text through.
 */
export class RuntimeDiagnosticError extends Error {
  readonly code: string;

  constructor(code: string, message: string) {
    super(message);
    this.name = "RuntimeDiagnosticError";
    this.code = code;
  }
}

export const ALLOCATION_DIAGNOSTIC_CODES = Object.freeze([
  "gpu_out_of_memory",
  "gpu_validation",
  "buffer_creation",
  "error_scope",
  "allocation_conflict",
  "gpu_ambiguous_scopes",
  "state_metadata",
  "state_progress",
  "unknown",
] as const);

export type AllocationDiagnosticCode =
  (typeof ALLOCATION_DIAGNOSTIC_CODES)[number];

const ALLOCATION_DIAGNOSTIC_CODE_SET = new Set<string>(
  ALLOCATION_DIAGNOSTIC_CODES,
);

/** Maps every external or lower-layer code into the closed public enum. */
export function sanitizeAllocationDiagnosticCode(
  value: unknown,
): AllocationDiagnosticCode {
  return typeof value === "string" && ALLOCATION_DIAGNOSTIC_CODE_SET.has(value)
    ? value as AllocationDiagnosticCode
    : "unknown";
}

/** Creates an allocation failure without accepting any source error text. */
export function allocationDiagnosticError(
  code: unknown,
): RuntimeDiagnosticError {
  return new RuntimeDiagnosticError(
    sanitizeAllocationDiagnosticCode(code),
    "GPU buffer allocation failed",
  );
}

export function diagnosticError(
  code: string,
  message: string,
): RuntimeDiagnosticError {
  return new RuntimeDiagnosticError(code, message);
}

export function isSafeDiagnosticCode(value: string): boolean {
  return /^[a-z][a-z0-9-]{0,63}$/.test(value);
}
