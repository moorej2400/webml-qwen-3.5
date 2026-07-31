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

export function diagnosticError(
  code: string,
  message: string,
): RuntimeDiagnosticError {
  return new RuntimeDiagnosticError(code, message);
}

export function isSafeDiagnosticCode(value: string): boolean {
  return /^[a-z][a-z0-9-]{0,63}$/.test(value);
}
