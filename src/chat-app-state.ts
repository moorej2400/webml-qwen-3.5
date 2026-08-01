export interface ChatGenerationFailurePresentation {
  readonly kind: "cancelled" | "error";
  readonly notice: string;
  readonly status: string;
}

/** Serializes UI actions that can mutate one loaded session. */
export class ChatOperationGate {
  #active = false;

  get busy(): boolean {
    return this.#active;
  }

  async run<T>(operation: () => Promise<T>): Promise<T | undefined> {
    if (this.#active) return undefined;
    this.#active = true;
    try {
      return await operation();
    } finally {
      this.#active = false;
    }
  }
}

export function emptyChatContextCopy(): string {
  return "Prefill a conversation to measure it";
}

export function presentChatGenerationFailure(
  error: unknown,
  options: {
    readonly cancellationRequested: boolean;
    readonly runtimeFailed: boolean;
  },
): ChatGenerationFailurePresentation {
  if (options.cancellationRequested) {
    return Object.freeze({
      kind: "cancelled" as const,
      notice: options.runtimeFailed
        ? "Generation stopped. The runtime was disposed to protect GPU state."
        : "Generation stopped.",
      status: options.runtimeFailed ? "Ready to reload" : "Ready for a prompt",
    });
  }
  return Object.freeze({
    kind: "error" as const,
    notice: error instanceof Error ? error.message : "The request failed.",
    status: "Needs attention",
  });
}
