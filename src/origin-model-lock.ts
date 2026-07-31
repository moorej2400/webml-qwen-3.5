export type OriginModelLockState =
  | "idle"
  | "blocked"
  | "acquired"
  | "aborted"
  | "released";

export interface ExclusiveLockRequestOptions {
  mode: "exclusive";
  signal: AbortSignal;
}

/**
 * Minimal Web Locks boundary so ownership and cancellation can be tested
 * without a browser global.
 */
export interface ExclusiveLockManager {
  request<T>(
    name: string,
    options: ExclusiveLockRequestOptions,
    callback: () => Promise<T> | T,
  ): Promise<T>;
}

export interface ExclusiveModelWork<T> {
  run(signal: AbortSignal): Promise<T>;
  /**
   * Cleanup gets an ownership signal that is separate from work cancellation.
   * The owner can apply its own bounded cleanup policy without prompt or
   * command cancellation invalidating GPU disposal.
   */
  cleanup(signal: AbortSignal): Promise<void>;
}

export class OriginModelLockCleanupError extends Error {
  readonly code = "origin-model-lock-cleanup-failed";

  constructor() {
    super("Model cleanup failed while the exclusive origin lock was held");
    this.name = "OriginModelLockCleanupError";
  }
}

const DEFAULT_LOCK_NAME = "webml-qwen-3.5-model-runtime-v1";

/**
 * Serializes every operation that can touch model bytes or WebGPU resources.
 *
 * The Web Lock callback does not exit until cleanup settles. This ordering is
 * the ownership boundary: releasing earlier can let a second Safari tab
 * allocate while the first tab still has live GPU resources.
 */
export class OriginModelLock {
  private controller = new AbortController();
  private running = false;
  private currentState: OriginModelLockState = "idle";
  private readonly stateTransitions: OriginModelLockState[] = [];

  constructor(
    private readonly manager: ExclusiveLockManager,
    private readonly name = DEFAULT_LOCK_NAME,
  ) {}

  get state(): OriginModelLockState {
    return this.currentState;
  }

  get transitions(): readonly OriginModelLockState[] {
    return [...this.stateTransitions];
  }

  cancel(): void {
    if (
      this.currentState === "blocked" ||
      this.currentState === "acquired"
    ) {
      this.transition("aborted");
    }
    this.controller.abort();
  }

  async runExclusive<T>(work: ExclusiveModelWork<T>): Promise<T> {
    if (this.running) {
      throw new Error("This origin model lock already has active work");
    }
    if (
      this.currentState === "released" ||
      this.currentState === "aborted"
    ) {
      this.controller = new AbortController();
    }
    this.running = true;
    this.transition("blocked");
    const cleanupController = new AbortController();

    try {
      return await this.manager.request(
        this.name,
        { mode: "exclusive", signal: this.controller.signal },
        async () => {
          this.transition("acquired");
          let value: T | undefined;
          let workFailure: unknown;
          let workFailed = false;
          try {
            value = await work.run(this.controller.signal);
          } catch (error) {
            workFailed = true;
            workFailure = error;
          }

          let cleanupFailed = false;
          try {
            await work.cleanup(cleanupController.signal);
          } catch {
            cleanupFailed = true;
          }

          // This is the last statement before the Web Lock callback exits.
          // A cleanup failure is surfaced only after ownership stayed exclusive
          // for the complete cleanup attempt.
          this.transition("released");
          if (cleanupFailed) {
            throw new OriginModelLockCleanupError();
          }
          if (workFailed) {
            throw workFailure;
          }
          return value as T;
        },
      );
    } catch (error) {
      if (
        this.controller.signal.aborted &&
        this.currentState === "blocked"
      ) {
        this.transition("aborted");
      }
      throw error;
    } finally {
      this.running = false;
    }
  }

  private transition(state: OriginModelLockState): void {
    if (this.currentState === state) {
      return;
    }
    this.currentState = state;
    this.stateTransitions.push(state);
  }
}

export function browserExclusiveLockManager(
  locks: LockManager = navigator.locks,
): ExclusiveLockManager {
  return {
    request<T>(
      name: string,
      options: ExclusiveLockRequestOptions,
      callback: () => Promise<T> | T,
    ): Promise<T> {
      // Web Locks adopts the callback promise at runtime, but lib.dom models
      // the generic callback result without Awaited<T>.
      return locks.request<Promise<T>>(
        name,
        options,
        async () => callback(),
      ) as Promise<T>;
    },
  };
}
