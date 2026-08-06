import assert from "node:assert/strict";
import test from "node:test";

import {
  startQwen35ChatApp,
  type Qwen35ChatAppHandle,
} from "../src/chat-app.js";
import type { RuntimeLoadEvent } from "../src/qwen35-session.js";

class FakeElement {
  textContent = "";
  className = "";
  hidden = false;
  disabled = false;
  readonly dataset: Record<string, string> = {};
  readonly children: unknown[] = [];
  readonly classList = { add() {} };

  addEventListener(): void {}
  append(...children: unknown[]): void { this.children.push(...children); }
  replaceChildren(...children: unknown[]): void {
    this.children.length = 0;
    this.children.push(...children);
  }
  scrollIntoView(): void {}
  setAttribute(): void {}
  toggleAttribute(): void {}
  hasAttribute(): boolean { return false; }
  matches(): boolean { return false; }
  querySelector(): FakeElement | null { return new FakeElement(); }
}

async function withChatEnvironment(
  fetchImplementation: typeof fetch,
  run: (input: {
    readonly application: Qwen35ChatAppHandle;
    readonly element: (selector: string) => FakeElement;
  }) => Promise<void>,
): Promise<void> {
  const originalWindow = Object.getOwnPropertyDescriptor(globalThis, "window");
  const originalDocument = Object.getOwnPropertyDescriptor(globalThis, "document");
  const originalFetch = Object.getOwnPropertyDescriptor(globalThis, "fetch");
  const elements = new Map<string, FakeElement>();
  const element = (selector: string): FakeElement => {
    const existing = elements.get(selector);
    if (existing !== undefined) return existing;
    const created = new FakeElement();
    elements.set(selector, created);
    return created;
  };
  const testDocument = {
    scripts: [],
    querySelector: (selector: string) => element(selector),
    querySelectorAll: () => [],
    createElement: () => new FakeElement(),
    addEventListener() {},
  };
  const testWindow = {
    __QWEN35_RUNTIME_CONFIG__: {
      manifestUrl: "https://example.invalid/manifest.json",
      packageBaseUrl: "https://example.invalid/package/",
      expectedPackageBaseUrl: "https://example.invalid/package/",
      expectedManifestSha256: "a".repeat(64),
      compiledTokenizerUrl: "https://example.invalid/tokenizer.bin",
    },
    addEventListener() {},
  };
  Object.defineProperty(globalThis, "document", {
    configurable: true,
    value: testDocument,
  });
  Object.defineProperty(globalThis, "window", {
    configurable: true,
    value: testWindow,
  });
  Object.defineProperty(globalThis, "fetch", {
    configurable: true,
    value: fetchImplementation,
  });

  try {
    const application = startQwen35ChatApp(
      element("#qwen-app") as unknown as HTMLElement,
      { autoLoad: false },
    );
    await run({ application, element });
  } finally {
    for (const [name, descriptor] of [
      ["window", originalWindow],
      ["document", originalDocument],
      ["fetch", originalFetch],
    ] as const) {
      if (descriptor === undefined) {
        Reflect.deleteProperty(globalThis, name);
      } else {
        Object.defineProperty(globalThis, name, descriptor);
      }
    }
  }
}

test("manifest resolution failure publishes one sanitized failed load event", async () => {
  await withChatEnvironment(
    async () => new Response("unavailable", { status: 503 }),
    async ({ application, element }) => {
    const observed: RuntimeLoadEvent[] = [];
    application.coordinator.subscribeLoadEvents((event) => observed.push(event));

    await assert.rejects(application.coordinator.load(), /could not be loaded/i);

    assert.deepEqual(observed, [{
      phase: "failed",
      completedBytes: 0,
      totalBytes: 0,
    }]);
    assert.equal(application.coordinator.state, "failed");
    assert.equal(element("[data-runtime-status]").textContent, "failed");
    },
  );
});

test("cancel aborts manifest resolution before session model work can start", async () => {
  let fetchStarted = false;
  let fetchSignal: AbortSignal | undefined;
  let rejectFetch!: (error: unknown) => void;
  const pendingFetch: typeof fetch = (_input, init) => {
    fetchStarted = true;
    fetchSignal = init?.signal ?? undefined;
    return new Promise<Response>((_resolve, reject) => {
      rejectFetch = reject;
      fetchSignal?.addEventListener("abort", () => {
        reject(fetchSignal?.reason ?? new DOMException("cancelled", "AbortError"));
      }, { once: true });
    });
  };

  await withChatEnvironment(pendingFetch, async ({ application }) => {
    const observed: RuntimeLoadEvent[] = [];
    application.coordinator.subscribeLoadEvents((event) => observed.push(event));
    const loading = application.coordinator.load();
    while (!fetchStarted) await Promise.resolve();

    await application.coordinator.cancel();
    const manifestWasAborted = fetchSignal?.aborted === true;
    if (!manifestWasAborted) {
      // Settle the intentionally broken implementation so the RED test cannot leak work.
      rejectFetch(new DOMException("cancelled", "AbortError"));
    }

    await assert.rejects(loading, { name: "AbortError" });
    assert.equal(manifestWasAborted, true);
    assert.deepEqual(observed, []);
  });
});
