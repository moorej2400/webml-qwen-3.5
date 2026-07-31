import { validateProtocolId } from "./protocol.js";

export interface CollisionBus {
  claim(tabId: string, claimantId: string): boolean;
  release(tabId: string, claimantId: string): void;
}

export class InMemoryCollisionBus implements CollisionBus {
  readonly #claims = new Map<string, string>();

  claim(tabId: string, claimantId: string): boolean {
    const owner = this.#claims.get(tabId);
    if (owner !== undefined && owner !== claimantId) return false;
    this.#claims.set(tabId, claimantId);
    return true;
  }

  release(tabId: string, claimantId: string): void {
    if (this.#claims.get(tabId) === claimantId) this.#claims.delete(tabId);
  }
}

export const claimUniqueTabId = async (options: {
  storedTabId: string;
  claimantId: string;
  bus: CollisionBus;
  generateTabId: () => string;
}): Promise<{ tabId: string; release: () => void }> => {
  const claimantId = validateProtocolId(options.claimantId, "claimantId");
  let tabId = validateProtocolId(options.storedTabId, "tabId");
  for (let attempt = 0; attempt < 8; attempt += 1) {
    if (options.bus.claim(tabId, claimantId)) {
      return { tabId, release: () => options.bus.release(tabId, claimantId) };
    }
    tabId = validateProtocolId(options.generateTabId(), "tabId");
  }
  throw new Error("could not allocate a unique live tabId");
};
