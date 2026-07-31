import assert from "node:assert/strict";
import test from "node:test";

import {
  InMemoryCollisionBus,
  claimUniqueTabId,
} from "../dev/control/tab-identity.js";

test("a cloned tab regenerates its sessionStorage tab id", async () => {
  const bus = new InMemoryCollisionBus();
  const storedTabId = "tab_0123456789abcdef";
  const first = await claimUniqueTabId({
    storedTabId,
    claimantId: "claimant_0123456789abcdef",
    bus,
    generateTabId: () => "tab_1123456789abcdef",
  });
  const second = await claimUniqueTabId({
    storedTabId,
    claimantId: "claimant_1123456789abcdef",
    bus,
    generateTabId: () => "tab_2123456789abcdef",
  });

  assert.equal(first.tabId, storedTabId);
  assert.equal(second.tabId, "tab_2123456789abcdef");
  first.release();
  second.release();
});
