import { assertHighEntropyCredential } from "./security.js";

export interface BrowserAgentSourceOptions {
  phoneToken: string;
  websocketPath?: string;
}

/**
 * This source is generated only by the local HTTPS server. The public build
 * compiles `src/` directly and cannot import this directory.
 */
export const createBrowserAgentSource = (options: BrowserAgentSourceOptions): string => {
  assertHighEntropyCredential(options.phoneToken, "phone credential");
  const path = options.websocketPath ?? "/.local-control";
  if (!path.startsWith("/") || path.includes("\n") || path.includes("\r")) {
    throw new TypeError("websocketPath must be an absolute path");
  }
  const token = JSON.stringify(options.phoneToken);
  const websocketPath = JSON.stringify(path);
  return `(() => {
  "use strict";
  const VERSION = 1;
  const TOKEN = ${token};
  const PATH = ${websocketPath};
  const safeId = (prefix) => prefix + "_" + crypto.randomUUID().replaceAll("-", "");
  const durable = (storage, key, prefix) => {
    let value = storage.getItem(key);
    if (!value) {
      value = safeId(prefix);
      storage.setItem(key, value);
    }
    return value;
  };
  const identity = {
    deviceId: durable(localStorage, "qwen.control.device", "device"),
    tabId: durable(sessionStorage, "qwen.control.tab", "tab"),
    documentId: safeId("document")
  };
  let sequence = 0;
  let socket;
  let reconnectDelay = 250;
  const records = new Map();
  const send = (message) => {
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ schemaVersion: VERSION, ...identity, eventSeq: ++sequence, ...message }));
    }
  };
  const transition = (commandId, state, reason) => {
    records.set(commandId, { state, reason });
    send({ type: "commandState", commandId, state, ...(reason ? { reason } : {}) });
  };
  const handleCommand = async (message) => {
    const previous = records.get(message.commandId);
    if (previous) {
      transition(message.commandId, previous.state, previous.reason);
      return;
    }
    transition(message.commandId, "accepted");
    transition(message.commandId, "started");
    if (message.command === "warmReload") {
      setTimeout(() => location.reload(), 0);
      return;
    }
    if (message.command === "coldAppReload") {
      transition(message.commandId, "indeterminate", "external_fallback_required");
      return;
    }
    try {
      const handlers = globalThis.__QWEN_LOCAL_CONTROL__;
      const handler = handlers && handlers[message.command];
      if (typeof handler !== "function" && message.command !== "getState") {
        throw new Error("handler_unavailable");
      }
      await handler?.(message.payload);
      transition(message.commandId, "completed");
    } catch {
      transition(
        message.commandId,
        message.command === "cancelPrompt" ? "cancelled" : "failed",
        "handler_failed"
      );
    }
  };
  const connect = () => {
    const scheme = location.protocol === "https:" ? "wss:" : "ws:";
    socket = new WebSocket(scheme + "//" + location.host + PATH, ["qwen-control.v1", TOKEN]);
    socket.addEventListener("open", () => {
      reconnectDelay = 250;
      socket.send(JSON.stringify({ schemaVersion: VERSION, type: "hello", ...identity }));
      send({ type: "ready" });
    });
    socket.addEventListener("message", (event) => {
      let message;
      try { message = JSON.parse(event.data); } catch { return; }
      if (message.schemaVersion !== VERSION) return;
      if (message.type === "command") void handleCommand(message);
      if (message.type === "reconcile") {
        for (const command of message.commands || []) {
          const local = records.get(command.commandId);
          if (!local && command.state !== "issued" && command.state !== "accepted") {
            records.set(command.commandId, { state: command.state, reason: command.reason });
          }
        }
      }
    });
    socket.addEventListener("close", () => {
      setTimeout(connect, reconnectDelay);
      reconnectDelay = Math.min(5000, reconnectDelay * 2);
    });
  };
  addEventListener("pagehide", () => send({
    type: "telemetry",
    event: { category: "lifecycle", name: "pagehide", timestampMs: Date.now(), metrics: { lifecycle: "pagehide" } }
  }));
  addEventListener("pageshow", connect, { once: true });
})();`;
};
