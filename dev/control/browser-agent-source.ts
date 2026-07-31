/**
 * The public local-agent response is credential-free. A user must enter the
 * process-only pairing code on the phone before this script can obtain a
 * device-bound session and single-use WSS ticket.
 */
export const createBrowserAgentSource = (): string => `(() => {
  "use strict";
  const VERSION = 1;
  const OUTBOX_LIMIT = 256;
  const ALLOWED_COMMANDS = new Set([
    "load", "dispose", "runPrompt", "cancelPrompt", "getState", "warmReload", "coldAppReload"
  ]);
  const safeId = (prefix) => prefix + "_" + crypto.randomUUID().replaceAll("-", "");
  const durable = (storage, key, prefix) => {
    let value = storage.getItem(key);
    if (!value) {
      value = safeId(prefix);
      storage.setItem(key, value);
    }
    return value;
  };
  const deviceId = durable(localStorage, "qwen.control.device", "device");
  const documentId = safeId("document");
  const claimantId = safeId("claimant");
  let tabId = durable(sessionStorage, "qwen.control.tab", "tab");
  let sequence = 0;
  let socket;
  let reconnectDelay = 250;
  let connectPromise;
  let runtimeReady = typeof globalThis.__QWEN_LOCAL_CONTROL__ === "object";
  let identifiedSocket;
  let tabClaimed = false;
  let tabCollision = false;
  const records = new Map();
  const outbox = new Map();
  let highestAcknowledged = 0;
  let activePrompt;
  const tabChannel = new BroadcastChannel("qwen-control-tabs-v1");
  tabChannel.addEventListener("message", ({ data }) => {
    if (!data || data.tabId !== tabId || data.claimantId === claimantId) return;
    if (data.type === "probe") {
      if (tabClaimed || claimantId < data.claimantId) {
        tabChannel.postMessage({ type: "occupied", tabId, claimantId, target: data.claimantId });
      } else {
        tabCollision = true;
      }
    }
    if (data.type === "occupied" && data.target === claimantId) tabCollision = true;
  });
  const claimTab = async () => {
    for (let attempt = 0; attempt < 8; attempt += 1) {
      tabCollision = false;
      tabChannel.postMessage({ type: "probe", tabId, claimantId });
      await new Promise((resolve) => setTimeout(resolve, 60));
      if (!tabCollision) {
        tabClaimed = true;
        return;
      }
      tabId = safeId("tab");
      sessionStorage.setItem("qwen.control.tab", tabId);
    }
    throw new Error("tab_identity_collision");
  };
  const identity = () => ({ deviceId, tabId, documentId });
  const requestJson = async (url, body, capability) => {
    const response = await fetch(url, {
      method: "POST",
      cache: "no-store",
      headers: {
        "content-type": "application/json",
        ...(capability ? { authorization: "Bearer " + capability } : {})
      },
      body: JSON.stringify(body)
    });
    if (!response.ok) throw new Error("control_auth_failed");
    return response.json();
  };
  const sendFrame = (frame) => {
    if (
      socket &&
      identifiedSocket === socket &&
      socket.readyState === WebSocket.OPEN
    ) {
      socket.send(JSON.stringify(frame));
    }
  };
  const send = (message) => {
    if (outbox.size >= OUTBOX_LIMIT) throw new Error("control_outbox_limit");
    const eventSeq = sequence + 1;
    const frame = {
      schemaVersion: VERSION,
      ...identity(),
      eventSeq,
      ...message
    };
    // Retain ownership before transport delivery so reconnect and synchronous
    // test transports cannot lose an unacknowledged state transition.
    outbox.set(eventSeq, frame);
    sequence = eventSeq;
    sendFrame(frame);
  };
  const replayFrom = (expectedSeq) => {
    if (!Number.isSafeInteger(expectedSeq) || expectedSeq < 1 || expectedSeq <= highestAcknowledged) {
      throw new Error("control_replay_sequence_invalid");
    }
    if (!outbox.has(expectedSeq)) {
      if (outbox.size === 0 && expectedSeq === sequence + 1) return;
      throw new Error("control_replay_unavailable");
    }
    for (const [eventSeq, frame] of [...outbox].sort(([left], [right]) => left - right)) {
      if (eventSeq >= expectedSeq) sendFrame(frame);
    }
  };
  const synchronize = (message) => {
    if (
      message.documentId !== documentId ||
      !Number.isSafeInteger(message.expectedSeq) ||
      message.expectedSeq < 1
    ) throw new Error("control_sync_invalid");
    const acknowledgedSeq = message.expectedSeq - 1;
    if (acknowledgedSeq < highestAcknowledged) {
      throw new Error("control_sync_regression");
    }
    if (acknowledgedSeq > sequence) throw new Error("control_sync_bounds_invalid");
    // expectedSeq proves that every lower event reached the authenticated server.
    highestAcknowledged = acknowledgedSeq;
    for (const eventSeq of outbox.keys()) {
      if (eventSeq <= highestAcknowledged) outbox.delete(eventSeq);
    }
    replayFrom(message.expectedSeq);
  };
  const acknowledge = (message) => {
    if (
      message.documentId !== documentId ||
      !["accepted", "gap", "replay"].includes(message.status) ||
      !Number.isSafeInteger(message.acknowledgedSeq) ||
      !Number.isSafeInteger(message.expectedSeq) ||
      message.acknowledgedSeq < 0 ||
      message.expectedSeq !== message.acknowledgedSeq + 1
    ) throw new Error("control_ack_invalid");
    if (message.status === "gap") {
      replayFrom(message.expectedSeq);
      return;
    }
    if (message.acknowledgedSeq < highestAcknowledged || message.acknowledgedSeq > sequence) {
      throw new Error("control_ack_bounds_invalid");
    }
    highestAcknowledged = message.acknowledgedSeq;
    for (const eventSeq of outbox.keys()) {
      if (eventSeq <= highestAcknowledged) outbox.delete(eventSeq);
    }
  };
  const resendState = (commandId, record) => {
    const retained = [...outbox.values()].reverse().find(
      (frame) => frame.type === "commandState" &&
        frame.commandId === commandId && frame.state === record.state
    );
    if (retained) {
      sendFrame(retained);
      return;
    }
    transition(commandId, record.state, record.reason, record.result);
  };
  const transition = (commandId, state, reason, result) => {
    records.set(commandId, { state, reason, result });
    send({
      type: "commandState",
      commandId,
      state,
      ...(reason ? { reason } : {}),
      ...(result ? { result } : {})
    });
  };
  const safeState = (value) => {
    const allowed = new Set([
      "modelState", "generationState", "cacheState", "deviceState", "loaded",
      "generating", "contextTokens", "maxContextTokens", "cpuBytes", "gpuBytes",
      "activeCommandId"
    ]);
    const result = {};
    if (value && typeof value === "object" && !Array.isArray(value)) {
      for (const [key, entry] of Object.entries(value).slice(0, 32)) {
        if (!allowed.has(key)) continue;
        if (typeof entry === "boolean" || typeof entry === "number") result[key] = entry;
        if (typeof entry === "string") result[key] = entry.slice(0, 128);
      }
    }
    return JSON.stringify(result).length <= 8192 ? result : {};
  };
  const showPairing = () => {
    if (document.getElementById("qwen-control-pairing-form")) return;
    const dialog = document.createElement("dialog");
    dialog.setAttribute("aria-labelledby", "qwen-control-pairing-title");
    const form = document.createElement("form");
    form.id = "qwen-control-pairing-form";
    form.method = "dialog";
    const title = document.createElement("strong");
    title.id = "qwen-control-pairing-title";
    title.textContent = "Pair development controls";
    const input = document.createElement("input");
    input.type = "password";
    input.required = true;
    input.autocomplete = "one-time-code";
    input.setAttribute("aria-label", "One-time pairing code");
    const button = document.createElement("button");
    button.type = "submit";
    button.textContent = "Pair";
    const status = document.createElement("span");
    status.setAttribute("role", "status");
    form.append(title, input, button, status);
    dialog.append(form);
    document.body.append(dialog);
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      button.disabled = true;
      status.textContent = "";
      try {
        await globalThis.__QWEN_LOCAL_PAIR__(input.value);
        input.value = "";
        dialog.close();
        dialog.remove();
      } catch {
        input.value = "";
        status.textContent = "Pairing failed.";
        button.disabled = false;
      }
    });
    dialog.showModal();
  };
  const handleCommand = async (message) => {
    if (!ALLOWED_COMMANDS.has(message.command)) return;
    const previous = records.get(message.commandId);
    if (previous) {
      resendState(message.commandId, previous);
      return;
    }
    transition(message.commandId, "accepted");
    transition(message.commandId, "started");
    if (message.command === "warmReload") {
      setTimeout(() => location.reload(), 0);
      return;
    }
    if (message.command === "coldAppReload") {
      transition(message.commandId, "started", "external_fallback_required");
      return;
    }
    const promptRecord =
      message.command === "runPrompt" && activePrompt === undefined
        ? { commandId: message.commandId, cancellation: undefined }
        : undefined;
    if (promptRecord) activePrompt = promptRecord;
    try {
      const handlers = globalThis.__QWEN_LOCAL_CONTROL__;
      const handler = handlers && Object.hasOwn(handlers, message.command)
        ? handlers[message.command]
        : undefined;
      if (typeof handler !== "function") throw new Error("handler_unavailable");
      let value;
      if (message.command === "cancelPrompt" && activePrompt) {
        activePrompt.cancellation ??= Promise.resolve().then(() => handler(message.payload));
        value = await activePrompt.cancellation;
      } else {
        value = await handler(message.payload);
      }
      if (promptRecord && promptRecord.cancellation) {
        await promptRecord.cancellation;
        transition(message.commandId, "cancelled", "operator_cancelled");
        return;
      }
      transition(
        message.commandId,
        "completed",
        undefined,
        message.command === "getState" ? safeState(value) : undefined
      );
    } catch {
      if (promptRecord && promptRecord.cancellation) {
        try {
          await promptRecord.cancellation;
          transition(message.commandId, "cancelled", "operator_cancelled");
        } catch {
          transition(message.commandId, "failed", "handler_failed");
        }
        return;
      }
      transition(
        message.commandId,
        message.command === "cancelPrompt" ? "cancelled" : "failed",
        "handler_failed"
      );
    } finally {
      if (activePrompt === promptRecord) activePrompt = undefined;
    }
  };
  const identifyIfRuntimeReady = (candidate) => {
    if (
      !runtimeReady ||
      candidate !== socket ||
      candidate.readyState !== WebSocket.OPEN ||
      identifiedSocket === candidate
    ) return;
    // The server can dispatch queued work as soon as it receives hello. Keep
    // the socket unidentified until the application has installed handlers.
    identifiedSocket = candidate;
    candidate.send(JSON.stringify({ schemaVersion: VERSION, type: "hello", ...identity() }));
    for (const [, frame] of [...outbox].sort(([left], [right]) => left - right)) {
      sendFrame(frame);
    }
    send({ type: "ready" });
  };
  const connectOnce = async () => {
    const capability = sessionStorage.getItem("qwen.control.session");
    if (!capability) {
      dispatchEvent(new CustomEvent("qwen-control-pairing-required"));
      showPairing();
      return;
    }
    try {
      const { ticket } = await requestJson("/.local-ticket", identity(), capability);
      const scheme = location.protocol === "https:" ? "wss:" : "ws:";
      const candidate = new WebSocket(
        scheme + "//" + location.host + "/.local-control",
        ["qwen-control.v1", ticket]
      );
      socket = candidate;
      candidate.addEventListener("open", () => {
        reconnectDelay = 250;
        identifyIfRuntimeReady(candidate);
      });
      candidate.addEventListener("message", (event) => {
        let message;
        try { message = JSON.parse(event.data); } catch { return; }
        if (message.schemaVersion !== VERSION) return;
        if (message.type === "eventAck") {
          try { acknowledge(message); } catch { candidate.close(1008, "protocol_error"); }
          return;
        }
        if (message.type === "sequenceSync") {
          try {
            synchronize(message);
          } catch {
            candidate.close(1008, "protocol_error");
          }
          return;
        }
        if (message.type === "command") void handleCommand(message);
        if (message.type === "reconcile") {
          for (const command of message.commands || []) {
            const local = records.get(command.commandId);
            if (local && local.state !== command.state) {
              resendState(command.commandId, local);
              continue;
            }
            if (!local && command.state !== "issued" && command.state !== "accepted") {
              records.set(command.commandId, {
                state: command.state,
                reason: command.reason,
                result: command.result
              });
            }
          }
        }
      });
      candidate.addEventListener("close", () => {
        if (identifiedSocket === candidate) identifiedSocket = undefined;
        setTimeout(() => void connect(), reconnectDelay);
        reconnectDelay = Math.min(5000, reconnectDelay * 2);
      });
    } catch {
      sessionStorage.removeItem("qwen.control.session");
      dispatchEvent(new CustomEvent("qwen-control-pairing-required"));
      showPairing();
    }
  };
  const connect = () => {
    if (connectPromise) return connectPromise;
    if (
      socket &&
      (socket.readyState === WebSocket.CONNECTING || socket.readyState === WebSocket.OPEN)
    ) return Promise.resolve();
    connectPromise = connectOnce().finally(() => {
      connectPromise = undefined;
    });
    return connectPromise;
  };
  globalThis.__QWEN_LOCAL_PAIR__ = async (pairingCode) => {
    const paired = await requestJson("/.local-pair", { pairingCode, ...identity() });
    sessionStorage.setItem("qwen.control.session", paired.sessionCapability);
    await connect();
  };
  addEventListener("qwen-local-runtime-ready", () => {
    if (typeof globalThis.__QWEN_LOCAL_CONTROL__ !== "object") return;
    runtimeReady = true;
    if (socket) identifyIfRuntimeReady(socket);
  });
  addEventListener("pagehide", () => send({
    type: "telemetry",
    event: {
      category: "lifecycle",
      name: "pagehide",
      timestampMs: Date.now(),
      metrics: { lifecycle: "pagehide" }
    }
  }));
  addEventListener("pageshow", () => void claimTab().then(connect), { once: true });
})();`;
