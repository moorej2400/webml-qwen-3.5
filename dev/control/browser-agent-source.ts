import { SAFE_DIAGNOSTIC_CODE_PATTERN } from "../../src/diagnostics.js";

/**
 * The local-agent response contains no reusable secret. Each document obtains
 * a one-use WSS ticket from the same local HTTPS origin immediately before it
 * opens the socket.
 */
export const createBrowserAgentSource = (): string => `(() => {
  "use strict";
  const VERSION = 1;
  const OUTBOX_LIMIT = 256;
  const LOAD_TELEMETRY_OUTBOX_LIMIT = OUTBOX_LIMIT - 16;
  const LOAD_PHASES = new Set([
    "lock_wait", "cache_scan", "cache_download", "cache_verify",
    "tokenizer_load", "device_probe", "state_allocate", "weights_allocate",
    "weights_upload", "driver_initialize", "ready", "failed"
  ]);
  const UPLOAD_STAGES = new Set(["before_write", "after_write", "after_retire"]);
  // One event per 64 MiB keeps three multi-gigabyte passes below the bounded
  // outbox even when acknowledgements pause during a device stall.
  const LOAD_TELEMETRY_STEP_BYTES = 64 * 1024 * 1024;
  const ALLOWED_COMMANDS = new Set([
    "load", "dispose", "runPrompt", "cancelPrompt", "getState", "warmReload", "coldAppReload"
  ]);
  const TERMINAL_STATES = new Set([
    "completed", "failed", "cancelled", "timed_out", "indeterminate"
  ]);
  const ALLOCATION_DIAGNOSTIC_CODES = new Set([
    "gpu_out_of_memory", "gpu_validation", "buffer_creation",
    "error_scope", "allocation_conflict", "gpu_ambiguous_scopes",
    "state_metadata", "state_progress", "unknown"
  ]);
  const SAFE_DIAGNOSTIC_CODE = new RegExp(${JSON.stringify(SAFE_DIAGNOSTIC_CODE_PATTERN)});
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
  let reloadScheduled = false;
  let activeLoadCommandId;
  const records = new Map();
  const outbox = new Map();
  let highestAcknowledged = 0;
  let activePrompt;
  let lastLoadTelemetry;
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
  const requestJson = async (url, body) => {
    const response = await fetch(url, {
      method: "POST",
      cache: "no-store",
      credentials: "omit",
      headers: { "content-type": "application/json" },
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
    if (message.expectedSeq === 1 && sequence > 0) {
      // A local server restart has no sequence state for this still-live
      // document. Discard only transport history. Command records must remain
      // available so a retry with the same ID cannot duplicate model work.
      outbox.clear();
      highestAcknowledged = 0;
      sequence = 0;
      announceReady();
      return;
    }
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
  const cancelServerSettledCommand = (command) => {
    const handlers = globalThis.__QWEN_LOCAL_CONTROL__;
    if (!handlers || typeof handlers !== "object") return;
    const handlerName = command.command === "runPrompt"
      ? "cancelPrompt"
      : command.command === "load" ||
          command.command === "dispose" ||
          command.command === "warmReload"
        ? "dispose"
        : undefined;
    if (handlerName === undefined) return;
    const handler = handlers[handlerName];
    if (typeof handler !== "function") return;
    void Promise.resolve().then(() => handler()).catch(() => {});
  };
  const transition = (commandId, state, reason, result) => {
    const previous = records.get(commandId);
    // Handler settlement races with server timeout/reload reconciliation. Once
    // adopted, only an explicit duplicate-command report may emit that server
    // terminal; local completion cannot overwrite it.
    if (previous && previous.settledByServer && previous.state !== state) return;
    records.set(commandId, {
      state,
      reason,
      result,
      ...(previous && previous.settledByServer ? { settledByServer: true } : {})
    });
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
  const safeRuntimeDiagnosticCode = (error) => {
    if ((!error || typeof error !== "object") && typeof error !== "function") return undefined;
    try {
      if (!Object.hasOwn(error, "code")) return undefined;
      if (typeof error.code !== "string") return "unknown";
      // Runtime errors use fixed machine codes. Never forward the message,
      // name, stack, or any free-form browser/compiler text.
      return ALLOCATION_DIAGNOSTIC_CODES.has(error.code) || SAFE_DIAGNOSTIC_CODE.test(error.code)
        ? error.code : "unknown";
    } catch {
      return "unknown";
    }
  };
  const safeLoadEvent = (value) => {
    if (!value || typeof value !== "object" || Array.isArray(value) || !LOAD_PHASES.has(value.phase)) {
      return undefined;
    }
    if (
      !Number.isSafeInteger(value.completedBytes) || value.completedBytes < 0 ||
      !Number.isSafeInteger(value.totalBytes) || value.totalBytes < 0 ||
      value.completedBytes > value.totalBytes
    ) return undefined;
    const result = {
      phase: value.phase,
      completedBytes: value.completedBytes,
      totalBytes: value.totalBytes
    };
    if (
      Number.isSafeInteger(value.currentGpuBytes) && value.currentGpuBytes >= 0 &&
      Number.isSafeInteger(value.peakGpuBytes) && value.peakGpuBytes >= 0 &&
      value.currentGpuBytes <= value.peakGpuBytes
    ) {
      result.currentGpuBytes = value.currentGpuBytes;
      result.peakGpuBytes = value.peakGpuBytes;
    }
    if (
      Number.isSafeInteger(value.shardIndex) && value.shardIndex >= 0 &&
      Number.isSafeInteger(value.shardCount) && value.shardCount > 0 &&
      value.shardIndex < value.shardCount
    ) {
      result.shardIndex = value.shardIndex;
      result.shardCount = value.shardCount;
    }
    return result;
  };
  const safeUploadEvent = (value) => {
    if (
      !value || typeof value !== "object" || Array.isArray(value) ||
      !UPLOAD_STAGES.has(value.stage) ||
      !Number.isSafeInteger(value.ordinal) || value.ordinal <= 0 ||
      !Number.isSafeInteger(value.shardIndex) || value.shardIndex < 0 ||
      !Number.isSafeInteger(value.shardCount) || value.shardCount <= 0 ||
      value.shardIndex >= value.shardCount ||
      !Number.isSafeInteger(value.segmentIndex) || value.segmentIndex < 0 ||
      !Number.isSafeInteger(value.segmentCount) || value.segmentCount <= 0 ||
      value.segmentIndex >= value.segmentCount ||
      !Number.isSafeInteger(value.globalOffset) || value.globalOffset < 0 ||
      value.globalOffset % 4 !== 0 ||
      !Number.isSafeInteger(value.byteCount) || value.byteCount <= 0 ||
      value.byteCount % 4 !== 0 ||
      !Number.isSafeInteger(value.globalOffset + value.byteCount) ||
      !Number.isSafeInteger(value.bufferShardBytes) || value.bufferShardBytes <= 0 ||
      value.bufferShardBytes % 4 !== 0 ||
      !Number.isSafeInteger(value.uploadLaneBytes) || value.uploadLaneBytes <= 0 ||
      value.uploadLaneBytes % 4 !== 0 ||
      value.byteCount > value.uploadLaneBytes ||
      typeof value.retireAfterEachWrite !== "boolean"
    ) return undefined;
    return {
      stage: value.stage,
      ordinal: value.ordinal,
      shardIndex: value.shardIndex,
      shardCount: value.shardCount,
      segmentIndex: value.segmentIndex,
      segmentCount: value.segmentCount,
      globalOffset: value.globalOffset,
      byteCount: value.byteCount,
      bufferShardBytes: value.bufferShardBytes,
      uploadLaneBytes: value.uploadLaneBytes,
      retireAfterEachWrite: value.retireAfterEachWrite
    };
  };
  const safeRuntimeMetrics = (value) => {
    if (!value || typeof value !== "object" || Array.isArray(value)) return undefined;
    const nonnegativeInteger = (entry) =>
      Number.isSafeInteger(entry) && entry >= 0;
    const nonnegativeNumber = (entry) =>
      typeof entry === "number" && Number.isFinite(entry) && entry >= 0;
    const nullableNumber = (entry) => entry === null || nonnegativeNumber(entry);
    const safePerformance = (candidate) => {
      if (!candidate || typeof candidate !== "object" || Array.isArray(candidate)) return undefined;
      const counters = [
        "diskReadBytes",
        "gpuUploadBytes",
        "dispatchCount",
        "queueSubmissionCount",
        "queueRetirementCount",
        "gpuReadbackCount"
      ];
      const output = {};
      for (const counter of counters) {
        if (!nonnegativeInteger(candidate[counter])) return undefined;
        output[counter] = candidate[counter];
      }
      return output;
    };
    if (
      !nonnegativeInteger(value.contextTokens) ||
      !nonnegativeInteger(value.trackedCpuBytes) ||
      !nonnegativeInteger(value.trackedGpuBytes) ||
      !nonnegativeInteger(value.peakTrackedGpuBytes) ||
      value.trackedGpuBytes > value.peakTrackedGpuBytes ||
      !nonnegativeInteger(value.prefillTokens) ||
      !nonnegativeNumber(value.prefillDurationMilliseconds) ||
      !nullableNumber(value.prefillTokensPerSecond) ||
      !nonnegativeInteger(value.generatedTokens) ||
      !nonnegativeInteger(value.targetStepCount) ||
      value.targetStepCount > value.generatedTokens ||
      !nonnegativeNumber(value.targetStepDurationMilliseconds) ||
      !nullableNumber(value.targetStepsPerSecond) ||
      !nonnegativeNumber(value.generationDurationMilliseconds) ||
      !nullableNumber(value.generatedTokensPerSecond) ||
      !nullableNumber(value.timeToFirstTokenMilliseconds) ||
      !nonnegativeInteger(value.decodedTextCodeUnits) ||
      !nonnegativeInteger(value.referenceTokenCount) ||
      !nonnegativeInteger(value.referenceTokenMismatchCount) ||
      value.referenceTokenMismatchCount > value.referenceTokenCount ||
      !(
        value.referenceFirstMismatchIndex === undefined ||
        (nonnegativeInteger(value.referenceFirstMismatchIndex) &&
          value.referenceFirstMismatchIndex < value.referenceTokenCount &&
          nonnegativeInteger(value.referenceExpectedTokenId) &&
          nonnegativeInteger(value.referenceObservedTokenId))
      ) ||
      !nonnegativeInteger(value.performanceSnapshotCount) ||
      value.performanceSnapshotCount > 2
    ) return undefined;
    const performance = value.performance === undefined
      ? undefined
      : safePerformance(value.performance);
    if (value.performance !== undefined && performance === undefined) return undefined;
    return {
      contextTokens: value.contextTokens,
      trackedCpuBytes: value.trackedCpuBytes,
      trackedGpuBytes: value.trackedGpuBytes,
      peakTrackedGpuBytes: value.peakTrackedGpuBytes,
      prefillTokens: value.prefillTokens,
      prefillDurationMilliseconds: value.prefillDurationMilliseconds,
      prefillTokensPerSecond: value.prefillTokensPerSecond,
      generatedTokens: value.generatedTokens,
      targetStepCount: value.targetStepCount,
      targetStepDurationMilliseconds: value.targetStepDurationMilliseconds,
      targetStepsPerSecond: value.targetStepsPerSecond,
      generationDurationMilliseconds: value.generationDurationMilliseconds,
      generatedTokensPerSecond: value.generatedTokensPerSecond,
      timeToFirstTokenMilliseconds: value.timeToFirstTokenMilliseconds,
      decodedTextCodeUnits: value.decodedTextCodeUnits,
      referenceTokenCount: value.referenceTokenCount,
      referenceTokenMismatchCount: value.referenceTokenMismatchCount,
      ...(value.referenceFirstMismatchIndex === undefined ? {} : {
        referenceFirstMismatchIndex: value.referenceFirstMismatchIndex,
        referenceExpectedTokenId: value.referenceExpectedTokenId,
        referenceObservedTokenId: value.referenceObservedTokenId
      }),
      performanceSnapshotCount: value.performanceSnapshotCount,
      ...(performance === undefined ? {} : { performance })
    };
  };
  const sendRuntimeMetrics = (metrics) => {
    const commonMemory = {
      cpuBytes: metrics.trackedCpuBytes,
      gpuBytes: metrics.trackedGpuBytes,
      peakBytes: metrics.peakTrackedGpuBytes
    };
    // Correctness and decode counters are the terminal record. Send them first
    // so outbox pressure cannot preserve a duplicate rate event while dropping
    // the oracle result.
    const events = [{
      category: "generation",
      name: "generation_completed",
      metrics: {
        durationMs: metrics.generationDurationMilliseconds,
        count: metrics.generatedTokens,
        targetStepCount: metrics.targetStepCount,
        targetStepDurationMs: metrics.targetStepDurationMilliseconds,
        ...(metrics.generatedTokensPerSecond === null ? {} : {
          emittedTokensPerSecond: metrics.generatedTokensPerSecond
        }),
        ...(metrics.timeToFirstTokenMilliseconds === null ? {} : {
          ttftMs: metrics.timeToFirstTokenMilliseconds
        }),
        ...(metrics.targetStepsPerSecond === null ? {} : {
          tokensPerSecond: metrics.targetStepsPerSecond
        }),
        contextTokens: metrics.contextTokens,
        decodedTextCodeUnits: metrics.decodedTextCodeUnits,
        referenceTokenCount: metrics.referenceTokenCount,
        referenceTokenMismatchCount: metrics.referenceTokenMismatchCount,
        ...(metrics.referenceFirstMismatchIndex === undefined ? {} : {
          referenceFirstMismatchIndex: metrics.referenceFirstMismatchIndex,
          referenceExpectedTokenId: metrics.referenceExpectedTokenId,
          referenceObservedTokenId: metrics.referenceObservedTokenId
        }),
        performanceSnapshotCount: metrics.performanceSnapshotCount,
        ...(metrics.performance === undefined ? {} : metrics.performance),
        ...commonMemory
      }
    }, {
      category: "prefill",
      name: "prefill_completed",
      metrics: {
        durationMs: metrics.prefillDurationMilliseconds,
        prefillTokens: metrics.prefillTokens,
        ...(metrics.prefillTokensPerSecond === null ? {} : {
          prefillTokensPerSecond: metrics.prefillTokensPerSecond
        }),
        ...commonMemory
      }
    }];
    for (const event of events) {
      // Preserve one outbox position for the command terminal transition.
      if (outbox.size >= OUTBOX_LIMIT - 1) return;
      try {
        send({
          type: "telemetry",
          event: { ...event, timestampMs: Date.now() }
        });
      } catch {
        return;
      }
    }
  };
  const shouldSendLoadEvent = (event) => {
    const previous = lastLoadTelemetry;
    if (!previous || previous.phase !== event.phase) return true;
    if (previous.shardIndex !== event.shardIndex || previous.shardCount !== event.shardCount) return true;
    if (event.totalBytes === 0) return false;
    if (event.completedBytes === event.totalBytes && previous.completedBytes !== event.completedBytes) return true;
    return event.completedBytes - previous.completedBytes >= LOAD_TELEMETRY_STEP_BYTES;
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
    if (message.command === "load" && activeLoadCommandId !== undefined) {
      transition(message.commandId, "failed", "load_in_progress");
      return;
    }
    const ownsLoad = message.command === "load";
    if (ownsLoad) activeLoadCommandId = message.commandId;
    if (message.command === "warmReload") {
      if (reloadScheduled) {
        transition(message.commandId, "failed", "reload_in_progress");
        return;
      }
      reloadScheduled = true;
      const handlers = globalThis.__QWEN_LOCAL_CONTROL__;
      if (!handlers || typeof handlers.dispose !== "function") {
        reloadScheduled = false;
        transition(message.commandId, "failed", "handler_unavailable");
        return;
      }
      try {
        // Retire GPU ownership before navigation. Safari can keep the old
        // document alive long enough for a replacement load to overlap it.
        await handlers.dispose();
      } catch {
        reloadScheduled = false;
        transition(message.commandId, "failed", "handler_failed");
        return;
      }
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
    } catch (error) {
      if (promptRecord && promptRecord.cancellation) {
        try {
          await promptRecord.cancellation;
          transition(message.commandId, "cancelled", "operator_cancelled");
        } catch {
          transition(message.commandId, "failed", "handler_failed");
        }
        return;
      }
      const diagnosticCode = safeRuntimeDiagnosticCode(error);
      if (diagnosticCode !== undefined) {
        try {
          send({
            type: "telemetry",
            event: {
              category: "error",
              name: "runtime_error",
              timestampMs: Date.now(),
              metrics: { code: diagnosticCode }
            }
          });
        } catch {}
      }
      transition(
        message.commandId,
        message.command === "cancelPrompt" ? "cancelled" : "failed",
        diagnosticCode ?? "handler_failed"
      );
    } finally {
      if (activePrompt === promptRecord) activePrompt = undefined;
      if (ownsLoad && activeLoadCommandId === message.commandId) {
        activeLoadCommandId = undefined;
      }
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
    announceReady();
  };
  const announceReady = () => {
    send({ type: "ready" });
    // The first local journal record joins server-derived socket metadata to
    // durable browser IDs even when the runtime has not produced metrics yet.
    send({
      type: "telemetry",
      event: {
        category: "device",
        name: "connected",
        timestampMs: Date.now(),
        metrics: { status: "connected" }
      }
    });
  };
  const connectOnce = async () => {
    try {
      const { ticket } = await requestJson("/.local-ticket", identity());
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
        if (candidate !== socket || identifiedSocket !== candidate) return;
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
            if (local) {
              if (TERMINAL_STATES.has(command.state)) {
                // A server terminal settles both conflicting terminal state
                // and work that is still running locally.
                const wasTerminal = TERMINAL_STATES.has(local.state);
                records.set(command.commandId, {
                  state: command.state,
                  reason: command.reason,
                  result: command.result,
                  settledByServer: true
                });
                if (!wasTerminal) cancelServerSettledCommand(command);
                continue;
              }
              if (local.state !== command.state) resendState(command.commandId, local);
              continue;
            }
            if (!local && command.state !== "issued" && command.state !== "accepted") {
              records.set(command.commandId, {
                state: command.state,
                reason: command.reason,
                result: command.result,
                ...(TERMINAL_STATES.has(command.state) ? { settledByServer: true } : {})
              });
            }
          }
        }
      });
      candidate.addEventListener("close", () => {
        if (identifiedSocket === candidate) identifiedSocket = undefined;
        if (socket === candidate) socket = undefined;
        setTimeout(() => void connect(), reconnectDelay);
        reconnectDelay = Math.min(5000, reconnectDelay * 2);
      });
    } catch {
      // This can fail while the local server is reloading; retry only through
      // a fresh same-origin request so an old WSS ticket never survives it.
      setTimeout(() => void connect(), reconnectDelay);
      reconnectDelay = Math.min(5000, reconnectDelay * 2);
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
  addEventListener("qwen-local-runtime-ready", () => {
    if (typeof globalThis.__QWEN_LOCAL_CONTROL__ !== "object") return;
    runtimeReady = true;
    if (socket) identifyIfRuntimeReady(socket);
  });
  addEventListener("qwen-local-runtime-load-event", ({ detail }) => {
    const event = safeLoadEvent(detail);
    if (!event || !shouldSendLoadEvent(event)) return;
    const terminal = event.phase === "ready" || event.phase === "failed";
    // Progress cannot consume the slots needed for a terminal load event and
    // the command state that follows it when acknowledgements pause.
    if (!terminal && outbox.size >= LOAD_TELEMETRY_OUTBOX_LIMIT) return;
    try {
      // Model loading must continue when the bounded unacknowledged outbox is full.
      send({
        type: "telemetry",
        event: {
          category: "phase",
          name: event.phase === "ready"
            ? "load_completed"
            : event.phase === "failed"
              ? "load_failed"
              : "load_started",
          timestampMs: Date.now(),
          metrics: event
        }
      });
      // Advance only after retention succeeds so a dropped progress sample
      // cannot suppress the later terminal event.
      lastLoadTelemetry = event;
    } catch {}
  });
  addEventListener("qwen-local-runtime-upload-event", ({ detail }) => {
    const event = safeUploadEvent(detail);
    if (!event || outbox.size >= LOAD_TELEMETRY_OUTBOX_LIMIT) return;
    try {
      // Upload probes are progress evidence and cannot consume the slots
      // reserved for load termination and command completion.
      send({
        type: "telemetry",
        event: {
          category: "upload",
          name: event.stage,
          timestampMs: Date.now(),
          metrics: {
            ordinal: event.ordinal,
            shardIndex: event.shardIndex,
            shardCount: event.shardCount,
            segmentIndex: event.segmentIndex,
            segmentCount: event.segmentCount,
            globalOffset: event.globalOffset,
            byteCount: event.byteCount,
            bufferShardBytes: event.bufferShardBytes,
            uploadLaneBytes: event.uploadLaneBytes,
            retireAfterEachWrite: event.retireAfterEachWrite
          }
        }
      });
    } catch {}
  });
  addEventListener("qwen-local-runtime-metrics", ({ detail }) => {
    const metrics = safeRuntimeMetrics(detail);
    if (metrics) sendRuntimeMetrics(metrics);
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
