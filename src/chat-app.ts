import {
  Qwen35Session,
  type RuntimeMetrics,
} from "./qwen35-session.js";
import type { Qwen35ChatContentPart, Qwen35ChatMessage } from "./qwen-chat-template.js";
import {
  QWEN35_DEFAULT_DECODED_SOURCE_BYTE_BUDGET,
  planQwen35VisionImage,
  preprocessQwen35VisionRgb,
  type Qwen35VisionPatchBatch,
} from "./qwen35-vision-preprocess.js";
import type { ModelPackageManifest } from "./manifest.js";
import type { Qwen35VisionPackagePins } from "./qwen35-vision-package-loader.js";
import { presentChatGenerationFailure } from "./chat-app-state.js";

const VISION_SETTINGS = Object.freeze({
  processorClass: "Qwen3VLProcessor",
  imageProcessorType: "Qwen2VLImageProcessorFast",
  patchSize: 16,
  temporalPatchSize: 2,
  mergeSize: 2,
  shortestEdge: 65_536,
  longestEdge: 16_777_216,
  imageMean: [0.5, 0.5, 0.5] as const,
  imageStd: [0.5, 0.5, 0.5] as const,
});

interface RuntimeConfig {
  readonly manifest?: ModelPackageManifest;
  readonly manifestUrl?: string;
  readonly allowInsecureLocalhost?: boolean;
  /** Local/device tuning may lower resolution after measured stability failures. */
  readonly maxVisualTokens?: number;
  /** Local smoke/device tuning may lower the processor's minimum edge. */
  readonly visionShortestEdge?: number;
  readonly visionPackagePins?: Qwen35VisionPackagePins;
  readonly packageBaseUrl: string;
  readonly expectedPackageBaseUrl: string;
  readonly expectedManifestSha256: string;
  readonly compiledTokenizerUrl: string;
}

interface RuntimeConfigWindow extends Window {
  readonly __QWEN35_RUNTIME_CONFIG__?: Partial<RuntimeConfig>;
}

interface PendingImage {
  readonly file: File;
  readonly previewUrl: string;
  readonly batch: Qwen35VisionPatchBatch;
}

const root = document.querySelector<HTMLElement>("#qwen-app");
if (root !== null) {
  void startApp(root);
}

function element<T extends Element>(selector: string): T {
  const value = document.querySelector<T>(selector);
  if (value === null) throw new Error(`Missing application element: ${selector}`);
  return value;
}

function statusText(value: string): void {
  element<HTMLElement>("[data-runtime-status]").textContent = value;
}

function setBusy(value: boolean): void {
  for (const control of document.querySelectorAll<HTMLButtonElement | HTMLTextAreaElement | HTMLInputElement>("[data-composer-control]")) {
    if (control.matches("[data-stop]")) {
      control.disabled = !value;
    } else {
      control.disabled = value;
    }
  }
  root?.toggleAttribute("data-busy", value);
}

function formatNumber(value: number): string {
  return new Intl.NumberFormat(undefined, { maximumFractionDigits: 0 }).format(value);
}

function formatRate(value: number | null): string {
  return value === null || !Number.isFinite(value) ? "—" : `${value.toFixed(1)} tok/s`;
}

function updateMetrics(metrics: RuntimeMetrics): void {
  const context = element<HTMLElement>("[data-metric-context]");
  const speed = element<HTMLElement>("[data-metric-speed]");
  const memory = element<HTMLElement>("[data-metric-memory]");
  const phase = element<HTMLElement>("[data-metric-phase]");
  context.textContent = `${formatNumber(metrics.contextTokens)} / 16,384`;
  speed.textContent = formatRate(metrics.generatedTokensPerSecond);
  memory.textContent = metrics.trackedGpuBytes > 0
    ? `${(metrics.trackedGpuBytes / (1024 ** 3)).toFixed(2)} GB`
    : "—";
  phase.textContent = metrics.state;
}

function appendMessage(role: "user" | "assistant", text: string, image?: PendingImage): HTMLElement {
  const item = document.createElement("article");
  item.className = `message message--${role}`;
  item.dataset.role = role;
  const header = document.createElement("div");
  header.className = "message__header";
  header.textContent = role === "user" ? "You" : "Qwen 3.5";
  const body = document.createElement("div");
  body.className = "message__body";
  if (image !== undefined) {
    const imageElement = document.createElement("img");
    imageElement.src = image.previewUrl;
    imageElement.alt = "Attached image";
    imageElement.className = "message__image";
    body.append(imageElement);
  }
  const copy = document.createElement("p");
  copy.className = "message__text";
  copy.textContent = text;
  body.append(copy);
  item.append(header, body);
  element<HTMLElement>("[data-messages]").append(item);
  item.scrollIntoView({ block: "end", behavior: "smooth" });
  return item;
}

function appendNotice(text: string, kind: "info" | "error" = "info"): void {
  const notice = document.createElement("p");
  notice.className = `notice notice--${kind}`;
  notice.textContent = text;
  element<HTMLElement>("[data-messages]").append(notice);
  notice.scrollIntoView({ block: "end", behavior: "smooth" });
}

function setPanel(open: boolean): void {
  const panel = element<HTMLElement>("[data-settings-panel]");
  panel.toggleAttribute("data-open", open);
  panel.setAttribute("aria-hidden", String(!open));
  element<HTMLButtonElement>("[data-settings-toggle]").setAttribute("aria-expanded", String(open));
}

function readRuntimeConfig(): RuntimeConfig | null {
  const injected = (window as RuntimeConfigWindow).__QWEN35_RUNTIME_CONFIG__;
  if (injected === undefined) return null;
  const packageBaseUrl = injected.packageBaseUrl;
  const expectedPackageBaseUrl = injected.expectedPackageBaseUrl;
  const expectedManifestSha256 = injected.expectedManifestSha256;
  const compiledTokenizerUrl = injected.compiledTokenizerUrl;
  const maxVisualTokens = injected.maxVisualTokens;
  const visionShortestEdge = injected.visionShortestEdge;
  const hasManifest = typeof injected.manifest === "object" && injected.manifest !== null;
  const hasManifestUrl = typeof injected.manifestUrl === "string" && injected.manifestUrl.length > 0;
  if (
    (!hasManifest && !hasManifestUrl) ||
    typeof packageBaseUrl !== "string" || packageBaseUrl.length === 0 ||
    typeof expectedPackageBaseUrl !== "string" || expectedPackageBaseUrl.length === 0 ||
    typeof expectedManifestSha256 !== "string" || expectedManifestSha256.length === 0 ||
    typeof compiledTokenizerUrl !== "string" || compiledTokenizerUrl.length === 0
  ) {
    return null;
  }
  return Object.freeze({
    ...(hasManifest ? { manifest: injected.manifest as ModelPackageManifest } : {}),
    ...(hasManifestUrl ? { manifestUrl: injected.manifestUrl } : {}),
    ...(injected.allowInsecureLocalhost === true ? { allowInsecureLocalhost: true } : {}),
    ...(typeof maxVisualTokens === "number" && Number.isSafeInteger(maxVisualTokens) && maxVisualTokens >= 1 && maxVisualTokens <= 16_384
      ? { maxVisualTokens: maxVisualTokens as number }
      : {}),
    ...(typeof visionShortestEdge === "number" && Number.isSafeInteger(visionShortestEdge) && visionShortestEdge >= 1 && visionShortestEdge <= 16_777_216
      ? { visionShortestEdge: visionShortestEdge as number }
      : {}),
    ...(typeof injected.visionPackagePins === "object" && injected.visionPackagePins !== null
      ? { visionPackagePins: injected.visionPackagePins as Qwen35VisionPackagePins }
      : {}),
    packageBaseUrl,
    expectedPackageBaseUrl,
    expectedManifestSha256,
    compiledTokenizerUrl,
  });
}

async function loadManifest(config: RuntimeConfig): Promise<ModelPackageManifest> {
  if (config.manifest !== undefined) return config.manifest;
  if (config.manifestUrl === undefined) throw new Error("The pinned language manifest is not configured.");
  const response = await fetch(config.manifestUrl, { cache: "no-store" });
  if (!response.ok) throw new Error("The pinned language manifest could not be loaded.");
  const value: unknown = await response.json();
  if (typeof value !== "object" || value === null) throw new Error("The language manifest is invalid.");
  return value as ModelPackageManifest;
}

async function imageToRgb(file: File, signal: AbortSignal): Promise<{ readonly rgb: Uint8Array; readonly width: number; readonly height: number }> {
  signal.throwIfAborted();
  const bitmap = await createImageBitmap(file);
  try {
    const canvas = document.createElement("canvas");
    canvas.width = bitmap.width;
    canvas.height = bitmap.height;
    const context = canvas.getContext("2d", { willReadFrequently: true });
    if (context === null) throw new Error("The browser could not create an image surface.");
    context.drawImage(bitmap, 0, 0);
    const pixels = context.getImageData(0, 0, bitmap.width, bitmap.height).data;
    const rgb = new Uint8Array(bitmap.width * bitmap.height * 3);
    for (let source = 0, target = 0; target < rgb.length; source += 4, target += 3) {
      rgb[target] = pixels[source]!;
      rgb[target + 1] = pixels[source + 1]!;
      rgb[target + 2] = pixels[source + 2]!;
    }
    return Object.freeze({ rgb, width: bitmap.width, height: bitmap.height });
  } finally {
    bitmap.close();
  }
}

async function preprocessImage(
  file: File,
  signal: AbortSignal,
  maxVisualTokens?: number,
  visionShortestEdge?: number,
): Promise<Qwen35VisionPatchBatch> {
  const source = await imageToRgb(file, signal);
  const plan = planQwen35VisionImage({
    sourceHeight: source.height,
    sourceWidth: source.width,
    settings: visionShortestEdge === undefined
      ? VISION_SETTINGS
      : { ...VISION_SETTINGS, shortestEdge: visionShortestEdge },
    ...(maxVisualTokens === undefined ? {} : { maxVisualTokens }),
  });
  return preprocessQwen35VisionRgb({
    plan,
    sourceRgb: source.rgb,
    sourceHeight: source.height,
    sourceWidth: source.width,
    decodedSourceByteBudget: QWEN35_DEFAULT_DECODED_SOURCE_BYTE_BUDGET,
    signal,
  });
}

function contentForUser(text: string, image: PendingImage | null): readonly Qwen35ChatContentPart[] | string {
  if (image === null) return text;
  const parts: Qwen35ChatContentPart[] = [{ type: "image", patches: image.batch }];
  if (text.length > 0) parts.push({ type: "text", text });
  return parts;
}

async function startApp(container: HTMLElement): Promise<void> {
  const messages: Qwen35ChatMessage[] = [];
  let session: Qwen35Session | null = null;
  let runtimeConfig: RuntimeConfig | null = readRuntimeConfig();
  let pendingImage: PendingImage | null = null;
  let preprocessController: AbortController | null = null;
  let sendController: AbortController | null = null;
  let hasPrefilled = false;
  let generationLimit = 192;

  const renderImagePreview = (): void => {
    const slot = element<HTMLElement>("[data-image-preview]");
    slot.replaceChildren();
    if (pendingImage === null) {
      slot.hidden = true;
      return;
    }
    slot.hidden = false;
    slot.classList.add("preview");
    const image = document.createElement("img");
    image.src = pendingImage.previewUrl;
    image.alt = "Selected image preview";
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "preview__remove";
    remove.textContent = "Remove";
    remove.setAttribute("data-composer-control", "true");
    remove.addEventListener("click", () => {
      if (pendingImage !== null) URL.revokeObjectURL(pendingImage.previewUrl);
      pendingImage = null;
      renderImagePreview();
    });
    slot.append(image, remove);
  };

  const ensureSession = async (): Promise<Qwen35Session> => {
    if (session !== null) return session;
    if (runtimeConfig === null) {
      throw new Error("Runtime config is not installed. Set the local model package configuration first.");
    }
    statusText("Authenticating package");
    const manifest = await loadManifest(runtimeConfig);
    const created = new Qwen35Session();
    session = created;
    await created.load({
      manifest,
      packageBaseUrl: runtimeConfig.packageBaseUrl,
      expectedPackageBaseUrl: runtimeConfig.expectedPackageBaseUrl,
      expectedManifestSha256: runtimeConfig.expectedManifestSha256,
      compiledTokenizerUrl: runtimeConfig.compiledTokenizerUrl,
      ...(runtimeConfig.allowInsecureLocalhost === true
        ? { allowInsecureLocalhost: true }
        : {}),
      ...(runtimeConfig.visionPackagePins === undefined
        ? {}
        : { visionPackagePins: runtimeConfig.visionPackagePins }),
    });
    statusText("Ready for a prompt");
    updateMetrics(created.getMetrics());
    return created;
  };

  const send = async (): Promise<void> => {
    const prompt = element<HTMLTextAreaElement>("[data-prompt]");
    const text = prompt.value.trim();
    if (text.length === 0 && pendingImage === null) return;
    setBusy(true);
    const controller = new AbortController();
    sendController = controller;
    const image = pendingImage;
    const userContent = contentForUser(text, image);
    const userMessage: Qwen35ChatMessage = { role: "user", content: userContent };
    const nextMessages = [...messages, userMessage];
    appendMessage("user", text.length > 0 ? text : "Describe this image.", image ?? undefined);
    prompt.value = "";
    pendingImage = null;
    renderImagePreview();
    let loaded: Qwen35Session | null = null;
    try {
      loaded = await ensureSession();
      if (hasPrefilled) await loaded.reset();
      const state = await loaded.prefill(nextMessages);
      messages.push(userMessage);
      hasPrefilled = true;
      element<HTMLElement>("[data-context-copy]").textContent = `${formatNumber(state.contextTokens)} context tokens`;
      const assistant = appendMessage("assistant", "");
      const output = assistant.querySelector<HTMLElement>(".message__text")!;
      let response = "";
      for await (const token of loaded.generate({ maxNewTokens: generationLimit, temperature: 0, topK: 1 })) {
        response += token.text;
        output.textContent = response;
        updateMetrics(loaded.getMetrics());
        assistant.scrollIntoView({ block: "end", behavior: "smooth" });
      }
      messages.push({ role: "assistant", content: response });
      statusText("Ready for a prompt");
      updateMetrics(loaded.getMetrics());
    } catch (error) {
      const failedSession = loaded ?? session;
      const presentation = presentChatGenerationFailure(error, {
        cancellationRequested: controller.signal.aborted,
        runtimeFailed: failedSession?.state === "failed",
      });
      if (failedSession?.state === "failed") {
        await failedSession.dispose().catch(() => undefined);
        updateMetrics(failedSession.getMetrics());
        session = null;
        hasPrefilled = false;
      }
      appendNotice(
        presentation.notice,
        presentation.kind === "error" ? "error" : "info",
      );
      statusText(presentation.status);
    } finally {
      sendController = null;
      setBusy(false);
    }
  };

  const chooseImage = async (file: File): Promise<void> => {
    if (!file.type.startsWith("image/")) {
      appendNotice("Choose a PNG, JPEG, WebP, or other browser-decodable image.", "error");
      return;
    }
    preprocessController?.abort();
    const controller = new AbortController();
    preprocessController = controller;
    statusText("Preparing image");
    try {
      const batch = await preprocessImage(
        file,
        controller.signal,
        runtimeConfig?.maxVisualTokens,
        runtimeConfig?.visionShortestEdge,
      );
      if (controller.signal.aborted) return;
      if (pendingImage !== null) URL.revokeObjectURL(pendingImage.previewUrl);
      pendingImage = Object.freeze({ file, previewUrl: URL.createObjectURL(file), batch });
      renderImagePreview();
      statusText("Image ready");
    } catch (error) {
      if (!controller.signal.aborted) appendNotice(error instanceof Error ? error.message : "Image preparation failed.", "error");
      statusText("Ready for a prompt");
    } finally {
      if (preprocessController === controller) preprocessController = null;
    }
  };

  element<HTMLFormElement>("[data-composer]").addEventListener("submit", (event) => {
    event.preventDefault();
    void send();
  });
  element<HTMLButtonElement>("[data-stop]").addEventListener("click", () => {
    const activeSession = session;
    sendController?.abort();
    if (activeSession !== null) void activeSession.cancel().catch(() => undefined);
  });
  element<HTMLButtonElement>("[data-image-button]").addEventListener("click", () => element<HTMLInputElement>("[data-image-input]").click());
  element<HTMLInputElement>("[data-image-input]").addEventListener("change", (event) => {
    const file = (event.target as HTMLInputElement).files?.[0];
    if (file !== undefined) void chooseImage(file);
  });
  document.addEventListener("paste", (event) => {
    const file = [...(event.clipboardData?.items ?? [])].find((item) => item.type.startsWith("image/"))?.getAsFile();
    if (file !== null && file !== undefined) void chooseImage(file);
  });
  element<HTMLButtonElement>("[data-settings-toggle]").addEventListener("click", () => {
    const panel = element<HTMLElement>("[data-settings-panel]");
    setPanel(!panel.hasAttribute("data-open"));
  });
  element<HTMLButtonElement>("[data-settings-close]").addEventListener("click", () => setPanel(false));
  element<HTMLInputElement>("[data-generation-limit]").addEventListener("change", (event) => {
    const value = Number((event.target as HTMLInputElement).value);
    if (Number.isSafeInteger(value) && value >= 1 && value <= 2_048) generationLimit = value;
  });
  element<HTMLButtonElement>("[data-new-chat]").addEventListener("click", async () => {
    await session?.reset().catch(() => undefined);
    messages.length = 0;
    hasPrefilled = false;
    element<HTMLElement>("[data-messages]").replaceChildren();
    appendNotice("New conversation ready.");
    setPanel(false);
  });
  element<HTMLTextAreaElement>("[data-prompt]").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void send();
    }
  });

  // Configuration is deliberately injected at runtime. No local address,
  // operator token, cache path, or private model URL enters the Pages build.
  if (runtimeConfig === null) {
    statusText("Runtime config not installed");
    appendNotice("Add a public model package config to start the Chrome validation loop.");
  } else {
    void ensureSession().catch((error) => {
      statusText("Load failed");
      appendNotice(error instanceof Error ? error.message : "Model load failed.", "error");
    });
  }

  window.addEventListener("beforeunload", () => {
    preprocessController?.abort();
    sendController?.abort();
    void session?.dispose();
  });
}
