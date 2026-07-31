import { Qwen35Session } from "../../src/qwen35-session.js";
import type { Qwen35BrowserLoadOptions } from "../../src/qwen35-model-loader.js";
import {
  createTextRuntimeController,
  type TextRuntimeController,
} from "./text-runtime.js";

declare global {
  // The injected development agent reads only this explicit command surface.
  // Local model pins remain closed over by the controller and cannot be
  // replaced by an operator command payload.
  var __QWEN_LOCAL_CONTROL__: TextRuntimeController | undefined;
}

const requireElement = <T extends HTMLElement>(id: string): T => {
  const element = document.getElementById(id);
  if (element === null) throw new Error(`Required application element is missing: ${id}`);
  return element as T;
};

const fetchRuntimeConfiguration = async (): Promise<Qwen35BrowserLoadOptions> => {
  const response = await fetch("/.local-runtime-config.json", {
    method: "GET",
    cache: "no-store",
    credentials: "omit",
  });
  if (!response.ok) throw new Error("Local runtime configuration is unavailable");
  const input: unknown = await response.json();
  if (typeof input !== "object" || input === null || Array.isArray(input)) {
    throw new Error("Local runtime configuration is invalid");
  }
  return input as Qwen35BrowserLoadOptions;
};

const status = requireElement<HTMLOutputElement>("runtime-status");
const output = requireElement<HTMLPreElement>("runtime-output");
const prompt = requireElement<HTMLTextAreaElement>("runtime-prompt");
const maxTokens = requireElement<HTMLInputElement>("runtime-max-tokens");
const configuration = await fetchRuntimeConfiguration();
const controller = createTextRuntimeController({
  session: new Qwen35Session(),
  loadOptions: configuration,
  onText(text) {
    output.textContent += text;
  },
});
globalThis.__QWEN_LOCAL_CONTROL__ = controller;

const reportState = (): void => {
  status.value = JSON.stringify(controller.getState());
};

const runUiAction = async (action: () => Promise<void>): Promise<void> => {
  status.value = "working";
  try {
    await action();
    reportState();
  } catch {
    status.value = "operation failed";
  }
};

requireElement<HTMLButtonElement>("runtime-load").addEventListener("click", () => {
  void runUiAction(() => controller.load());
});
requireElement<HTMLButtonElement>("runtime-run").addEventListener("click", () => {
  output.textContent = "";
  void runUiAction(() => controller.runPrompt({
    prompt: prompt.value,
    maxNewTokens: Number(maxTokens.value),
  }));
});
requireElement<HTMLButtonElement>("runtime-cancel").addEventListener("click", () => {
  void runUiAction(() => controller.cancelPrompt());
});
requireElement<HTMLButtonElement>("runtime-dispose").addEventListener("click", () => {
  void runUiAction(() => controller.dispose());
});
requireElement<HTMLButtonElement>("runtime-state").addEventListener("click", reportState);
reportState();
dispatchEvent(new CustomEvent("qwen-local-runtime-ready"));
