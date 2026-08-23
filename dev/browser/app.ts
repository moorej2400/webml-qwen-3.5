import {
  startQwen35ChatApp,
} from "../../src/chat-app.js";
import type { BufferShardPolicy, WebGpuProbeSurface } from "../../src/device-profile.js";
import type { ModelPackageManifest } from "../../src/manifest.js";
import {
  createTextRuntimeController,
  type TextRuntimeController,
} from "./text-runtime.js";
import { createLocalWeightUploadProbe } from "./upload-probe.js";

interface LocalRuntimeConfiguration {
  readonly manifest?: ModelPackageManifest;
  readonly bufferShardPolicy?: BufferShardPolicy;
  readonly uploadDiagnostics?: true;
  readonly uploadLaneBytes?: number;
  readonly retireUploadAfterEachWrite?: boolean;
  readonly residencyPolicy?: "auto" | "resident" | "rolling" | "hybrid";
  readonly residentLayerCount?: number;
}

interface LocalRuntimeWindow extends Window {
  readonly __QWEN35_RUNTIME_CONFIG__?: LocalRuntimeConfiguration;
}

declare global {
  // The injected development agent reads only this explicit command surface.
  // Local model pins remain closed over by the controller and cannot be
  // replaced by an operator command payload.
  var __QWEN_LOCAL_CONTROL__: TextRuntimeController | undefined;
}

const root = document.querySelector<HTMLElement>("#qwen-app");
if (root === null) throw new Error("The polished application root is missing");

const MIB = 1024 * 1024;
const runtimeConfiguration = (window as LocalRuntimeWindow)
  .__QWEN35_RUNTIME_CONFIG__;
const localProbe = (() => {
  if (runtimeConfiguration?.uploadDiagnostics !== true) return null;
  const bufferShardBytes = runtimeConfiguration.bufferShardPolicy === "evidence-128"
    ? 128 * MIB
    : runtimeConfiguration.bufferShardPolicy === "evidence-64"
    ? 64 * MIB
    : null;
  const uploadLaneBytes = runtimeConfiguration.uploadLaneBytes;
  const retireAfterEachWrite = runtimeConfiguration.retireUploadAfterEachWrite;
  if (
    runtimeConfiguration.manifest === undefined ||
    bufferShardBytes === null ||
    !Number.isSafeInteger(uploadLaneBytes) ||
    uploadLaneBytes === undefined ||
    uploadLaneBytes < 4 ||
    uploadLaneBytes % 4 !== 0 ||
    typeof retireAfterEachWrite !== "boolean"
  ) {
    throw new Error("The local upload probe configuration is invalid");
  }
  const navigatorWithGpu = navigator as Navigator & {
    readonly gpu?: WebGpuProbeSurface["gpu"];
  };
  const surface: WebGpuProbeSurface = {
    ...(navigatorWithGpu.gpu === undefined ? {} : { gpu: navigatorWithGpu.gpu }),
    performance: globalThis.performance as unknown as NonNullable<
      WebGpuProbeSurface["performance"]
    >,
  };
  return createLocalWeightUploadProbe({
    surface,
    manifest: runtimeConfiguration.manifest,
    bufferShardBytes,
    uploadLaneBytes,
    retireAfterEachWrite,
    onEvent(event) {
      dispatchEvent(new CustomEvent("qwen-local-runtime-upload-event", {
        detail: event,
      }));
    },
  });
})();

// Development defers loading until the authenticated operator asks for it.
const application = startQwen35ChatApp(root, {
  autoLoad: false,
  ...(localProbe === null
    ? {}
    : {
        webGpuSurface: localProbe.surface,
        uploadLaneBytes: runtimeConfiguration!.uploadLaneBytes!,
        uploadRetirementPolicy:
          runtimeConfiguration!.retireUploadAfterEachWrite === true
            ? "per-write" as const
            : "window" as const,
      }),
});
const controller = createTextRuntimeController({
  coordinator: application.coordinator,
  onRunMetrics(metrics) {
    // The injected local agent applies a second allowlist before transport.
    // This event intentionally contains only numeric command measurements.
    dispatchEvent(new CustomEvent("qwen-local-runtime-metrics", {
      detail: metrics,
    }));
  },
});
globalThis.__QWEN_LOCAL_CONTROL__ = Object.freeze({
  async load(payload?: unknown) {
    try {
      await controller.load(payload);
    } finally {
      // Cancellation can intentionally suppress a failed load event.
      localProbe?.endWeightsUpload();
    }
  },
  runPrompt: (payload?: unknown) => controller.runPrompt(payload),
  async cancelPrompt() {
    localProbe?.endWeightsUpload();
    await controller.cancelPrompt();
  },
  async dispose() {
    localProbe?.endWeightsUpload();
    await controller.dispose();
  },
  getState: () => controller.getState(),
});
application.coordinator.subscribeLoadEvents((event) => {
  if (event.phase === "weights_upload" && event.completedBytes === 0) {
    localProbe?.beginWeightsUpload();
  } else if (
    event.phase === "driver_initialize" ||
    event.phase === "ready" ||
    event.phase === "failed"
  ) {
    localProbe?.endWeightsUpload();
  }
  dispatchEvent(new CustomEvent("qwen-local-runtime-load-event", {
    detail: event,
  }));
});
dispatchEvent(new CustomEvent("qwen-local-runtime-ready"));
