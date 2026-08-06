import { lstat, readFile, realpath } from "node:fs/promises";
import path from "node:path";

import {
  assertQwen35ConvertedPackageTrust,
  assertQwen35PackageIdentity,
  snapshotQwen35Manifest,
  type Qwen35BrowserLoadOptions,
} from "../../src/qwen35-model-loader.js";
import type { BufferShardPolicy } from "../../src/device-profile.js";

export type BrowserRuntimeConfiguration = Readonly<
  Pick<
    Qwen35BrowserLoadOptions,
    | "manifest"
    | "packageBaseUrl"
    | "expectedPackageBaseUrl"
    | "expectedManifestSha256"
    | "compiledTokenizerUrl"
    | "bufferShardPolicy"
    | "uploadLaneBytes"
  >
  & {
    readonly uploadDiagnostics?: true;
    readonly retireUploadAfterEachWrite?: boolean;
  }
>;

export interface BrowserRuntimeEnvironment {
  readonly manifestPath: string;
  readonly packageBaseUrl: string;
  readonly expectedManifestSha256: string;
  readonly compiledTokenizerUrl: string;
  readonly bufferShardPolicy: BufferShardPolicy;
  readonly uploadDiagnostics?: true;
  readonly uploadLaneBytes?: number;
  readonly retireUploadAfterEachWrite?: boolean;
}

const MIB = 1024 * 1024;
// Every candidate changes one baseline variable so a physical-device run can
// attribute a failure boundary to buffer shaping, lane width, or retirement.
const UPLOAD_PROBE_TRIALS = Object.freeze({
  "baseline-128": Object.freeze({
    bufferShardPolicy: "evidence-128",
    uploadLaneBytes: 32 * MIB,
    retireUploadAfterEachWrite: false,
  }),
  "buffer-64": Object.freeze({
    bufferShardPolicy: "evidence-64",
    uploadLaneBytes: 32 * MIB,
    retireUploadAfterEachWrite: false,
  }),
  "lane-16": Object.freeze({
    bufferShardPolicy: "evidence-128",
    uploadLaneBytes: 16 * MIB,
    retireUploadAfterEachWrite: false,
  }),
  "lane-8": Object.freeze({
    bufferShardPolicy: "evidence-128",
    uploadLaneBytes: 8 * MIB,
    retireUploadAfterEachWrite: false,
  }),
  "paced-retirement": Object.freeze({
    bufferShardPolicy: "evidence-128",
    uploadLaneBytes: 32 * MIB,
    retireUploadAfterEachWrite: true,
  }),
} as const satisfies Readonly<Record<string, {
  readonly bufferShardPolicy: BufferShardPolicy;
  readonly uploadLaneBytes: number;
  readonly retireUploadAfterEachWrite: boolean;
}>>);

const parseUploadProbeTrial = (
  environment: NodeJS.ProcessEnv,
): Readonly<{
  readonly bufferShardPolicy: BufferShardPolicy;
  readonly uploadDiagnostics?: true;
  readonly uploadLaneBytes?: number;
  readonly retireUploadAfterEachWrite?: boolean;
}> => {
  const trial = environment.QWEN_RUNTIME_UPLOAD_PROBE_TRIAL;
  if (trial === undefined || trial === "") {
    return Object.freeze({
      bufferShardPolicy: parseBufferShardPolicy(
        environment.QWEN_RUNTIME_BUFFER_SHARD_POLICY,
      ),
    });
  }
  if (
    environment.QWEN_RUNTIME_BUFFER_SHARD_POLICY !== undefined &&
    environment.QWEN_RUNTIME_BUFFER_SHARD_POLICY !== ""
  ) {
    throw new Error("Upload probe trials cannot combine with a buffer shard policy");
  }
  if (!Object.hasOwn(UPLOAD_PROBE_TRIALS, trial)) {
    throw new Error("Runtime upload probe trial is invalid");
  }
  return Object.freeze({
    ...UPLOAD_PROBE_TRIALS[trial as keyof typeof UPLOAD_PROBE_TRIALS],
    uploadDiagnostics: true,
  });
};

const parseBufferShardPolicy = (value: string | undefined): BufferShardPolicy => {
  if (value === undefined || value === "") return "default";
  if (value === "evidence-128" || value === "evidence-64") return value;
  throw new Error("Runtime buffer shard policy is invalid");
};

const requireValue = (
  environment: NodeJS.ProcessEnv,
  key: string,
  label: string,
): string => {
  const value = environment[key];
  if (typeof value !== "string" || value.length === 0) {
    throw new Error(`${label} is required`);
  }
  return value;
};

const validateLocalPath = (value: string): string => {
  const normalized = path.posix.normalize(value.replaceAll("\\", "/"));
  if (!normalized.startsWith(".local/") || normalized.includes("../")) {
    throw new Error("Runtime manifest path must be under .local");
  }
  return normalized;
};

const immutableHuggingFaceUrl = (
  value: string,
  label: string,
  kind: "base" | "file",
): string => {
  let parsed: URL;
  try {
    parsed = new URL(value);
  } catch {
    throw new Error(`${label} must be an immutable URL`);
  }
  if (
    parsed.protocol !== "https:" ||
    parsed.hostname !== "huggingface.co" ||
    parsed.port !== "" ||
    parsed.username !== "" ||
    parsed.password !== "" ||
    parsed.search !== "" ||
    parsed.hash !== ""
  ) {
    throw new Error(`${label} must use credential-free HTTPS`);
  }
  const basePattern = /^\/[^/]+\/[^/]+\/resolve\/[a-f0-9]{40}\/$/;
  const filePattern = /^\/[^/]+\/[^/]+\/resolve\/[a-f0-9]{40}\/.+[^/]$/;
  if (!(kind === "base" ? basePattern : filePattern).test(parsed.pathname)) {
    throw new Error(`${label} must use an immutable Hugging Face revision URL`);
  }
  return parsed.href;
};

export const loadBrowserRuntimeEnvironment = (
  environment: NodeJS.ProcessEnv,
): BrowserRuntimeEnvironment => {
  const expectedManifestSha256 = requireValue(
    environment,
    "QWEN_RUNTIME_MANIFEST_SHA256",
    "Runtime manifest SHA-256",
  );
  if (!/^[a-f0-9]{64}$/.test(expectedManifestSha256)) {
    throw new Error("Runtime manifest SHA-256 must be an exact lowercase digest");
  }
  const uploadProbe = parseUploadProbeTrial(environment);
  return Object.freeze({
    manifestPath: validateLocalPath(
      requireValue(environment, "QWEN_RUNTIME_MANIFEST", "Runtime manifest path"),
    ),
    packageBaseUrl: immutableHuggingFaceUrl(
      requireValue(
        environment,
        "QWEN_RUNTIME_PACKAGE_BASE_URL",
        "Runtime package base URL",
      ),
      "Runtime package base URL",
      "base",
    ),
    expectedManifestSha256,
    compiledTokenizerUrl: immutableHuggingFaceUrl(
      requireValue(
        environment,
        "QWEN_RUNTIME_TOKENIZER_URL",
        "Compiled tokenizer URL",
      ),
      "Compiled tokenizer URL",
      "file",
    ),
    ...uploadProbe,
  });
};

const resolveManifestFile = async (
  projectRoot: string,
  relativePath: string,
): Promise<string> => {
  if (path.isAbsolute(relativePath)) {
    throw new Error("Runtime manifest path must be under .local");
  }
  const localRoot = path.resolve(projectRoot, ".local");
  const requested = path.resolve(projectRoot, relativePath);
  if (!requested.startsWith(`${localRoot}${path.sep}`)) {
    throw new Error("Runtime manifest path must be under .local");
  }
  const [rootInfo, requestedInfo] = await Promise.all([
    lstat(localRoot),
    lstat(requested),
  ]);
  if (
    !rootInfo.isDirectory() ||
    rootInfo.isSymbolicLink() ||
    !requestedInfo.isFile() ||
    requestedInfo.isSymbolicLink()
  ) {
    throw new Error("Runtime manifest must be a regular local file");
  }
  const [realLocalRoot, realRequested] = await Promise.all([
    realpath(localRoot),
    realpath(requested),
  ]);
  if (!realRequested.startsWith(`${realLocalRoot}${path.sep}`)) {
    throw new Error("Runtime manifest must resolve under .local");
  }
  return realRequested;
};

export const loadBrowserRuntimeConfiguration = async (options: {
  readonly projectRoot: string;
  readonly environment: BrowserRuntimeEnvironment;
}): Promise<BrowserRuntimeConfiguration> => {
  const manifestPath = await resolveManifestFile(
    options.projectRoot,
    options.environment.manifestPath,
  );
  const bytes = await readFile(manifestPath);
  if (bytes.length === 0 || bytes.length > 16 * 1024 * 1024) {
    throw new Error("Runtime manifest size is invalid");
  }
  let input: unknown;
  try {
    input = JSON.parse(bytes.toString("utf8"));
  } catch {
    throw new Error("Runtime manifest is not valid JSON");
  }
  const manifest = snapshotQwen35Manifest(
    input as Qwen35BrowserLoadOptions["manifest"],
  );
  assertQwen35PackageIdentity(manifest);
  const trust = {
    packageBaseUrl: options.environment.packageBaseUrl,
    expectedPackageBaseUrl: options.environment.packageBaseUrl,
    expectedManifestSha256: options.environment.expectedManifestSha256,
  };
  assertQwen35ConvertedPackageTrust(manifest, trust);
  return Object.freeze({
    manifest,
    ...trust,
    compiledTokenizerUrl: options.environment.compiledTokenizerUrl,
    bufferShardPolicy: options.environment.bufferShardPolicy,
    ...(options.environment.uploadDiagnostics === true
      ? {
          uploadDiagnostics: true as const,
          uploadLaneBytes: options.environment.uploadLaneBytes!,
          retireUploadAfterEachWrite:
            options.environment.retireUploadAfterEachWrite!,
        }
      : {}),
  });
};
