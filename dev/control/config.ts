import path from "node:path";

import { assertHighEntropyCredential } from "./security.js";

export const RECOVERY_ORDER = [
  "reconnect",
  "reconcile",
  "dispose",
  "reload",
  "retry",
  "external_fallback",
] as const;

export interface ControlEnvironment {
  certPath: string;
  keyPath: string;
  operatorToken: string;
  publicHost: string;
  publicPort: number;
  operatorPort: number;
}

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

const validateLocalPath = (value: string, label: string): string => {
  const normalized = path.posix.normalize(value.replaceAll("\\", "/"));
  if (!normalized.startsWith(".local/") || normalized.includes("../")) {
    throw new Error(`${label} must be under .local`);
  }
  return normalized;
};

const parsePort = (value: string | undefined, fallback: number, label: string): number => {
  const port = value === undefined ? fallback : Number(value);
  if (!Number.isSafeInteger(port) || port < 1 || port > 65_535) {
    throw new Error(`${label} must be a valid TCP port`);
  }
  return port;
};

export const loadControlEnvironment = (
  environment: NodeJS.ProcessEnv,
): ControlEnvironment => {
  const certPath = validateLocalPath(
    requireValue(environment, "QWEN_CONTROL_TLS_CERT", "TLS certificate path"),
    "TLS certificate path",
  );
  const keyPath = validateLocalPath(
    requireValue(environment, "QWEN_CONTROL_TLS_KEY", "TLS key path"),
    "TLS key path",
  );
  const operatorToken = assertHighEntropyCredential(
    environment.QWEN_CONTROL_OPERATOR_TOKEN,
    "operator credential",
  );
  const publicHost = requireValue(
    environment,
    "QWEN_CONTROL_PUBLIC_HOST",
    "public host",
  );
  if (!/^[A-Za-z0-9.:[\]_-]{1,255}$/.test(publicHost)) {
    throw new Error("public host is invalid");
  }
  return {
    certPath,
    keyPath,
    operatorToken,
    publicHost,
    publicPort: parsePort(environment.QWEN_CONTROL_PUBLIC_PORT, 8443, "public port"),
    operatorPort: parsePort(environment.QWEN_CONTROL_OPERATOR_PORT, 8899, "operator port"),
  };
};
