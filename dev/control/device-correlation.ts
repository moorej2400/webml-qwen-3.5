import { isIP } from "node:net";

export type CoarseOsFamily = "ios" | "macos" | "android" | "windows" | "linux" | "other";

export interface SocketDeviceMetadata {
  osFamily: CoarseOsFamily;
  osVersion?: string;
  remoteIp: string;
}

const versionFrom = (value: string, pattern: RegExp): string | undefined => {
  const match = pattern.exec(value)?.[1];
  if (match === undefined) return undefined;
  const version = match.replaceAll("_", ".");
  return /^\d{1,2}(?:\.\d{1,2}){0,2}$/.test(version) ? version : undefined;
};

const coarseOs = (userAgent: string): Pick<SocketDeviceMetadata, "osFamily" | "osVersion"> => {
  if (/\b(?:iPhone|iPad|iPod)\b/i.test(userAgent)) {
    const osVersion = versionFrom(userAgent, /(?:CPU (?:iPhone )?OS|iPhone OS) ([0-9_]+)/i);
    return { osFamily: "ios", ...(osVersion === undefined ? {} : { osVersion }) };
  }
  if (/\bAndroid\b/i.test(userAgent)) {
    const osVersion = versionFrom(userAgent, /Android ([0-9.]+)/i);
    return { osFamily: "android", ...(osVersion === undefined ? {} : { osVersion }) };
  }
  if (/\bWindows NT\b/i.test(userAgent)) {
    const osVersion = versionFrom(userAgent, /Windows NT ([0-9.]+)/i);
    return { osFamily: "windows", ...(osVersion === undefined ? {} : { osVersion }) };
  }
  if (/\bMac OS X\b/i.test(userAgent)) {
    const osVersion = versionFrom(userAgent, /Mac OS X ([0-9_]+)/i);
    return { osFamily: "macos", ...(osVersion === undefined ? {} : { osVersion }) };
  }
  if (/\bLinux\b/i.test(userAgent)) return { osFamily: "linux" };
  return { osFamily: "other" };
};

/**
 * Derives local-only correlation data from the direct TLS socket and handshake
 * header. Forwarded headers are intentionally ignored because proxies can forge them.
 */
export const deriveSocketDeviceMetadata = (input: {
  remoteAddress?: string;
  userAgent?: string | string[];
}): SocketDeviceMetadata => {
  const rawAddress = input.remoteAddress ?? "";
  const ipv4Mapped = rawAddress.startsWith("::ffff:") ? rawAddress.slice("::ffff:".length) : rawAddress;
  const remoteIp = isIP(ipv4Mapped) === 0 ? "unknown" : ipv4Mapped;
  const userAgent = typeof input.userAgent === "string" ? input.userAgent.slice(0, 512) : "";
  const os = coarseOs(userAgent);
  return {
    osFamily: os.osFamily,
    ...(os.osVersion === undefined ? {} : { osVersion: os.osVersion }),
    remoteIp,
  };
};
