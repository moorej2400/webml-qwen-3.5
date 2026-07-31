export const MTP_EXCLUSION_REASON = "excluded-by-mtp-name-policy-v1";

/**
 * Policy v1 matches only complete MTP/nextn name segments; substring matches
 * would incorrectly exclude ordinary names such as `attempt.weight`.
 */
export function isMtpTensorName(name: string): boolean {
  return /(?:^|\.)(?:mtp|nextn)(?:\.|$)/i.test(name);
}
