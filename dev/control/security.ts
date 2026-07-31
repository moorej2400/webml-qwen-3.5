const CREDENTIAL_PATTERN = /^[A-Za-z0-9_-]{43,256}$/;

export const assertHighEntropyCredential = (value: unknown, field: string): string => {
  if (typeof value !== "string" || !CREDENTIAL_PATTERN.test(value)) {
    throw new TypeError(`${field} must be a high-entropy base64url credential`);
  }
  return value;
};
