import assert from "node:assert/strict";
import { randomBytes } from "node:crypto";
import test from "node:test";

import { loadControlEnvironment, RECOVERY_ORDER } from "../dev/control/config.js";

const validEnvironment = {
  QWEN_CONTROL_TLS_CERT: ".local/tls/cert.pem",
  QWEN_CONTROL_TLS_KEY: ".local/tls/key.pem",
  QWEN_CONTROL_PAIRING_CODE: randomBytes(32).toString("base64url"),
  QWEN_CONTROL_OPERATOR_TOKEN: randomBytes(32).toString("base64url"),
  QWEN_CONTROL_PUBLIC_HOST: "development-host.invalid",
};

test("control environment fails closed without TLS, auth, or an explicit public host", () => {
  assert.throws(() => loadControlEnvironment({}), /TLS certificate/i);
  assert.throws(
    () => loadControlEnvironment({ ...validEnvironment, QWEN_CONTROL_PAIRING_CODE: undefined }),
    /pairing code/i,
  );
  assert.throws(
    () => loadControlEnvironment({ ...validEnvironment, QWEN_CONTROL_PUBLIC_HOST: undefined }),
    /public host/i,
  );
});

test("control environment keeps TLS paths under .local and tokens distinct", () => {
  assert.throws(
    () =>
      loadControlEnvironment({
        ...validEnvironment,
        QWEN_CONTROL_TLS_CERT: "cert.pem",
      }),
    /\.local/,
  );
  assert.throws(
    () =>
      loadControlEnvironment({
        ...validEnvironment,
        QWEN_CONTROL_OPERATOR_TOKEN: validEnvironment.QWEN_CONTROL_PAIRING_CODE,
      }),
    /distinct/i,
  );

  const config = loadControlEnvironment(validEnvironment);
  assert.equal(config.certPath, ".local/tls/cert.pem");
  assert.equal(config.publicHost, "development-host.invalid");
});

test("recovery policy exhausts protocol recovery before external fallback", () => {
  assert.deepEqual(RECOVERY_ORDER, [
    "reconnect",
    "reconcile",
    "dispose",
    "reload",
    "retry",
    "external_fallback",
  ]);
});
