# Local iPhone Control

This development service is not part of the public application. It injects a
browser agent only when the local HTTPS server serves the page. The public build
starts at `src/` and has an automated scan that rejects control code, endpoints,
credentials, debug flags, and local addresses.

## Trusted HTTPS

Create all TLS files under the ignored `.local/` directory. One option is
`mkcert`:

```sh
mkdir -p .local/tls
mkcert -install
mkcert -cert-file .local/tls/cert.pem -key-file .local/tls/key.pem "<development-hostname>"
chmod 600 .local/tls/cert.pem .local/tls/key.pem
```

Install the local CA certificate on the test iPhone. With `mkcert`, obtain the
CA directory with `mkcert -CAROOT`, transfer only `rootCA.pem` to the phone,
install the downloaded profile, and enable full trust for that certificate in
the iOS certificate trust settings. Never transfer `rootCA-key.pem`.

Use a development hostname that resolves to the Mac on the local network. Do
not put that hostname or an address in this repository.

## Process-only credentials

Generate a different token for the phone and operator in the current shell.
The commands below keep the values in process environment only:

```sh
export QWEN_CONTROL_PHONE_TOKEN="$(openssl rand -base64 48 | tr -d '\n' | tr '+/' '-_')"
export QWEN_CONTROL_OPERATOR_TOKEN="$(openssl rand -base64 48 | tr -d '\n' | tr '+/' '-_')"
export QWEN_CONTROL_TLS_CERT=".local/tls/cert.pem"
export QWEN_CONTROL_TLS_KEY=".local/tls/key.pem"
export QWEN_CONTROL_PUBLIC_HOST="<development-hostname>"
npm run control:dev
```

Do not put tokens in `.env`, shell scripts, command arguments, source files, or
logs. The service fails closed if TLS material, either token, or the explicit
public host is missing. The operator API always binds to the IPv4 loopback
interface. Only the HTTPS app and authenticated phone WebSocket use the
configured development host.

## Recovery order

Recovery uses reconnect, state reconciliation, dispose, reload, and command
retry before it asks for an external fallback. A cold app reload reports
`external_fallback_required`; the control service does not claim it can operate
the phone or a remote desktop.

Structured telemetry is written under ignored `.local/runs/`. The journal uses
an allowlist and does not store prompts, responses, URLs, cookies, headers,
addresses, or stack traces.
