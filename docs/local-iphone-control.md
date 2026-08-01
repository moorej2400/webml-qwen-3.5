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

## Immutable runtime inputs

Copy the generated language manifest to a regular file under the ignored
`.local/` directory. Do not use a symbolic link. The model shards and compiled
tokenizer must be available from credential-free HTTPS URLs that contain exact
40-character Hugging Face revisions:

```sh
mkdir -p .local/model
cp <generated-manifest-path> .local/model/manifest.json
export QWEN_RUNTIME_MANIFEST=".local/model/manifest.json"
export QWEN_RUNTIME_PACKAGE_BASE_URL="https://huggingface.co/<owner>/<repository>/resolve/<40-character-revision>/"
export QWEN_RUNTIME_MANIFEST_SHA256="<64-character-canonical-manifest-sha256>"
export QWEN_RUNTIME_TOKENIZER_URL="https://huggingface.co/<owner>/<repository>/resolve/<40-character-revision>/<compiled-tokenizer-file>"
```

For a measured allocation experiment, optionally set
`QWEN_RUNTIME_BUFFER_SHARD_POLICY` to `evidence-128` or `evidence-64` before
starting the local server. Omit it for the default 256 MiB per-buffer shape.
This setting changes buffer segmentation only. It does not reduce the model,
quantization, package, or 16K context target.

`QWEN_RUNTIME_MANIFEST_SHA256` is the canonical package-manifest digest used by
the runtime cache. It is not a digest of incidental JSON whitespace. Startup
loads the local manifest through a realpath boundary, validates the pinned Qwen
source identities, and requires the exact canonical digest before serving the
page. Runtime URLs and manifest data are returned from a no-store endpoint.
Credentials and local file paths are never included in that response.

When the package producer does not report the canonical digest, calculate it
from the local manifest with the runtime's own canonicalizer:

```sh
export QWEN_RUNTIME_MANIFEST_SHA256="$(node --import tsx --input-type=module -e 'import { readFileSync } from "node:fs"; import { modelCacheKey } from "./src/opfs-model-cache.ts"; process.stdout.write(modelCacheKey(JSON.parse(readFileSync(process.env.QWEN_RUNTIME_MANIFEST, "utf8"))))')"
```

## Operator credential and automatic phone connection

Generate only the operator token in the current shell. The phone does not use a
code, token entry, persistent device secret, or dialog. The commands below keep
the operator token in process environment only:

```sh
export QWEN_CONTROL_OPERATOR_TOKEN="$(openssl rand -base64 48 | tr -d '\n' | tr '+/' '-_')"
export QWEN_CONTROL_TLS_CERT=".local/tls/cert.pem"
export QWEN_CONTROL_TLS_KEY=".local/tls/key.pem"
export QWEN_CONTROL_PUBLIC_HOST="<development-hostname>"
npm run control:dev
```

Do not put the operator token in `.env`, shell scripts, command arguments,
source files, or logs. Each locally served document automatically requests one
short-lived, single-use WSS ticket from its exact same HTTPS origin. The server
checks both `Host` and `Origin` before it issues the ticket and checks them again
at the WebSocket upgrade. The ticket is only in the immediate browser request
and WebSocket handshake; it is never stored in HTML, JavaScript, local storage,
or session storage. This is a trusted-local-network, browser same-origin and
CSRF boundary, not cryptographic phone authentication: a network client that
can forge the required headers is not cryptographically identified. Ticket
issuance has a short lifetime, a one-use rule, a bounded pending-ticket pool,
and a bounded issue rate to limit accidental reconnect storms.

The service fails closed if TLS material, the operator credential, or the
explicit public host is missing. The operator API always binds to the IPv4 loopback
interface. Only the HTTPS app and same-origin-gated phone WebSocket use the
configured development host. The operator API remains bearer-authenticated and
loopback-only.

## Phone and operator flow

Open `https://<development-hostname>:<public-port>/` in Safari on the iPhone.
It connects to the local protocol automatically. The minimal page has Load, Run
prompt, Cancel, Dispose, and Get state controls. These controls call the same
session handlers used by the authenticated operator commands.

The loopback operator API accepts `load`, `runPrompt`, `cancelPrompt`,
`dispose`, `getState`, `warmReload`, and `coldAppReload`. `load` uses only the
startup configuration and rejects payload overrides. `runPrompt` accepts only
this bounded payload shape:

```json
{
  "prompt": "<text>",
  "maxNewTokens": 128
}
```

The operator API returns command lifecycle records. `getState` can return only
model, generation, cache, device, context, and CPU/GPU byte fields. Prompt and
generated text stay in the phone page and are not returned as telemetry.

## Recovery order

Recovery uses reconnect, state reconciliation, dispose, reload, and command
retry before it asks for an external fallback. A cold app reload reports
`external_fallback_required`; the control service does not claim it can operate
the phone or a remote desktop.

Structured telemetry is written under ignored `.local/runs/`. The journal uses
an allowlist and omits unknown fields and free-form strings; it does not redact
and store prompts, responses, URLs, cookies, forwarded addresses, or stack
traces. For local device correlation only, each record may
include the durable device, tab, and document IDs, a coarse server-parsed OS
family/version, and the direct TLS socket IP. It never uses forwarded headers.
These local fields are ignored by Git and excluded from the public build. A run
uses capped segments and stops accepting new events at its hard cap instead of
overwriting prior evidence. The operator API returns sanitized benchmark
identifiers and numeric metric summaries.
