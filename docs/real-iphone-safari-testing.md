# Real iPhone Safari Testing

Use this workflow to verify WebGPU/WebML behavior on a physical iPhone instead of relying on desktop Safari or simulator behavior.

## One-Time Device Setup

1. Connect the iPhone to the Mac with a USB cable.
2. Unlock the iPhone and trust the Mac when prompted.
3. On the iPhone, enable `Settings > Apps > Safari > Advanced > Web Inspector`.
4. On the Mac, enable `Safari > Settings > Advanced > Show features for web developers`.
5. In Mac Safari, open `Develop > Inspect Apps and Devices`.
6. Select the iPhone. A working setup shows Safari pages from the iPhone, not `Device is not paired`.

When pairing succeeds, Mac Safari can attach to the iPhone and open a Web Inspector window for an iPhone Safari page.

## Running The Site On The LAN

Start the web app so it listens on the LAN, not only on localhost.

```sh
npm run dev -- --host 0.0.0.0
```

Open the site on the iPhone using the Mac LAN address, for example:

```text
https://<mac-lan-ip>:<port>
```

WebGPU requires a secure context. Prefer trusted local HTTPS with a development certificate. Plain `http://<mac-lan-ip>:<port>` is useful for basic layout checks, but it is not enough for WebGPU validation.

### Trusting The Local WebML Server

The Gemma WebML server uses a private development CA because iPhone Safari will not accept an untrusted self-signed server certificate for WebGPU testing.

1. Start the server on HTTPS port `8443` and HTTP bootstrap port `8080`.
2. On the iPhone, open:

   ```text
   http://<mac-lan-ip>:8080/certs/ios-webml-ca.mobileconfig
   ```

3. Allow the configuration profile download.
4. Open `Settings > General > VPN & Device Management`.
5. Select `iOS WebML Development CA` and install it.
6. Open `Settings > General > About > Certificate Trust Settings`.
7. Enable full trust for `iOS WebML Development CA`.
8. Return to Safari and open:

   ```text
   https://<mac-lan-ip>:8443/
   ```

The profile contains only the public CA certificate. Its private key stays on the Mac in the local WebML server directory.

## Inspecting The iPhone Page

1. Keep the iPhone unlocked and Safari in the foreground.
2. Open the LAN site in iPhone Safari.
3. On the Mac, open `Safari > Develop > Inspect Apps and Devices`.
4. Select the iPhone in the sidebar.
5. Open the page listed under Safari.
6. Use the Web Inspector tabs:
   - `Console` for runtime errors and WebGPU feature probes.
   - `Network` for model, tokenizer, WASM, and shader asset loading.
   - `Storage` for cache and IndexedDB state.
   - `Timelines` for expensive startup or inference work.

## WebGPU Smoke Checks

Run these in the iPhone page's Web Inspector console:

```js
isSecureContext
"gpu" in navigator
await navigator.gpu.requestAdapter()
```

Expected result:

```text
true
true
GPUAdapter {...}
```

If `isSecureContext` is false, fix HTTPS before debugging model code. If `navigator.gpu` is missing or `requestAdapter()` returns null, check iOS/Safari version, WebGPU feature flags, and whether the page is actually loaded in Safari on the physical iPhone.

## Troubleshooting

- `Device is not paired`: open the iPhone in Finder, click `Trust`, then tap `Trust` on the iPhone and enter the passcode.
- `No inspectable contents`: unlock the iPhone and keep Safari in the foreground with a real webpage open.
- iPhone appears in iPhone Mirroring but not Safari inspection: iPhone Mirroring alone is not enough; USB trust/pairing is still required.
- LAN page does not load: confirm the Mac and iPhone are on the same network, the dev server is bound to `0.0.0.0`, and the macOS firewall allows the port.
- `Safari can't open the page because the network connection was lost` on port `8443`: install and fully trust the local CA profile above. Apple reports TLS policy failures as connection failures before the page can run.
- WebGPU is unavailable: verify HTTPS first, then verify Safari/iOS support and any relevant feature flags.
