# Public Repository Safety

This is a public Git repository. Treat every tracked file, commit, branch, tag,
issue, pull request, build artifact, and log as publicly visible.

## Required Checks Before Committing

- Commit only source files and documentation intended for the public project.
- Review the complete staged diff before every commit.
- Scan staged content for credentials, tokens, private keys, secrets, personal
  information, private hostnames, private IP addresses, absolute local paths,
  device names, account names, and internal URLs.
- Use neutral placeholders such as `<host>`, `<port>`, and `<path>` in public
  documentation.
- Keep company names, company systems, customer information, internal project
  details, and employer-specific material out of this repository.
- Keep personal notes, local debugging output, AI-session files, editor state,
  downloaded models, certificates, and machine-specific configuration
  untracked.
- Confirm that generated files do not embed local paths, usernames, secrets, or
  private environment values before committing them.

## Prohibited Content

Never commit:

- `.env` files or local environment overrides
- API keys, access tokens, passwords, cookies, credentials, or private keys
- certificates or provisioning profiles created for local development
- model weights, caches, browser profiles, logs, database files, or debug dumps
- personal names, email addresses, account identifiers, device names, or home
  directory paths
- employer, customer, or other company-specific information
- proprietary code, documents, screenshots, URLs, infrastructure details, or
  data from another project

If public suitability is uncertain, leave the file untracked until it has been
reviewed and sanitized.

## Experimental Feasibility

- Treat an early device failure as evidence about the tested implementation,
  not proof that the browser target is impossible.
- Do not use a rule of thumb such as a 1 GB browser-model ceiling as a product
  constraint. Assume the selected model can work and can perform very well;
  use experiments to find the required design.
- Do not turn conventional browser guidance, reported limits, adapter limits,
  or a failed allocation into a stop condition. Change the design and measure
  the result on the physical target.
- Do not impose an assumed model-size or browser-memory ceiling. Test materially
  different allocation, packing, streaming, kernel, and scheduling designs.
- Report a limit only after repeated physical-device evidence across different
  approaches, and describe it as the tested boundary rather than a universal
  browser rule.
- Preserve the selected model, quantization, multimodal support, and context
  target unless the user explicitly changes those product requirements.

## Lessons learned — local iPhone validation

- Use Computer Use with the direct iPhone Mirroring app. Do not substitute
  desktop screenshot automation for the physical Safari test.
- Keep the local development server in a persistent session. Before opening
  Safari, verify its listener and make an HTTP request to the Mac LAN address;
  a loopback-only check is not enough.
- Do not report an iPhone result until Safari or the local control telemetry
  confirms the actual page outcome. Recheck the listener before changing
  firewall or phone-network settings.
- Select the target by OS, durable device ID, tab ID, and current document ID.
  Never use the first connected device because stale or desktop tabs can appear
  before the physical phone in the control snapshot.
- The local control server serves `dev-dist`, not TypeScript source files. Run
  `npm run build:dev` before every reload used for a benchmark; otherwise the
  browser can execute an older runtime while the source tree looks current.
