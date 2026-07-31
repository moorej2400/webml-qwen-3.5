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
- Do not impose an assumed model-size or browser-memory ceiling. Test materially
  different allocation, packing, streaming, kernel, and scheduling designs.
- Report a limit only after repeated physical-device evidence across different
  approaches, and describe it as the tested boundary rather than a universal
  browser rule.
- Preserve the selected model, quantization, multimodal support, and context
  target unless the user explicitly changes those product requirements.
