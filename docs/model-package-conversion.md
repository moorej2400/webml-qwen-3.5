# Model package conversion

The repository-owned Node converter accepts only explicit source and output
paths. The source must be the exact language or vision GGUF pinned for the
selected command in `model-sources.json`.
The output parent must already exist, and the output directory must not exist.
The output must be outside a Git worktree so model shards cannot enter source
control by mistake.

Run a read-only inventory first:

```sh
npm run convert:qwen35-language -- --source <gguf-path> --output <package-path> --inventory
```

Inventory authenticates the complete 2,665,441,248-byte source size and SHA-256,
parses the GGUF directory through bounded random-access reads, reports the actual
GGML type counts, and lists only the tensors selected by the versioned MTP
exclusion policy. It does not create the output directory.

Run a read-only conversion plan next:

```sh
npm run convert:qwen35-language -- --source <gguf-path> --output <package-path> --dry-run
```

Dry-run performs the same source authentication and reports the shard count,
segment count, and converted byte count. It does not write shards.

Create the package after the inventory and plan match `model-sources.json`:

```sh
npm run convert:qwen35-language -- --source <gguf-path> --output <package-path>
```

The converter streams through one authenticated open file handle. Reads are at
most 8 MiB, shard segments preserve complete tensor rows, and quantized weights
are never expanded into a full-model floating-point buffer. It writes a sibling
staging directory and publishes the package with one final directory rename.

The output contains:

- `manifest.json`: versioned browser package layout and immutable source pins;
- `shards/model-NNNNN.bin`: aligned WebGPU tensor storage;
- `SHA256SUMS`: hashes for every shard and metadata document;
- `source-provenance.json`: source identity, observed directory facts, actual
  GGML inventory, runtime ABI, and recorded exclusions; and
- `LICENSES.json`: Apache-2.0 model-material license metadata and upstream URL.

The converter does not overwrite output and does not remove failed work. If a
conversion fails after staging starts, the sibling staging directory is
retained for inspection. Review it manually before moving it to trash.

## Vision projector package

The vision converter uses the separately pinned BF16 projector and its pinned
`preprocessor_config.json` identity from `model-sources.json`. It authenticates
the complete source before parsing its GGUF directory. It then creates a
bootstrap shard group plus independently addressable, numerically ordered
vision-layer shard groups. A shard never belongs to more than one group.

Run the same inventory and dry-run sequence before conversion:

```sh
npm run convert:qwen35-vision -- --source <vision-gguf-path> --output <vision-package-path> --inventory
npm run convert:qwen35-vision -- --source <vision-gguf-path> --output <vision-package-path> --dry-run
npm run convert:qwen35-vision -- --source <vision-gguf-path> --output <vision-package-path>
```

The package manifest has `packageKind: "vision"`, the processor identity, the
versioned vision runtime ABI, and the pinned preprocessing settings. Its
checksummed `layer-index.json` maps bootstrap and each vision layer to exact
shard indexes for OPFS streaming. Conversion reads at most 8 MiB at a time and
does not expand the complete projector into a floating-point buffer. Before it
writes a shard, it exclusively creates the requested output directory. It
writes only inside that reserved directory and retains it after a failure for
inspection; it never replaces an output that appears during conversion.

Vision package publication is the next separate operation. There is no vision
staging or upload command yet; do not use the language flat-staging command
below for a vision package.

## Language package publication

The following flat-staging command is language-only. Before language
publication, supply an existing local copy of the exact pinned GGUF, a new
outside-Git output path, the final public model-repository identifier, and
credentials for that hosting service.

For Hugging Face's flat web uploader, authenticate and stage the converted
package with the compiled tokenizer:

```sh
npm run stage:qwen35-huggingface -- \
  --package <converted-package-path> \
  --tokenizer-bin <compiled-tokenizer-bin> \
  --tokenizer-manifest <compiled-tokenizer-manifest> \
  --output <new-upload-directory>
```

The staging tool verifies the converted package checksum file, the exact model
and tokenizer identities, every shard hash, and the compiled tokenizer. It then
creates an exclusively reserved outside-Git directory with flat shard names, a
rewritten validated manifest, publication provenance, and complete checksums.
Shard files use copy-on-write cloning when the filesystem supports it and always
have ownership independent from the converter output. The tool verifies every
staged shard again, never overwrites output, and retains failed output for
inspection.
