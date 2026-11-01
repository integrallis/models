# release-pilot2/bundle — verified locally, deliberately not committed

The `bundle/` directory holds the trained answerability adapter that the published component and
composition are built from: `adapter_model.safetensors` (119 MB), its `models-activated-lora.json`
descriptor, `LICENSE`, `NOTICE` and `SHA256SUMS`.

**Verified on 2026-09-26, on this workstation:**

```
shasum -a 256 bundle/adapter_model.safetensors
67533dff14cd0cfa9ae0d00e83a9955226426ff1f98f1fd28790b2fd8757eea6   # matches
(cd bundle && shasum -a 256 -c SHA256SUMS)                          # LICENSE, NOTICE, descriptor OK
```

That digest is not a new fact recorded here — it is already pinned by five committed files in this
tree, which is why the binary itself does not need to be in Git:

- `../REPRODUCE.md`
- `../pilot2/training-manifest.json`
- `component-report/component-qualification.json`
- `component-report-rust/component-qualification.json`
- `composition-report/composition-clean-host-run.json`

The descriptor additionally pins the base it activates over — `ibm-granite/granite-4.1-3b` at
revision `ab4701481089b58a082ef63cc1cee738887293ff`, artifact SHA-256
`662b0626…fb32eb29` — plus rank 16, alpha 32, the seven target modules, and the invocation-token
sequence. Weight identity and base identity are therefore both checkable from committed files alone.

**Why it is gitignored:** the adapter is a published artifact distributed through
`ModelJars/activated-adapters`, so a 119 MB copy in this research repository would be redundant
redistribution, not evidence. Anything that must be reproduced from it is reproducible from the
pinned digest.

**What that does not do:** this is a byte-identity check against a digest already in the tree. It is
not a fresh download from the distribution point, so it does not by itself prove the published copy
still matches. Re-verify against `activated-adapters` before any re-release that depends on it.

The sibling `known_hosts_*` files are SSH host keys from the pilot hosts — machine configuration, not
evidence — and are ignored for the same reason they should never have been staged.
