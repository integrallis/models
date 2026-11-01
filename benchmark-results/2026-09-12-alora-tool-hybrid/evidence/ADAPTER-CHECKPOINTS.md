# V9 and V11 adapter checkpoints — rejected experiments, weights kept out of Git

Two untracked `training/adapter/` directories, 148 MB each, from the Qwen3 1.7B activated-LoRA tool
experiments: `qwen3-17b-r32-full-v9` and `qwen3-17b-short-mask-v11`. Both arms were **rejected**;
their negative result is recorded in `../README.md`.

**Verified on 2026-09-26:** each `adapter_model.safetensors` digest matches a digest already pinned in
that arm's committed training manifest.

| arm | safetensors sha256 | pinned in a committed manifest |
|---|---|---|
| `qwen3-17b-r32-full-v9` | `f1a4a1bc15e3942e…ce851c21` | yes — `training/training-manifest.json` |
| `qwen3-17b-short-mask-v11` | `f3238e4f9b997c64…` | yes — `training/manifest.json` |

So each checkpoint's identity is checkable from Git without the weights being in Git.

**What is committed** from these directories: nothing but this note. The small companions
(`adapter_config.json`, `chat_template.jinja`, `added_tokens.json`, tokenizer files, the generated
`README.md`) are byte-identical between the two arms except for the adapter config, and the arm's real
configuration — rank, masking strategy, data, and the decision that rejected it — is already in the
committed training manifests, which are the authority. Duplicating a tokenizer per rejected arm adds
no recoverable information.

**What is gitignored:** the two 139 MB safetensors files and their tokenizer/config companions.

**The part that is a real risk, stated plainly:** unlike the shipped Granite adapter, these weights
are *not* published anywhere. They exist only on this workstation. A rejected experiment's weights
are usually not worth archiving — the manifest, the data recipe and the recorded rejection are what
make the negative result reusable — but that is a judgement, not a certainty, and deleting them is
irreversible. They are being neither committed nor deleted here. If the V-series is ever revisited,
decide deliberately whether these two checkpoints are worth a durable archive before anything cleans
this tree.
