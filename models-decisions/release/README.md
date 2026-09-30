# Released decision artifacts

`squad2-noul-v0.2.idsn` — 82,088 bytes, sha256 `d1b7f3d5…`, digests in `SHA256SUMS`.

A Noul head: binary answerability over Granite 4.1 3B hidden states. Contains the standardiser,
the head and the temperature that were measured together, plus the SHA-256 of the base it was
fitted against. It does not contain the base model.

**Requires base** `granite-4.1-3b-Q4_K_M.gguf` sha256
`662b0626cd58f443baea23559b469df6576a81d349649c59413b36a9fb32eb29`. The artifact checks this before
spending any inference and refuses a file that disagrees — the HuggingFace build of the same name
is the same byte size with different weights.

Sealed split, read once: accuracy 0.8000 against a 0.5033 majority floor, ECE 0.0477, Brier 0.1459.

Run it with `:models-decisions:decide` or `:models-decisions:briefing`; see `../RELEASE.md` for the
commands, the limits, and what this head gets wrong.
