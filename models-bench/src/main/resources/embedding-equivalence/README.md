# Embedding equivalence gate

Compares vectors this runtime produces against ones a pinned reference build produced from the
same model bytes and the same probes.

Retrieval quality is a published property of the weights. What a runtime can get wrong is pooling,
rotary embeddings, dequantization, and normalization. Passing means the vectors are the model's,
not that they are good.

## Running it

```bash
./gradlew :models-bench:run --args="embedding-equivalence \
  --model ~/.modeljars/cache/sha256/06/06507c7b.../model.gguf \
  --report benchmark-results/embedding/qwen3-embedding-0.6b-q8_0.json"
```

Exits `0` when every probe cleared the floors, `1` when any did not, `2` on a usage or integrity
problem. Takes seconds: only this side runs.

## Why the reference is committed

The reference vectors cannot drift unless the oracle build or the probe set changes, and both are
pinned by digest. Recomputing them each run would require a built llama.cpp on every machine that
runs the gate, to re-derive a constant. The CLI refuses to run if either digest has moved.

## The probe set

Eight probes covering the paths a runtime gets wrong:

| Probe | What it exercises |
| --- | --- |
| `The cat sat on the mat.` | Ordinary short prose, the baseline case |
| `How do I reset my password?` | Question form, the dominant retrieval-query shape |
| `Quantum chromodynamics describes...` | Rare tokens, far from the common subword paths |
| `a` | Single token: pooling and position handling with no context to hide in |
| `The quick brown fox... at dawn...` | Long input, where accumulated error and RoPE drift show |
| `Database backups are retained for 30 days.` | Digits and a factual statement |
| `def add(a, b): return a + b` | Code, punctuation-dense tokenization |
| `¿Dónde está la biblioteca?` | Non-Latin script, multi-byte UTF-8, inverted punctuation |

## Regenerating the reference

Required when the probe set changes or the oracle build is bumped. Both are recorded in the
reference file, and the CLI verifies both before comparing.

```bash
cd /path/to/llama.cpp
git rev-parse --short HEAD          # goes into oracleVersion

./build/bin/llama-embedding \
  -m /path/to/model.gguf \
  -f models-bench/src/main/resources/embedding-equivalence/probes.txt \
  --pooling last --embd-output-format array --embd-normalize 2 -c 512 -ngl 0
```

`--pooling last` matches how Qwen3-Embedding reduces states to a vector, `--embd-normalize 2` is
L2, and `-ngl 0` keeps it on CPU so the reference does not vary with the GPU. Changing any of them
changes what the vectors mean, so they must match the recorded `pooling` and `normalized`.

Write the output into `reference-<model>.json` with `oracleVersion`, `probeSetSha256`
(`sha256sum probes.txt`), and `artifactSha256` updated. Regenerate against the *unmodified*
runtime — a reference generated to accommodate a change proves nothing.

## The floors

**Cosine >= 0.999**, mirroring `ModelEmbeddingQualification.MINIMUM_ORACLE_COSINE` in ModelJars.
Duplicated because ModelJars depends on this project, not the reverse; every report writes the
value it used so the two cannot diverge silently.

**Unit length within 1e-3**, checked separately.

Both were placed against measurements. Agreement is not bit-exact — two independent
implementations accumulate floating point differently — so what matters is the distance between
the good band and the broken band:

| Run | Min cosine | Max ‖v‖ − 1 |
| --- | --- | --- |
| Correct (this runtime vs the reference) | 0.99950 | 2.7e-09 |
| Mean pooling instead of last-token | 0.66156 | — |
| L2 normalization skipped | **1.00000** | ~11 |
| *floor* | *0.999* | *1e-3* |

Wrong pooling scores 0.66 against a 0.9995 good band, so the agreement floor has wide margin
either side.

Cosine is scale-invariant, so an unnormalized runtime agrees with a normalized reference at
*exactly* 1.0 — measured via `llama-embedding --embd-normalize -1`. Callers that use a bare dot
product as a cosine shortcut, as `vectors` does, would be wrong while the gate reported perfection.
Hence the length check.

The gate takes the worst probe: averaging lets one broken case hide behind seven good ones.
