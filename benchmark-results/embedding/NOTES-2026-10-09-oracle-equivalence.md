# Embedding oracle equivalence on 0.3.54: two reproduce, one does not

The route to 100 qualified entries runs through the embedding track, not the RAG one. The catalogue
holds **60 embedding artifacts across 8 model families with 5 qualified**, and the embedding gate is
cheap: eight pinned probes against a committed llama.cpp reference, seconds per artifact on one CPU,
no fleet and no comparator arm.

Oracle build for all three runs below: llama.cpp `a5822222909b785f23ddc74ce3c8f85bd0e38562`, the same
commit the existing references were generated against. Floors from
`models-bench/src/main/resources/embedding-equivalence/README.md`: **worst-probe cosine >= 0.999**
and unit length within 1e-3, the worst probe rather than the mean so one broken case cannot hide
behind seven good ones.

## Results

| artifact | pooling | min cosine | ‖v‖−1 | verdict |
|---|---|---|---|---|
| `bge-small-en-v1.5-f16` | cls | **0.9999988** | 4.71e-09 | REPRODUCED |
| `nomic-embed-text-v1.5-f16` | mean | **0.9999999** | 4.47e-09 | REPRODUCED |
| `nomic-embed-text-v1.5-Q4_K_M` | mean | **0.9957449** | 3.75e-09 | **NOT REPRODUCED** |

The first two had committed references and no qualification entry, so they were measurable
immediately — a gap worth noticing on its own: a reference exists precisely because someone intended
to qualify the artifact.

## The Q4_K_M non-reproduction is the interesting one

`nomic-embed-text-v1.5` reproduces at **0.9999999 in f16** and **0.9957449 in Q4_K_M**: same weights,
same pooling, same probe set, same oracle build, same runtime. Unit length holds at 3.75e-09, so this
is not a normalization fault, and 0.9957 is nowhere near the 0.66 that wrong pooling scores — it sits
between the good band and the broken band.

**That points at the Q4_K dequantization path, and it is unmeasured whether it is ours or
llama.cpp's.** Both sides read the same bytes; either could be the one that differs from the true
weights. Nothing here establishes which, and the reference was generated against the unmodified
runtime, so it is not a reference bent to accommodate anything.

What this does *not* license: qualifying the artifact, or calling our dequantization wrong. The
honest statement is that the two implementations disagree beyond the floor on this artifact and the
cause is open.

**That test was run, and it answers the question.**
`granite-embedding-107m-multilingual-Q4_K_M` — a different model, also Q4_K_M, already qualified —
**REPRODUCED on 0.3.54 at min cosine 0.9997590**, unit length 6.41e-09.

So Q4_K dequantization is not generally broken, and 0.3.54 did not break it. The divergence is
**specific to `nomic-embed-text-v1.5`'s Q4_K_M tensors**. Note the ladder it sits on:

| artifact | quant | min cosine |
|---|---|---|
| nomic-embed-text-v1.5 | f16 | 0.9999999 |
| bge-small-en-v1.5 | f16 | 0.9999988 |
| granite-embedding-107m-multilingual | Q4_K_M | 0.9997590 |
| nomic-embed-text-v1.5 | **Q4_K_M** | **0.9957449** |

Q4_K costs roughly two orders of magnitude of agreement against f16 and still clears the floor
comfortably; this one artifact is a further order of magnitude worse and does not. That is a
per-artifact problem — a tensor layout, a quantization choice made by whoever produced this GGUF, or
a type in it our kernel handles differently — and not a property of the quantization family.

**Next:** inspect this artifact's tensor types against granite's. If it carries a type the other does
not, that is the suspect, and the comparison is a header read rather than another benchmark.

## Reproduce

```
./gradlew :models-bench:run --args="embedding-equivalence \
  --model <artifact.gguf> --report benchmark-results/embedding/<name>.json"
```

A new artifact needs a reference first, generated exactly as the gate README specifies — for a BERT
encoder, `--pooling mean --attention non-causal --embd-output-format array --embd-normalize 2
--device none -c 512` — then registered in `references.txt`. The CLI refuses to run if the probe-set
digest or the oracle build has moved, which is why the probe digest is carried in every reference.
