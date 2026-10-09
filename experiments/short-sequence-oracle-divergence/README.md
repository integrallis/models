# Where the embedding gate's failures actually live

**Status: finding recorded, follow-up experiment pre-registered and NOT yet run.**

## The finding

Nine of 43 measured embedding artifacts are NOT_REPRODUCED against the llama.cpp oracle. The
disagreement is not spread across the probe set. In **8 of the 9**, the worst probe is the
one-character input `"a"`, and in most of them it is a lone outlier:

| artifact | worst probe | second worst | gap |
| --- | --- | --- | --- |
| all-MiniLM-L6-v2 Q5_K_S | **0.9756330** (`a`) | 0.9996987 | 0.024 |
| all-MiniLM-L6-v2 Q5_K_M | **0.9779482** (`a`) | 0.9998214 | 0.022 |
| nomic-embed-text-v1.5 Q4_K_S | **0.9761418** (`a`) | 0.9916998 | 0.016 |
| nomic-embed-text-v1.5 Q4_K_M | 0.9957449 (`a`) | 0.9976475 | 0.002 |
| nomic-embed-text-v1.5 Q4_0 | 0.9969204 (`a`) | 0.9988028 | 0.002 |
| LFM2.5-Embedding Q5_K_M | 0.9983368 (`a`) | 0.9991401 | 0.001 |
| LFM2.5-Embedding Q6_K | 0.9985676 (`a`) | 0.9992638 | 0.001 |
| LFM2.5-Embedding Q4_K_M | 0.9986829 (`a`) | 0.9992232 | 0.001 |
| Qwen3-Embedding-4B Q4_K_M | 0.9982640 (long probe) | 0.9983135 | 0.00005 |

Full table including every passing artifact in `raw/worst-probe-table.tsv`. For all-MiniLM Q5_K_S,
seven of eight probes clear the floor by a wide margin and a single degenerate input decides the
verdict.

The last row is the exception and is a different shape: Qwen3-Embedding-4B Q4_K_M is uniformly
slightly off across every probe, so whatever is wrong there is not this.

## Two hypotheses this kills

Both were plausible from the quantization labels and both are refuted by the data in `raw/`.

**"A K-quant kernel is wrong."** Q5_K alone reproduces at 0.99999986 on nomic, and Q6_K alone at
0.99999986, while the same kernels on `lfm2` land at 0.9985. A kernel that was wrong would not be
right on one architecture and wrong on another.

**"Q5_1 count drives it."** all-MiniLM Q5_K_S carries 30 Q5_1 tensors and fails at 0.9756;
`granite-embedding-107m-multilingual` Q5_K_S carries **the identical type profile** -- 30 Q5_1 and
6 Q5_K -- and passes at 0.9997886. Same scheme, same counts, opposite verdict. The apparent
dose-response within the all-MiniLM family (4 Q5_1 -> pass, 28 -> fail, 30 -> fail) disappears the
moment a second model is tabulated, which is exactly the trap of reading one family.

What the two models do differ in is pooling: all-MiniLM declares mean, granite declares CLS. With a
single-token input there is almost nothing to average, so a per-token difference that mean pooling
would normally dilute arrives at full strength.

## Pre-registered follow-up

**Hypothesis.** The residual disagreement is dominated by short sequences, where a fixed per-token
difference is not diluted by pooling and where the pooled vector's norm is smallest, so the same
absolute error is a larger angle.

**Prediction, stated before measuring.** For each failing artifact, cosine against the oracle is
monotonically increasing in token count, and the `"a"` probe is the minimum for every artifact whose
pooling is mean. For CLS-pooled artifacts the effect is present but smaller, since CLS reads one
position either way.

**Decision rule.** Measure oracle cosine against token count on a sweep of synthetic inputs of
1, 2, 4, 8, 16, 32, 64, 128 tokens, on one failing artifact per architecture. If cosine at 1-2
tokens is below the floor while cosine at >= 16 tokens is above it on **all** of them, the finding
is confirmed and the fix is scoped to short-sequence handling. If any artifact is below the floor at
>= 16 tokens, this explanation is insufficient and the kernel is back under suspicion.

**Kill criterion.** If the sweep is flat -- no relationship between token count and cosine -- this
hypothesis is dead and gets written up as a null, not reframed.

## What does not move

The floor stays at 0.999 worst-probe and the probe set stays as it is. The gate README already
states why the metric is the worst probe and not the mean: "averaging lets one broken case hide
behind seven good ones." The `"a"` probe is in the set precisely to catch a degenerate input, a
one-character query is a real thing to embed, and finding that it is the case that fails is a
reason to fix it rather than to stop asking. **These nine artifacts stay NOT_REPRODUCED.**

## Reproduce

    # the worst-probe table, from the committed reports
    python3 - <<'PY'
    import json,pathlib
    for p in sorted(pathlib.Path('benchmark-results/embedding').glob('*.json')):
        d=json.loads(p.read_text())
        if 'perProbe' not in d: continue
        pr=sorted(d['perProbe'], key=lambda x: x['cosine'])
        print(f"{p.stem:56s} {pr[0]['cosine']:.7f} {pr[0]['probe'][:20]!r}")
    PY
