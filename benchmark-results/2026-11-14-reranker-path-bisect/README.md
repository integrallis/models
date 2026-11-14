# Where the pure-Java reranker path lost 27–35% between v0.5.20 and `main` (#457)

**Measured here.** Nothing in this directory is projected. Every number below is read from
`runs/` by `derive.py`; the table is its output pasted verbatim.

    python3 derive.py runs

## The question

`benchmark-results/2026-11-14-reranker-predequantize-ablation/runs-regression/` measured released
`v0.5.20` against `main` + #456 with #360 off — the arithmetic 0.5.21 ships — five interleaved
rounds: `score()` 62.6 → 79.6 ms (1.27x), `scoreAll` 449 → 605 ms (1.35x), cold load 157 → 227 ms
(1.45x), numerics identical to nine decimal places. Speed, not arithmetic. This run asks **which
commit in `v0.5.20..main`** did it.

## Host, artifact, protocol

| | |
| --- | --- |
| instance | `m6a.4xlarge`, AMD EPYC 7R13 (Zen 3, AVX2, **no AVX-512**), 8 cores / 16 threads |
| JDK | Temurin 25.0.5 |
| artifact | `ms-marco-MiniLM-L-6-v2-q4_k-imatrix-g7c-f7.gguf`, sha256 `0752a92bc33289f3fe230d1a33cce471f1a7fe1c8d5a491ca7fff6b068b0fc83` — registered fixture `ms_marco_minilm_l6_v2_q4_k_imatrix_g7c_f7` |
| experiment | `MsMarcoMiniLmRerankerPerformanceExperiment`, `warmups=5 pairIterations=60 batchIterations=20`, six documents |
| route | `GgufRerankingModel` — no batched matrix kernel; the pure-Java path |
| schedule | six arms **interleaved V, T, F, W, X, A per round, five rounds**, each run one forked JVM under `/usr/bin/time -v` |

## Arms

The commits in `v0.5.20..main` that touch this path, in order. Two things are held constant so
each arm runs the arithmetic that ships: **#360 is reverted** wherever it is present (its
pre-dequantization is opt-in-off in 0.5.21 and would otherwise confound every later arm by 1.33x),
and **#456 is applied** wherever `scoreAll` would otherwise throw on this probe (#440 onward). The
`sha-*.txt` files record each arm's exact resulting commit; the construction log is `bisect.log`.

| arm | commit | what it adds over the previous arm | #360 | #456 |
| --- | --- | --- | --- | --- |
| `V-0520` | `83641c18` (tag `v0.5.20`) | released baseline | absent | not needed |
| `T-tokenizer` | `63b46a83` | #435/#437: tokenizer hot-path allocation reduction (`List<Integer>` → `IntArrayBuilder`) | absent | not needed |
| `F-436` | `4819990c` | #360 (reverted here) and #436: tokenizer `EI_EXPOSE_REP` + "BERT F32 projection path" | reverted | not needed |
| `W-440` | `3b37bc3c` | #440 batched BERT pair scoring, #438 compact F16 KV, the `BertForwardPass` refactor around them | reverted | applied |
| `X-vectors` | `5c72d535` | vectors 0.1.34 → 0.1.35 (`VectorUtil`, Panama kernels come from `vectors-core`) | reverted | applied |
| `A-off` | `cdbc9edc` (`main`) | #447–#449 (`readInto`, reranker `scoreBatch` de-synchronised) and the release prep | off by flag | applied |

Not an arm: `dfd3964f` (`models-tinystruct`, a separate module) and the MFCQI badge commit.

A step in a metric between two adjacent arms is attributable to the commits between them and
nothing else; a flat line across an arm means those commits did not move that metric on this host.

## Result

```
| metric | V-0520 | T-tokenizer | F-436 | W-440 | X-vectors | A-off |
| --- | --- | --- | --- | --- | --- | --- |
| coldLoadMillis | 157.14 [156.39..158.75] | 225.13 [222.90..232.67] (1.433x) | 221.54 [220.12..244.98] (1.410x) | 226.40 [222.16..237.68] (1.441x) | 223.75 [221.06..236.19] (1.424x) | 232.43 [222.97..241.39] (1.479x) |
| pairP50Millis | 62.74 [62.19..63.02] | 62.40 [61.78..62.95] (0.995x) | 62.00 [61.81..63.00] (0.988x) | 79.72 [76.37..80.86] (1.271x) | 78.84 [76.41..81.07] (1.257x) | 79.57 [77.53..80.83] (1.268x) |
| pairP95Millis | 100.59 [99.77..100.97] | 99.25 [99.14..100.96] (0.987x) | 99.40 [99.06..101.09] (0.988x) | 100.32 [99.99..101.97] (0.997x) | 101.25 [99.85..102.08] (1.007x) | 100.11 [99.86..101.66] (0.995x) |
| batchP50Millis | 449.77 [446.64..452.68] | 451.16 [444.85..453.13] (1.003x) | 450.20 [443.43..451.15] (1.001x) | 592.56 [591.19..601.91] (1.317x) | 601.73 [590.18..606.82] (1.338x) | 597.56 [593.62..604.94] (1.329x) |
| batchP95Millis | 647.30 [642.26..648.58] | 640.97 [639.90..647.91] (0.990x) | 642.35 [640.11..650.50] (0.992x) | 594.30 [591.86..603.02] (0.918x) | 603.11 [591.85..608.08] (0.932x) | 598.86 [595.61..606.66] (0.925x) |
| batchP50DocumentsPerSecond | 13.34 [13.25..13.43] | 13.30 [13.24..13.49] (0.997x) | 13.33 [13.30..13.53] (0.999x) | 10.13 [9.97..10.15] (0.759x) | 9.97 [9.89..10.17] (0.747x) | 10.04 [9.92..10.11] (0.753x) |
| maxRssMiB | 593.00 [580.92..608.19] | 620.90 [605.38..727.87] (1.047x) | 613.48 [586.31..697.21] (1.035x) | 705.81 [593.21..716.01] (1.190x) | 698.87 [576.62..736.26] (1.179x) | 683.18 [608.43..701.53] (1.152x) |
```

Ratios in parentheses are median(arm) / median(`V-0520`).

```
scoreChecksum per arm (distinct, 9dp): {'V-0520': [-161.44592762], 'T-tokenizer': [-161.44592762], 'F-436': [-161.44592762], 'W-440': [-161.44592762], 'X-vectors': [-161.44592762], 'A-off': [-161.44592762]}
every arm reproduces released v0.5.20 scores to 9dp: True
```

## Verdict: two commits, cleanly separated

**`score()` and `scoreAll` regress at `3b37bc3c` (#440) and nowhere else.** pair p50 runs
62.7 → 62.4 → 62.0 ms through the tokenizer commits, then **79.7 ms** at `W-440`, then 78.8 and 79.6
through the vectors bump and `main`. batch p50: 450 → 451 → 450 → **593** → 602 → 598. The step is
1.27x and 1.32x, the spreads on either side of it do not overlap, and everything after it is flat.
Two things follow:

- The regression is in the sequential path too. `score()` does not use batching, so #440's
  `BertForwardPass` refactor slowed the ordinary encode, not only the new batched one.
- The vectors 0.1.34 → 0.1.35 bump (`X-vectors`) is **not** the cause: 1.257x against 1.271x at
  `W-440`, inside the noise. #447–#449 and the release prep (`A-off`) are flat too.

**Cold load regresses at `63b46a83` (#435/#437) and nowhere else.** 157 → **225 ms** (1.43x) at the
tokenizer "hot-path allocation reduction", then flat within noise through every later arm
(222–232 ms). An allocation-reduction change made every model load 68 ms slower; whatever it builds
up front — the `IntArrayBuilder` path or the structures around it — is paid at construction.

**No arm changes the arithmetic.** All six reproduce released v0.5.20's `scoreChecksum` to nine
decimal places.

`maxRssMiB` rises ~19% at `W-440` and stays there; the spreads are wide and overlap, so it is
suggestive of larger scratch buffers in the batched forward pass and not a measured figure.

## What this does not establish

- *Which lines* inside `3b37bc3c` or `63b46a83` are responsible. Each is one commit here; the
  next step is a diff read of `BertForwardPass` in `3b37bc3c` (the sequential `encodeCls` path
  changed, since `score()` slowed) and of `GgufTokenizer` construction in `63b46a83`.
- Any other model or host. See the ablation directory's caveats; they apply unchanged.

## Decisive run: the `score()` step was the harness, not the code — `runs-decisive/`

The verdict above left open whether the sequential-path step at `3b37bc3c` was steady-state or an
artefact of the experiment's warm-up, which from #440 onward exercised only the batched path. The
test that separates them is equal warm-up for both arms: **`warmups=0`, 300 timed pair iterations**,
released `v0.5.20` against `main` (`9db11325`, with #456 and #454), five interleaved rounds on the
same host class.

```
warmups=0 pairIterations=300 batchIterations=20; runs V=5 A=5
pairP50Millis    V-0520    80.38 [79.10..80.55]   A-main    79.22 [79.09..80.27]   A/V 0.986
pairP95Millis    V-0520   100.49 [99.15..100.94]   A-main    99.03 [98.99..100.49]   A/V 0.985
batchP50Millis   V-0520   448.93 [442.18..449.96]   A-main   587.06 [585.18..594.11]   A/V 1.308
coldLoadMillis   V-0520   160.55 [155.54..166.03]   A-main   223.63 [222.16..232.76]   A/V 1.393
scoreChecksum identical: True
```

**Correction to the verdict above.** `score()` on `main` is **not** slower than v0.5.20: 0.986x with
equal warm-up. The 1.27x step at `W-440` in the bisect table was produced by the harness — v0.5.20's
five `scoreAll` warm-ups ran `score()` thirty times before the timed loop; `main`'s ran it zero
times, and sixty iterations never amortised the compilation. v0.5.20 itself reads 80.4 ms here with
no warm-up against 62.7 ms with warm-up, which is the size of that effect. The profile-pollution
hypothesis in the issue is therefore **not supported** and is withdrawn.

**What stands, reproduced at zero warm-up:** `scoreAll` 1.308x slower (the batched path against
the loop it replaced) and cold load 1.393x slower (the trie constructor). Numerics identical.

The experiment now warms `score()` and `scoreAll()` explicitly, so this cannot recur in this harness.
