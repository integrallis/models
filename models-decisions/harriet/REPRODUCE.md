# Reproducing Harriet

Every number in `README.md` comes from these commands. Nothing is quoted from a paper, a vendor
page, or another system's self-report.

## Host

Hetzner CCX33, 8 dedicated vCPU AMD EPYC-Milan (AVX2), 30 GiB, Ubuntu 24.04, $0.266/hr.
Temurin JDK 25 (verified bit-identical on JDK 27 EA `jdk-27+35-ea-beta` over 40 items).
`vectors-core` 0.1.22 Panama path, `rust-ffm` native kernels.

A dedicated host matters: a shared vCPU box produced latency swings that reversed a prefill
conclusion twice. Every timing here is from an idle dedicated box.

## Weights

```
Qwen3.5-4B-Q4_K_M.gguf   2,740,937,888 bytes
sha256 00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4
```

Verify the digest, do not trust the size: an earlier Granite download matched on byte count and
differed on hash.

## Build

```
./gradlew :models-decisions:testClasses :backend-native:jar
```

`jdk.incubator.vector` is required, never optional. The native kernel must actually build —
a run that silently falls back to pure Java landed within 1% of a previous number and hid an
8.2x difference. `-Ddecisions.requireKernel=rust-ffm` fails the run closed if it does.

## Prepare the tasks

```
python3 benchmark-results/2026-09-22-cascade-and-cuda/prepare_semif.py semif-flat.tsv
```

Renders the 231 public JevBench decisions into the chat-template prompt described in
`ATTRIBUTION.md`. The chat wrapper was rendered from our own GGUF's `tokenizer.chat_template`
with jinja2 under `add_generation_prompt=True, enable_thinking=False`, not written by hand.

## Run

```
java --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED -Xmx24g \
  -Dmodels.purejava.maxContextLength=8192 \
  -Dmodels.purejava.prefillBatchSize=512 \
  -Ddecisions.letterLogits=true \
  -Ddecisions.rawPrompt=true \
  -Ddecisions.kernel=rust -Ddecisions.requireKernel=rust-ffm \
  -Ddecisions.warmupItems=3 \
  -cp "demo-dist/*:demo-dist:native" \
  com.integrallis.models.decisions.JevBenchRunner \
  qwen4b.gguf semif-flat.tsv harriet-results.tsv 1.0
```

`warmupItems` is load-bearing. Without it the first two items of every run are scored on
partly-interpreted code and are not reproducible between JVMs; with it the whole run is
bit-identical across processes. See `../benchmark-results/2026-09-22-cascade-and-cuda/DETERMINISM.md`.

## Score

```
python3 benchmark-results/2026-09-22-cascade-and-cuda/jevscore.py \
  harriet-results.tsv <tokens-per-decision> 0.03 cpu
```

The scorer imports `jevbench.composite_v12` and `jevbench.metrics` rather than reimplementing
them. Token counts come from `TokenCount.java` against the same GGUF, not from a characters/4
estimate.

**Do not pass `--calibrate` for a published number.** It fits a temperature on half the hard tier
and reports on the other half. That is disclosed benchmark tuning; Harriet's headline runs without
it because the 4B is natively calibrated. On the 0.6B the same flag was worth +10.5 composite
points, which is exactly why it is not used.
