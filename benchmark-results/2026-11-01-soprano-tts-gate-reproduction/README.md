# Reproducing the TTS gate on Soprano, 2026-11-01

Asked for by `MODEL-QUALIFICATION-PLAN-REVIEW.md` item 5: validate the TTS recipe against a model
we already ship before spending anything on a new one. Zero rented compute was used.

It produced a correction rather than a validation.

## The gate already existed, and it is not the one the plan designed

`catalog/speech-qualifications.json` in ModelJars carries `policyVersion
speech-oracle-streaming-latency-v1` and two qualified entries — Soprano Q8_0 on `rust-ffm` and
Soprano BF16 on `pure-java`. It is a **waveform-parity plus streaming-latency** gate, not a
round-trip-WER gate: PCM cosine against the publisher's own PyTorch implementation at a pinned
oracle revision (0.998389 and 0.999331), signal-to-difference in dB (24.9 and 28.7), streaming that
emits audio before completion, and p95 real-time factor and time-to-first-audio over five trials.
`docs/MODALITY-EXPANSION.md:67` had already recorded that an ASR gate should be built "in the shape
of the existing `speech-oracle-streaming-latency-v1`".

The `TtsProductionQualificationPolicy` the plan designed duplicated a shipped gate that nobody had
read first.

## What was run here

The artifact came from the CLI, which pins and verifies it:

```
modeljars pull walkingcat-q8
# 117.5 MiB, SHA-256 4758ad908395dc73a1b973d9a29ce96941f4328594d1c6c1223e7b7710a6a131
```

That digest is the same artifact the catalog's Q8_0 entry qualified.

```
MODELS_SOPRANO_GGUF=<cache path> ./gradlew :models-audio:sopranoIntegrationTest
# 11 passed, 1 skipped

MODELS_SOPRANO_GGUF=<cache path> MODELS_SOPRANO_QUALIFICATION_BACKEND=<pure-java|rust-ffm> \
  ./gradlew :models-audio:sopranoIntegrationTest -i
```

The twelfth test is `recordsFiveControlledStreamingQualificationTrials` and it is the performance
half of the gate. Raw logs: `trials-pure-java.log`, `trials-rust-ffm.log`,
`trials-rust-ffm-pass2.log`. Machine: `machine.txt`.

`derived.json` is produced by `derive.py` from those logs. No number below is hand-typed:

```
python3 derive.py trials-pure-java.log trials-rust-ffm.log trials-rust-ffm-pass2.log > derived.json
```

## Result: the gate runs, and a single pass of it cannot carry a verdict

| pass | trials | p95 RTF | median RTF | p95 TTFA ms | median TTFA ms | min blocking cosine |
| --- | --- | --- | --- | --- | --- | --- |
| `pure-java` | 5 | 1.757446 | 1.716477 | 1977.553 | 1931.105 | 1.0 |
| `rust-ffm` | 5 | 1.564047 | 0.785890 | 1227.565 | 624.817 | 1.0 |
| `rust-ffm-pass2` | 5 | 0.583533 | 0.570543 | 411.118 | 394.714 | 1.0 |

Streaming PCM equals blocking PCM at cosine 1.000000000 in all fifteen trials, and every trial
emitted its first audio before completion. That half of the gate is solid.

The throughput half is not. **Two five-trial passes of the same artifact, the same backend, the
same machine and the same protocol differ by 2.68x in p95 real-time factor** (1.564047 against
0.583533). Pass 1's first trial reads 1.564 and then settles monotonically to 0.786; pass 2 runs
0.533–0.584 throughout. The native path needs more warm-up than the one blocking synthesis the test
performs before its loop — the catalog's own reports record `warmups: 1` for the same reason, and on
this machine one is not enough.

The consequence for the pure-Java-to-native ratio is direct: it is 2.184x against pass 1 and 3.008x
against pass 2 for median RTF, and 3.091x against 4.892x for median TTFA. Reporting either single
figure would read as more settled than the data is.

**Each of the catalog's two Soprano entries rests on one such pass.** Their recorded p95 figures
(1.225912 and 1.886551) were measured on a different machine, so they are not comparable with the
figures above and nothing here contradicts them. What the across-pass spread does say is that the
protocol needs replication before a p95 real-time factor decides anything — the same conclusion the
plan reached for decode throughput, now measured on the TTS path.

## Two gaps in the gate as it stands

1. **Nothing computes it.** `grep -rn p95RealTimeFactor` over the repository matches the two Soprano
   reports and one fleet fixture, and no script. The performance block in those reports was
   transcribed by hand from test stdout. `derive.py` here closes that for the performance half; the
   oracle half comes from a separate comparison and is untouched.
2. **The test prints at `-i` only.** At Gradle's default log level the run shows `BUILD SUCCESSFUL`
   and no numbers. I first read that as the test skipping and misdiagnosed a Gradle daemon
   environment problem that does not exist. A gate whose output is invisible at the default log
   level is a gate that gets read wrong.

## Machine

Intel Core i7-9750H at 2.60 GHz, 12 logical cores, 32 GiB, macOS 26.6.2 (25G83). A laptop, with a
Gradle build and a Cargo build in the same session — which is part of why pass 1 looks the way it
does, and is recorded rather than excluded.
