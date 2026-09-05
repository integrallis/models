# Soprano Q8 scheduling experiment

Date: 2026-09-04

## Question

Would lowering Vectors' Q8 row-parallel threshold from 4,194,304 to 1,048,576 matrix elements
improve Soprano's 2304x512 and 512x2304 language-model projections without hurting the complete
text-to-speech pipeline?

## Environment

- Vultr instance `9c468503-e86b-4b96-bbb3-00d2853979e8`, plan `vc2-4c-8gb`, Newark.
- 4 vCPU Intel Xeon Skylake virtual CPU, AVX2/AVX-512 exposed, 8 GiB RAM.
- Ubuntu 24.04, Temurin 25.0.4.1, Vector API preferred width 256 bits.
- Persistent GGUF executor, four workers, two chunks per worker.
- Soprano 1.1 80M Q8_0, SHA-256
  `4758ad908395dc73a1b973d9a29ce96941f4328594d1c6c1223e7b7710a6a131`.
- Every forced-row microbenchmark verifies bit-identical output against the production Q8 path
  before measuring.

Full CPU and OS data are retained in `cpuinfo.txt` and `os-release.txt`. Raw JMH output is in the
two CSV files; full-pipeline results are retained as Gradle/JUnit XML.

## Results

| Operation | Existing threshold | Candidate threshold | Change |
| --- | ---: | ---: | ---: |
| Q8 gate/up, 2304x512 | 0.890 ms | 0.296 ms | 3.01x faster |
| Q8 down, 512x2304 | 0.841 ms | 0.292 ms | 2.88x faster |
| Q8 output head, 8192x512 | 1.234 ms | 1.101 ms | no material regression |

The isolated production kernel benefits. The complete pipeline does not:

| Full Java Soprano run | Total | Language model | Vocoder | RTF |
| --- | ---: | ---: | ---: | ---: |
| Existing threshold, run 1 | 7.111 s | 5.149 s | 1.899 s | 3.472 |
| Candidate threshold, run 1 | 10.029 s | 7.958 s | 2.009 s | 4.897 |
| Candidate threshold, run 2 | 9.889 s | 7.917 s | 1.909 s | 4.829 |
| Existing threshold, run 2 | 7.542 s | 4.624 s | 2.855 s | 3.683 |

The test synthesizes the same 2.048-second utterance twice in one JVM and reports the second run,
so load time and first-use warmup are outside the measured interval.

## Decision

Rejected. The lower global Q8 threshold makes isolated matrix calls faster, but repeated
per-layer executor barriers make end-to-end language generation roughly 40% slower. Vectors keeps
the 4,194,304-element Q8 threshold. A future optimization must schedule work at the transformer
stage or layer level so it amortizes barriers across projections; it must pass this same
full-pipeline gate before adoption.
