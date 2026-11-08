# Vision arithmetic regression screen

Candidate `c7acc590147032ae78adad01ae1e869b64922274`, measured on two owned
AWS m6a.4xlarge Linux workers, 8 threads, context 1024, maximum 32 output tokens.
The public `VisionTaskQualificationCli` loaded the complete image-question runtime.

All five previously divergent answers now match the pinned llama.cpp sequences:

| Quantization | Case | Answer |
| --- | --- | --- |
| Q4_K_S | TextVQA 39587 | Aegerter |
| Q4_K_S | TextVQA 36979 | Lug nuts |
| Q4_K_S | TextVQA 35501 | Weihenstphaner |
| Q4_K_M | TextVQA 38508 | 0 |
| Q4_K_M | TextVQA 38337 | Electronics |

RMS normalization now accumulates float32 products in float64, and vision SwiGLU
uses the pinned float32 vector exponential. Appended KV rows are rounded once to
F16 and retained in Java-owned storage; native attention borrows the widened rows
without copying and rounding the entire prefix again. Independent fixtures test
these arithmetic contracts. The Linux native suite passed 65 tests with three
existing ignored tests; Clippy passed. The full Mac Java/native test suite and
native coverage gate passed before deployment.

These are targeted correctness observations. They establish neither full-corpus
accuracy nor a speed improvement. The frozen 80-case corpus, matched F32/F16
performance comparisons, confirmation run, and released-coordinate consumer test
remain required. Both model admissions remain false. Build receipts bind each
worker's actual Java jars and native binary to the immutable source payload.
