| Backend | Version | Load ms | p95 TTFT ms | p95 TPOT ms | Prefill tok/s | Decode tok/s | Peak RSS GiB | vs llama.cpp | Output match | Latency tier | Relative |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| pure-java | models@10949c7c4273a2769365cdca80091ba91c7f2015 vectors@fde9858901624d1661a1cf51195d2c59737bcf87 | 286.4 | 764.9 | 22.5 | 214.49 | 44.86 | 0.98 | 45.6% | 60.0% | RESPONSIVE | NEEDS_OPTIMIZATION |
| llama.cpp | version: 10012 (c71854292) | 574.0 | 296.8 | 11.7 | 573.90 | 98.31 | 0.58 | 100.0% | 100.0% | INTERACTIVE | COMPETITIVE |
| ollama | ollama version is 0.32.0 | 833.8 | 343.0 | 26.5 | 587.44 | 45.02 | 0.63 | 45.8% | 100.0% | INTERACTIVE | NEEDS_OPTIMIZATION |
