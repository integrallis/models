| Backend | Version | Load ms | p95 TTFT ms | p95 TPOT ms | Prefill tok/s | Decode tok/s | Peak RSS GiB | vs llama.cpp | Output match | Latency tier | Relative |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| pure-java | models@10949c7c4273a2769365cdca80091ba91c7f2015 vectors@fde9858901624d1661a1cf51195d2c59737bcf87 | 393.7 | 765.5 | 17.3 | 207.68 | 60.34 | 1.25 | 59.4% | 40.0% | RESPONSIVE | VIABLE |
| llama.cpp | version: 10012 (c71854292) | 1112.0 | 364.9 | 13.3 | 458.32 | 101.57 | 1.19 | 100.0% | 100.0% | INTERACTIVE | COMPETITIVE |
| ollama | ollama version is 0.32.0 | 1471.4 | 512.5 | 26.0 | 461.14 | 47.99 | 1.01 | 47.2% | 100.0% | RESPONSIVE | NEEDS_OPTIMIZATION |
