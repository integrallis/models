| Backend | Version | Load ms | p95 TTFT ms | p95 TPOT ms | Prefill tok/s | Decode tok/s | Peak RSS GiB | vs llama.cpp | Output match | Latency tier | Relative |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| pure-java | models@10949c7c4273a2769365cdca80091ba91c7f2015 vectors@fde9858901624d1661a1cf51195d2c59737bcf87 | 327.3 | 7167.8 | 63.6 | 21.27 | 15.87 | 1.09 | 21.9% | 10.0% | OFFLINE | NEEDS_OPTIMIZATION |
| llama.cpp | version: 10012 (c71854292) | 1121.0 | 509.6 | 14.4 | 307.43 | 72.51 | 1.12 | 100.0% | 100.0% | RESPONSIVE | COMPETITIVE |
| ollama | ollama version is 0.32.0 | 1446.0 | 630.9 | 28.8 | 308.19 | 41.42 | 0.87 | 57.1% | 100.0% | RESPONSIVE | VIABLE |
