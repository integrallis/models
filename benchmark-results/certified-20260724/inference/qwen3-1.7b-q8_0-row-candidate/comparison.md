| Backend | Version | Load ms | p95 TTFT ms | p95 TPOT ms | Prefill tok/s | Decode tok/s | Peak RSS GiB | vs llama.cpp | Output match | Latency tier | Relative |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| pure-java | models@10949c7c4273a2769365cdca80091ba91c7f2015 vectors@fde9858901624d1661a1cf51195d2c59737bcf87 | 378.1 | 4187.0 | 54.1 | 37.85 | 18.63 | 2.37 | 74.4% | 40.0% | OFFLINE | VIABLE |
| llama.cpp | version: 10012 (c71854292) | 1548.0 | 1123.7 | 43.6 | 141.91 | 25.04 | 2.27 | 100.0% | 100.0% | USABLE | COMPETITIVE |
| ollama | ollama version is 0.32.0 | 2477.4 | 1261.3 | 61.4 | 142.43 | 17.74 | 2.32 | 70.8% | 100.0% | USABLE | VIABLE |
