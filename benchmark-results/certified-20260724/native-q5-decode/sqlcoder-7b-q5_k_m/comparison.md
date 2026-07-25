| Backend | Version | Load ms | p95 TTFT ms | p95 TPOT ms | Prefill tok/s | Decode tok/s | Peak RSS GiB | vs llama.cpp | Output match | Latency tier | Relative |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| rust-ffm | models-native-3c7bd98 | 252.8 | 1409.5 | 113.3 | 14.27 | 9.41 | 4.67 | 92.8% | 100.0% | USABLE | COMPETITIVE |
| llama.cpp | llama.cpp-b10012-c71854292 | 749.0 | 1156.2 | 99.7 | 18.21 | 10.14 | 4.73 | 100.0% | 100.0% | USABLE | COMPETITIVE |
| ollama | ollama-0.32.0 | 3049.1 | 1067.9 | 99.9 | 19.43 | 10.11 | 4.79 | 99.7% | 100.0% | USABLE | COMPETITIVE |
