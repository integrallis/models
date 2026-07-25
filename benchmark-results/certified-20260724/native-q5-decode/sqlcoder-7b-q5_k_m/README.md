# SQLCoder Q5_K native decode qualification

This directory retains the controlled CPU qualification for Models commit `3c7bd98`. That commit
fixed single-token transformer dispatch so an eligible injected kernel is used during decode instead
of only during batched prefill.

## Controls

- Host: `vectors-bench`, Ubuntu Linux 6.8, AMD EPYC Milan, 32 GiB RAM
- JVM: GraalVM Community Java 25.0.3, four active processors
- Native workers: four
- Models: `3c7bd98`
- Vectors: `fde9858901624d1661a1cf51195d2c59737bcf87`
- ModelJars: `936034cd55174fccd42121aa1dc7309c49d55d6c`
- llama.cpp: build 10012, commit `c71854292`
- Ollama: 0.32.0
- Model: SQLCoder-7B-2 Q5_K_M, 4,783,256,288 bytes
- Model SHA-256: `0068f25d1fc37cb25aa6be85064432eeeb1a0754d97139c0d2eb3529fc8fc32b`
- Prompt: raw `SQL:` with deterministic nonce prefixes
- Sampling: greedy, temperature 0, top-k 1, top-p 1, seed 42
- Repetition: two warmups and ten measured trials

The host was checked for competing inference processes before the retained runs. Every backend used
the exact GGUF bytes and four inference workers.

## Strict comparator

The strict run generated four tokens, before SQLCoder's early stop. All ten corresponding output
hashes match across Models, llama.cpp, and Ollama:
`b828ee954291cba36252f48c256a3f64e509979c20e551ddd7af2ea97c178d33`.

| Backend | p95 TTFT | p95 TPOT | p50 prefill | p50 decode | Peak RSS | vs llama.cpp decode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Models Rust/FFM | 1,409.5 ms | 113.3 ms | 14.27 tok/s | 9.41 tok/s | 4.67 GiB | 92.8% |
| llama.cpp | 1,156.2 ms | 99.7 ms | 18.21 tok/s | 10.14 tok/s | 4.73 GiB | 100.0% |
| Ollama | 1,067.9 ms | 99.9 ms | 19.43 tok/s | 10.11 tok/s | 4.79 GiB | 99.7% |

Models reaches 93.1% of Ollama decode throughput. All three implementations are classified
`USABLE`, and the generated comparison classifies Models as `COMPETITIVE`.

## Dispatch retention gate

With identical current code and controls, disabling `models.native.quantizedDecode` reduced p50
decode from 9.41 to 1.74 tok/s and increased p95 TPOT from 113.3 to 579.1 ms. Correct single-token
dispatch therefore provides a 5.42x decode gain and changes SQLCoder from `OFFLINE` to `USABLE`.
Prefill is unchanged, which is expected because batched prefill already reached the Rust kernel.

## Early-stop screen

A second run allowed up to 64 generated tokens. SQLCoder stopped after ten visible tokens in Models;
the HTTP engines report the terminal token as an eleventh generated token. The benchmark comparator
correctly rejects those unequal token-count series, so these reports are diagnostic rather than a
strict comparison.

| Backend | p95 TTFT | p95 TPOT | p50 decode | Relative to llama.cpp |
| --- | ---: | ---: | ---: | ---: |
| Models Rust/FFM | 1,413.1 ms | 118.7 ms | 8.56 tok/s | 84.2% |
| llama.cpp | 1,167.3 ms | 102.3 ms | 10.16 tok/s | 100.0% |
| Ollama | 1,087.5 ms | 101.6 ms | 9.95 tok/s | 97.9% |

Models reaches 86.0% of Ollama in this longer early-stop screen. Nine of ten output-text hashes
match all three engines; one nonce case diverges after the four-token oracle. This screen supports
the latency conclusion but is not an exact-output qualification.

## Report SHA-256

| Report | SHA-256 |
| --- | --- |
| `models.json` | `a20e25c0afa2371b83eed5cc1b65e2c85857c31dbdbf2e88e2a6f761280939c0` |
| `llama.cpp.json` | `f6304db26506b944710834a1fb0114462f0fc68c5aef336b47686c4bcc69f94f` |
| `ollama.json` | `3e118910385b4c20af287b96f0b35aacd21e2b44cbe0448d0ecb0cbd68c80e58` |
| `comparison.json` | `24705f2fd6107a6a624a62730c1553a22589de30024b1437f28663a2c79db1dd` |
| `decode-disabled.json` | `a0a3918b59ceeebc6684b489ee7cbd85275f20abeabf576c9294b2ff57de4953` |
| `models-early-stop.json` | `f358da1274a948216fe95578f80395db85282f2943393fa75e0bd1e6e36a737e` |
| `llama.cpp-early-stop.json` | `9fc1f171e2a056a55e4af8b4ed575f7ac549c2609f47309f7eaeb7770bebaf05` |
| `ollama-early-stop.json` | `3ea9afc4a74214ba0b67c2af931d43aabaefcf625e573f94269c4a3ba9c70c8f` |
