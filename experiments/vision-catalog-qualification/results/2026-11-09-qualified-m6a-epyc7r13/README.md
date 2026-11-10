# LLaVA 1.5 7B qualified — 2026-11-09, runtime 54b086c7

**Both variants pass all six task gates.** `compare.py` exits 0.

| gate | Q4_K_S | Q4_K_M |
|---|---|---|
| `accuracyLoss` | PASS, **+0.0000** both datasets | PASS, **+0.0000** both datasets |
| `vqaFloor` | PASS 0.6875 | PASS 0.6750 |
| `textVqaFloor` | PASS 0.4156 | PASS 0.3844 |
| `imageContribution` | PASS 0.5375 / 0.375 | PASS 0.5375 / 0.375 |
| `tokenAgreement` | **1.000, zero disagreements** | **1.000, zero disagreements** |
| `performance` | PASS, worst **0.9546** | PASS, worst **1.0113** |

All 64 nonblank token sequences match the reference exactly on both variants, and the candidate
reproduces the reference's dataset score to four decimals.

Per-case ratios against the 1.10 ceiling:

| case | Q4_K_S prefill / decode / total | Q4_K_M prefill / decode / total |
|---|---|---|
| vqav2-132415014 | 0.9546 / 0.8873 / 0.9537 | 1.0113 / 0.8974 / 1.0089 |
| vqav2-423731001 | 0.9434 / 0.8806 / 0.9420 | 0.9877 / 0.8933 / 0.9855 |
| textvqa-36063 | 0.9478 / 0.8722 / 0.9469 | 0.9755 / 0.8851 / 0.9744 |
| textvqa-37476 | 0.9359 / 0.8638 / 0.9350 | 0.9869 / 0.8808 / 0.9856 |

Decode is 11-14% faster than the reference on every case of both variants.

## Host, and why it matters

`m6a.4xlarge`, **AMD EPYC 7R13, 16 logical / 8 physical cores, AVX2 only, no AVX-512** — the same
class every recorded campaign ran on (`Core(s) per socket: 8`, `Thread(s) per core: 2` in their
`cpu.txt`). The manifest pins `threads: 8`, so the pool lands on eight physical cores with no
oversubscription.

**This run exists because the host class turned out to decide the performance gates.** Two Hetzner
boxes measured the identical runtime at decode ratios of **1.44 to 1.67** — a `ccx33` with
8 logical / 4 physical cores (`results/2026-11-09-milan-4core-oversubscribed/`) and a 16-physical-core
`cpx62` Genoa box. On the registered class the same code measures **0.86 to 0.90**. The earlier
figures were properties of the hardware rented, not of the runtime, and are retained as failed
evidence rather than deleted.

## Retracted: token agreement IS host-portable

An earlier version of this file claimed the arithmetic was not host-portable and that the cause was
unidentified, on the strength of one `cpx62` Genoa run reporting 0.96875 / 0.984375.

**That claim is withdrawn.** See `../2026-11-10-host-portability-genoa/`: the same CPU model
produces **byte-identical token sequences to this run — 0 of 80 differing** — with AVX-512 fully
exposed and again with the JVM capped to 256-bit vectors. EC2 EPYC 7R13, Hetzner Milan and Genoa in
both configurations all agree.

Ruled out by measurement: instruction set, vector width, the `-XX:MaxVectorSize` flag that had been
named as the likely cause, thread count (pinned at 8, and the rayon pool is built from that value),
and the corpus (identical `corpusSha256`). The diverging run's artifacts were not retained, so its
own cause cannot be established — it was one unreproducible run reported as a property of the code.

## Provenance

- runtime `54b086c730bce68109b789c5a22566fd2f8def20`, JDK 25.0.5, 8 threads, context 1024, F32 KV
- reference llama.cpp at `2cbca389f515df455412dafa07e2b1359cc72a6d`, the `oracleCommit` every prior
  receipt records, built Release from a clean source tree
- scorer: pinned `m4c_evaluator.py`, sha256 `8b49e1033e29d551c0bdd86d9149e2b54370e4cd17e52176ac9061b3521f0774`
- model artifacts sha256-verified against `manifest.json` by `prepare.py`
- native projections and `NativeCausalAttention` confirmed in the recorded diagnostics
- `receipt.json` carries every jar, artifact and binary hash
- encoder `embeddings.f32` omitted (~755 MB); every receipt, log and verdict retained

`task-oracle`'s build command is documented nowhere in the repository, so it was reconstructed:
`g++ -O3 -std=c++17 -I source/include -I source/ggml/include task-oracle.cpp -o task-oracle
-L source/build/bin -lllama`. A compiled binary will not hash-match across machines, so
byte-equivalence with earlier arms is **not** claimed. An A/B on one host showed `GGML_NATIVE=OFF`
moves reference decode by only -1.5% to +1.3%, so build flags are not a material variable.
