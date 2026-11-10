# LLaVA 1.5 fidelity run — 2026-11-09, runtime 54b086c7

First campaign with committed artifacts for the current `fix/vision-native-arithmetic` head. The
newest campaign previously in the branch records `revision = c7acc590`, which the PR describes as
the earlier attempt retained as failed evidence; nothing under `results/` referenced `78528c14`.

## Use the fidelity gates. Do NOT use the performance gates.

**Fidelity reproduces across hosts and these numbers stand** (confirmed on four hosts, see
`../2026-11-10-host-portability-genoa/`):

| gate | Q4_K_S | Q4_K_M |
|---|---|---|
| `tokenAgreement` | **1.000, zero disagreements** | **1.000, zero disagreements** |
| `accuracyLoss` | PASS, loss +0.0000 both datasets | PASS, loss +0.0000 both datasets |
| `vqaFloor` | PASS 0.6875 | PASS 0.6750 |
| `textVqaFloor` | PASS 0.4156 | PASS 0.3844 |
| `imageContribution` | PASS 0.5375 / 0.375 | PASS 0.5375 / 0.375 |
| `performance` | FAIL — **see below, not usable** | FAIL — **see below, not usable** |

All 80 generated token sequences match the reference exactly on both variants, and the candidate
reproduces the reference's dataset score to four decimals.

**The performance ratios on this host are an artifact of the host, not of the runtime.** The
manifest pins `threads: 8`. Every recorded prior arm ran on a host with 16 logical / **8 physical**
cores (`cpu.txt` in the earlier result directories: `Core(s) per socket: 8`,
`Thread(s) per core: 2`). This host is a Hetzner `ccx33`, which presents 8 logical / **4 physical**
cores (`cpu.txt` here: `Core(s) per socket: 4`, `Thread(s) per core: 2`) — Hetzner's "dedicated"
means dedicated threads, not cores. Eight threads on four physical cores is 2x oversubscribed.

The measured ratios show exactly that signature rather than a code change:

| | prefill | decode |
|---|---|---|
| Q4_K_S | 0.76 - 0.77 | **1.52 - 1.63** |
| Q4_K_M | 0.83 - 0.84 | **1.48 - 1.73** |

Decode at 1.5-1.7 against a recorded 1.05-1.09, while prefill *inverts* to 0.76 (the candidate 24%
faster), is not something a runtime change produces. It matches this project's own measurement that
oversubscription costs about 6.6x on decode against 1.8x on prefill.

A topology-matched repeat is the authoritative source for the performance gates.

## Provenance

- runtime `54b086c730bce68109b789c5a22566fd2f8def20`, JDK 25.0.5, 8 threads, context 1024, F32 KV
- host AMD EPYC-Milan, 8 logical / 4 physical, `avx2 fma`, no AVX-512, `active-vector-bits = 256`
- reference: llama.cpp at `2cbca389f515df455412dafa07e2b1359cc72a6d`, the `oracleCommit` the prior
  receipts record, built Release with a clean source tree
- scorer: pinned `m4c_evaluator.py`, sha256 `8b49e1033e29d551c0bdd86d9149e2b54370e4cd17e52176ac9061b3521f0774`
- model artifacts sha256-verified against `manifest.json` by `prepare.py`
- `receipt.json` carries every jar, artifact and binary hash
- encoder `embeddings.f32` files are omitted (about 755 MB); every receipt and log is retained

How `task-oracle` is built is documented nowhere in the repository, so the command used here was
reconstructed: `g++ -O3 -std=c++17 -I source/include -I source/ggml/include task-oracle.cpp -o
task-oracle -L source/build/bin -lllama`. The prior binary's sha256 is recorded but a compiled
binary will not hash-match across machines, so byte-equivalence with the earlier arms is **not**
claimed.
