# Meta MobileMoE-S QAT INT4 G32 — pure Java qualification

Date: 2026-09-02

This evidence qualifies the gated `facebook/MobileMoE-S-QAT` checkpoint through the Models
pure-Java backend. Models parses the Hugging Face configuration, tokenizer, and Safetensors bundle,
owns the full graph and KV state, and executes every kernel in-process. No Rust shim or external
inference engine participates in the product path.

## Immutable input

- Hugging Face revision: `afda132cad380ac47da5ef055f186884d1c12f65`
- `model.safetensors`: 713,916,240 bytes
- SHA-256: `1a54d8eb2adf19c296a1c129f9cd7d7395c21f5facfc6d13d2945e002d55e37d`
- Artifact layout: signed packed INT4, group size 32, FP16 scales
- License: Meta fair noncommercial research license; Hugging Face access approval is required

The 1.26B-parameter graph activates about 272M parameters per token. It has 20 layers, hidden size
768, 12 query heads, four KV heads, 60 routed experts with top-4 selection, and one shared expert.

## Gated acquisition and reference setup

The checkpoint returned HTTP 401 until the Meta access form was approved for the Hugging Face
account associated with the read-only token. Qualification began only after the pinned revision and
all required files were available through authenticated HTTPS. Credentials stayed in the operator's
ignored `projects/.env`; neither token nor downloaded gated bytes are committed.

The independent reference executes Meta's reviewed Python files from that already-downloaded local
snapshot with `local_files_only=True` and explicit `--trust-remote-code`. The snapshot's
`modeling_mobilemoe.py` imports `configuration_mobilemoe` as a top-level module, so the snapshot
directory must also be on `PYTHONPATH`. Transformers 4.46.3 cannot load the model because it lacks
`transformers.masking_utils`; Transformers 4.57.6 fixes that boundary. PyTorch 2.2.2 then fails to
register the unsigned packed-INT4 parameters, while the model-card-verified PyTorch 2.8.0 succeeds.
The retained control therefore pins Transformers 4.57.6, tokenizers 0.22.2, Safetensors 0.7.0, and
PyTorch 2.8.0. The model card explicitly says not to apply the tokenizer's generic
`fix_mistral_regex` suggestion, because this snapshot uses the Llama 3 tokenizer.

This Python stack is an oracle and benchmark peer only. It is not packaged, downloaded, or invoked
by the Models product path.

## Correctness gates

The official PyTorch implementation and the same pinned checkpoint produced retained full-logit
oracles for BOS and BOS+`hello`. Java preserves both greedy tokens and clears unchanged cosine
floors: greater than 0.9999 at BOS and greater than 0.9997 with KV history. A separate test compares
batched prefill with sequential execution and then verifies the following token from equivalent KV
state. Tokenizer tests pin the official MobileMoE chat template and token IDs.

The independent default-configuration smoke in
[`default-correctness/models-pure-java.json`](default-correctness/models-pure-java.json) answers all
nine cases correctly with no `models.*` tuning properties. Its p95 TTFT is 1,590.5 ms and p95
end-to-end latency is 2,830.7 ms, including the first cold inference case.

## Controlled qualification

- Host: `vectors-bench`
- CPU: AMD EPYC Milan, 4 physical cores / 8 vCPUs, AVX2
- Memory: 32,857,444,352 bytes
- Runtime: Eclipse Temurin 25.0.3 HotSpot C2
- Vector API: 256-bit preferred and active width
- Workload: `general`, 2,048-token context, top-1 retrieval
- Generation: temperature 0, top-k 1, top-p 1, seed 42, repetition penalty 1
- Measurement: one warmup followed by three measured iterations over nine cases
- Runtime properties: library defaults; no model-specific tuning flags
- Comparator: same-host official Transformers implementation, one measured pass over the same nine
  rendered prompt hashes and generation controls

[`models-pure-java.json`](models-pure-java.json) retains all 27 measured attempts and the backend's
resolved diagnostics.

| Metric | Result |
| --- | ---: |
| Successful/correct attempts | 27/27 |
| Performance tier | `PRODUCTION_READY` |
| p95 TTFT | 958.0 ms |
| p95 TPOT | 85.82 ms |
| Median decode | 21.83 tokens/s |
| Median prefill | 78.34 tokens/s |
| p95 end-to-end | 2,315.6 ms |
| Peak RSS | 2,554,105,856 bytes |

[`transformers.json`](transformers.json) retains the independent nine-case control. It answers 9/9
correctly but needs 12,358.0 ms p95 TTFT, 14,635.6 ms p95 end-to-end latency, and reaches 8.43
tokens/s median decode with 4,281,786,368 bytes peak RSS. The Java candidate reaches 2.59x its
decode throughput and 0.158x its p95 end-to-end latency. The checksum-bound
[`qualification.json`](qualification.json) records the `QUALIFIED` decision under
`production-rag-model-contribution-v6`.

The candidate retains three measured iterations for stable production metrics. The expensive
reference runs each case once. Policy v6 permits different repetition and warmup counts only when
artifact identity, hardware, corpus, case list, prompt template, retrieval settings, context,
threads, grounding policy, sampling controls, and the exact distinct `(case ID, prompt SHA-256)` set
match. Repeating the same reference outputs two more times added cost, not coverage; omitting a case
or changing any rendered prompt still makes the control incomparable.

## Optimization history

The first correct direct packed-INT4 graph required about 22.58 seconds to first token. SIMD packed
INT4 kernels, batched prompt execution, and safe activation quantization reduced that to about nine
seconds. Expanding all routed experts to BF16 was both slower and roughly one GiB larger than the
retained runtime layout, so it was rejected.

The accepted path keeps the 681 MiB source checkpoint mapped and prepares row-major Q8_0 execution
weights in backend-owned native memory. Existing Java Vector API Q8-by-Q8 kernels then execute the
projections; grouped QKV and shared gate/up calls reuse activation quantization. A 256-token retained
prefill batch cleared the latency gate. The direct mapped packed-INT4 path remains available with
`-Dmodels.mobilemoe.runtimeLayout=packed-int4` for lower-memory environments.

The primitive measurements and rejected BF16 result are recorded in the Vectors report
`vectors-bench/jmh-results/mobilemoe-packed-int4-runtime-layout-20260902.md`. They also establish a
specific JVM opportunity: a well-lowered scaled signed-INT4 dot product could remove the prepared
Q8 memory copy without restoring the original latency.
