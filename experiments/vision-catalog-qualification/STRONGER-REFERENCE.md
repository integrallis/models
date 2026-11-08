# Additional F16 task performance coverage

Declared before collecting these timings. The frozen task reference uses F32 KV to
match the candidate. Catalog admission also needs the faster available llama.cpp
F16 configuration, rather than relying on the matching-storage result alone.

For **each** quantization, repeat the same four already-selected performance cases
with F16 KV, 8 threads, context 1024, batch 128, greedy decoding and 32 output tokens.
Reuse the independent encoder outputs and its recorded timings, as in the F32 arm.
Use the same one warmup and three measured repetitions. Save a distinct binary and
raw JSONL file; the original F32 files and binary remain untouched.

The final per-case latency ratio is the worse of candidate/F32 and candidate/F16,
for prefill, decode and total. The ceiling remains 1.10. Compare generated sequences
as well; differing generation lengths do not establish a matched decode-speed claim.
The F32 arm remains the full-corpus numerical fidelity comparator. This additional
arm covers the registered performance subset, not full-dataset F16 accuracy.

Run sequentially after the task campaign and the integration confirmation have both
exited. Do not overlap workloads on the worker. Preserve all failures.
