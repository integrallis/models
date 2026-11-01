# Q5 final combined-runtime confirmation

Compare previous Java BERT with current Java BERT on 82 pinned SciFact/NFCorpus
embeddings and 10 counterbalanced process pairs of 24 measured embeddings. The
current implementation shares a stateless decoder and row scratch. ARM captures
separate JFR recordings of each arm before timing; those runs are diagnostic only.

Use the same derived Qwen2.5-0.5B all-Q5_1 fixture as final-v1. Rebuild from pinned
source/artifact; require the recorded output SHA. Compare current Java and current
Java+Rust through 148 tasks with nine cloned full-logit snapshots each. Observe
actual provider calls separately: x86 allows batch one; ARM allows none. Both
paths preserve the same output contract. Run 10 counterbalanced process pairs,
32 declared warmups and 3 timed tasks/process. Preserve all warmup records.

Every process is wrapped in `/usr/bin/time -v` for peak RSS. JSON load records
capture startup. Allocation is measured over all Java threads. No resource failure
is excused by a latency gain. Prior failed evidence remains unchanged.

All arms use the same frozen Vectors jar combining the exact activation array
boundary, BF16 dispatch and executor guards. Before any model runs, verify the
public SwiGLU path with 580,911,104 exhaustive relevant float inputs, 10,000,000
random float bit patterns and cold/hot, special-value, offsets/tails/aliasing and
concurrent-caller probes at each forced 64/128/256-bit width. These flags belong
only to correctness probes; model calls retain normal hardware selection.

One isolated 16-vCPU x86 host and one 16-vCPU ARM host, maximum 45 minutes each,
combined budget $2.50, automatic termination on completion/failure/deadline.
