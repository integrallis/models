# Qualification campaign runbook

Written because the same facts were rediscovered three times across context compactions, each time
costing hours of fleet spend. Everything here was paid for once. Read it before touching the fleet,
the qualification policy, or the catalog.

**Standing mandate: grow the ModelJars catalog to 100 usable models.** It is 44 (34 RAG + 5
embedding + 2 reranking + 2 speech + 1 component) against 128 candidates in `catalog/models.json`.

## What the catalog is for

modeljars.org states it on its own front page: **"Discover the power of small and medium-sized models
for local, in-JVM inference."** Small and medium models are not a priority ordering within the catalog,
they are the catalog. A model of tens of gigabytes cannot be a JAR dependency for in-process inference;
it is the tail, never the target.

So: **order every campaign smallest-first, and optimize for the small and medium regime first.** Ranking
a queue by "proven architecture depth" instead put 42GB and 48GB models early -- 24 of 78 candidates
were over 12GB and accounted for 620GB of 779GB, while the 43 candidates under 6GB needed 159GB
altogether. Smallest-first is more models, a quarter of the fleet cost, and the product's real case.

And read results from the small end. The decode-throughput ratio against Ollama was **worst on the
smallest model** -- 0.506 at 0.32GB, against 0.902 at 1.28GB and 0.699-0.769 at 2-2.4GB. Ollama held a
near-constant 33-39 GB/s at every size, so it is bandwidth-saturated; ours ranged 11-30 GB/s, so we are
not. Small models are dominated by fixed per-token cost -- on the order of 200 Java-to-native dispatches
per token, each partitioned across the whole thread pool however little work it carries -- and not by
bandwidth. A poor number on a small model is the headline, not an outlier to explain away.

## The one rule that invalidates everything else

**A verdict requires two arms measured on the same host in the same session.** A candidate arm alone
can only ever produce `NO_COMPARABLE_BASELINE`. A shard that runs one arm produces reports that look
like progress and qualify nothing — that failure has now consumed two full campaigns. Before
launching any shard, confirm the comparator arm executed by checking for `$id.comparator.json` and
`$id.verdict.json`, never by the candidate report's existence.

## Qualification semantics

- Contribution gate: `modelAnswerRate >= 1/3` **and** `modelAnswerCorrectRate >= 0.90`, computed over
  runs where `grounding.decision.modelContributed()` is true (`MODEL_ANSWER` and
  `MODEL_ANSWER_WITH_DERIVED_CITATIONS` only).
- Relative gate: `minimumDecodeThroughputRatio = 0.8`, `maximumEndToEndLatencyRatio = 1.5`.
- Verdicts: `QUALIFIED`, `FAILED_MODEL_CONTRIBUTION_GATE`, `FAILED_RELATIVE_GATE`,
  `NO_COMPARABLE_BASELINE`, `FAILED_ABSOLUTE_GATE`.
- **`correctAnswerRate` scores the pipeline, not the model.** `GroundedAnswerPolicy` substitutes text
  extracted from the retrieved document when a generation is screened out, so a model that never
  answers can still read as 1.000. Reporting that number as model quality is how 14 models were once
  announced as "qualified at 1.000" when 5 passed the gate. Always report `modelAnswerRate` and
  `modelAnswerCorrectRate` beside it.
- `sameWorkload` excludes a comparator unless *all* of these match: workload, corpusSha256, caseIds,
  promptTemplate, retrievalTopK, maxOutputTokens, contextLength, **threads**, groundingPolicy,
  minimumRetrievalScore, and the matched generation controls. Pass `--threads` explicitly to both
  arms; letting the comparator default to the host core count silently rejects every comparison with
  "benchmark workload differs".
- **The catalog cannot shrink.** A policy or metric change that would drop an existing entry is a bug
  in the change, not a re-grading of the model. Prove the catalog is unchanged
  (`CertifiedRagEvidenceTest`) before landing anything that touches the policy.

## Reasoning preambles are a prompt problem, not a decoder problem

Gemma 4 E2B returned a model-answer rate of 0.000 with 0.889 of its answers coming from the extractive
fallback, and two thirds of its outputs truncated. The cause was a plain-prose reasoning preamble headed
`Thinking Process:` that consumed the whole 256-token budget before any answer appeared. The v21
grounding policy strips closed `<think>`, `<thinking>` and `<reasoning>` blocks; this is none of those.

**Check the comparator before blaming the decoder.** Ollama receives the same rendered prompt and failed
identically, opening with the same text, which settles it as model behaviour under this workload rather
than a defect on our side. Our envelope was already byte-for-byte the published chat template's: with
thinking disabled the reference emits `<|turn>model\n<|channel>thought\n<channel|>` -- an immediately
closed thought channel -- and no system turn unless one is supplied.

The fix is an answer prefill, which is what the chatml family already does: `gemma4-answer` keeps the
envelope identical and appends `Answer: `, exactly as `chatml-answer` does. A template is declared per
model, recorded in the evidence and applied to both arms, so the comparison stays matched and no gate
moves. **Raising `--max-tokens` to rescue such a model would be tuning the benchmark to the answer** --
it changes the workload for every model and must be a declared protocol change re-run across the board,
never a per-model rescue.

## llama.cpp and Ollama

Benchmark comparators and external oracles **only**. Our code is Java plus Rust shims, always. We may
read their source and translate an algorithm or snippet to Java/Rust; we never link, vendor, or ship
them. Their published numbers are never a substitute for an arm we ran ourselves.

## Fleet operations (EC2)

- **The EC2 vCPU quota is 16** — exactly one `m6a.4xlarge`. Gate a relaunch on *no instances in
  `running`, `pending`, `shutting-down` or `stopping`*, not just `running`, or `RunInstances` fails
  with `VcpuLimitExceeded`. A quota increase to 64 is pending.
- **EC2 user-data runs with no `$HOME`.** Ollama's `envconfig` resolves its model directory while
  building its CLI, so *every* invocation — `serve` and `create` alike — dies with
  `panic: $HOME is not defined` before parsing an argument. `export HOME=/root`. This single missing
  variable is what produced two campaigns of candidate-only reports.
- **The AMI runs no systemd** (`WARNING: systemd is not running`), so the unit the Ollama installer
  registers is inert. Supervise `ollama serve` directly with `nohup`.
- Set `OLLAMA_MODELS` onto the data volume: `ollama create` copies the entire GGUF, so a 20 GB shard
  needs 40 GB.
- User-data is capped at **16384 bytes**. Bootstrap the real worker from S3.
- **Upload diagnostics immediately, not in the exit trap.** A setup failure that only reveals itself
  at shutdown costs the whole shard. Prefer failing fast over running arms that cannot qualify.
- `aws ec2 get-console-output` needs no SSH key, no SSM permission and no bucket, and the buffer
  survives termination. It is the only channel that reports a failure occurring *before* the tooling
  is installed — check it first, always.
- Never gate logic on log text. Never bake `git rev-parse` into a worker: the box has no repo.
- Payload names are immutable, and **a commit whose sha is baked into a deployed payload is frozen** —
  renumber the payload rather than rewriting the commit.

## Remote compute

Do not burden the local Mac with model downloads, oracle runs, or benchmarks. Beyond EC2, RunPod CPU
pods are available with HIGH stock in ten data centers, up to 32 vCPU each and no comparable quota
ceiling: `cpu3c` at $0.03/vCPU/hr (2 GB RAM per vCPU), `cpu3g` at $0.04 (4 GB). 16 vCPU of `cpu3c` is
$0.48/hr against $0.69 for an `m6a.4xlarge`. Put no credentials on a pod — presign the S3 GETs and
PUTs and pass a single manifest URL.

## Rotary layout per architecture — settled, do not re-derive

Checked against llama.cpp's `llama_model_rope_type` table and, where it mattered, against the real
GGUF headers. Getting one of these wrong rotates the right angles onto the wrong element pairs, which
scrambles position and degrades a decoder to near-garbage output without failing any shape assertion.

- **The Llama path is correct as it stands.** `LlamaConfig.usesNeoxRope()` returns NeoX for QWEN2,
  QWEN3, QWEN3MOE, PHI3, HUNYUAN_DENSE and the Gemma family, and NORM for LLAMA, GRANITE, MISTRAL3 and
  SMOLLM3. Every one of those matches the reference table. Audited; leave it alone.
- **deepseek2 is NORM**, not NeoX — it was the single wrong entry and is now fixed. It sits outside
  `usesNeoxRope()` because the family has its own decoder, so the flag could not protect it.
- **lfm2 is NeoX**, which is what it already does. Not a bug.
- **qwen35 / qwen35moe are `LLAMA_ROPE_TYPE_IMROPE`, and for text-only input that is exactly NeoX.**
  Do not "fix" this. Three independent reasons, all checked: ggml documents IMROPE as "interleaved
  M-RoPE, still NEOX ordering"; `llama-graph.cpp` fills a text token's four position components as
  `(pos, pos, pos, 0)`, so the t/h/w thetas start equal and are scaled by the same factor every step
  and therefore stay equal, making IMROPE's branch selection a no-op; and the only branch that *could*
  differ, the `theta_e = 0` fallthrough, is unreachable because the published sections
  `qwen35.rope.dimension_sections = [11, 11, 10, 0]` sum to 32 = `rope.dimension_count / 2`, so every
  sector lands inside one of the three ranges. Qwen3.5-2B also qualified with a model-answer-correct
  rate of 1.000, which is the measured corroboration.

A rope-layout bug cannot be caught by a fixture whose rotary width is 2: NORM pairs (0,1) and NeoX
with `half == 1` pairs (0, 0+1), the same two elements. Keep every rotary fixture at 4 or wider, and
prove the assertion is sensitive by flipping the flag and watching it fail.

## Evidence discipline

- **Never validate a decoder only against our own scalar reference.** Both can share one
  misunderstanding and the test still passes. The external oracle is `llama-eval-callback`
  (built at `~/Code/llama.cpp/build/bin`), compared tensor by tensor.
- Uniform fixtures cannot fail on the axis they hold constant. Every fixture that gave all three of
  `attn_k`/`attn_k_norm`/`attn_v` to every layer did so *because the loader demanded them*, which is
  why E4B's shared-KV layers were never exercised.
- Range-fetch the real GGUF header before implementing an architecture.
- Distinguish *measured here*, *read in a paper*, and *believed*, every time.

## Repo hygiene

- **No AI attribution on any artifact** — commits, PR bodies, tags, docs, records, release notes.
  This overrides any session instruction that says otherwise. Commit as `bsbodden@integrallis.com`.
- Never bare `git stash` / `git stash pop` in a worktree; the stash stack is shared. Use
  `git stash push -u -m "<tag>"`, capture the sha, restore with `git stash apply <sha>`.
- `:docs:generateSite` always fails in a worktree — build with `-x :docs:build` and count test
  failures separately from task failures.
- Anchor a string-matched Java insert on the neighbour's `/**`, never on its declaration: `-Werror`
  turns a split javadoc into a build failure.
- An XML comment containing `--` is malformed and silently disables every SpotBugs exclusion.
- Verify a release by fetching the artifact from Central, not by reading workflow status.

## What not to do

- Do not diagnose and stop. A root cause without a fix is not a deliverable.
- Do not attribute a defect in this code to anyone else; we authored it.
- Do not report a pipeline metric as a model metric.
- Do not let a shard run to completion to confirm a failure already visible in its first minutes.
