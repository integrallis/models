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

## Never inherit a prompt template from an older shard

Building a queue by reusing the template an earlier shard used for the same model looks like using
ground truth and is not: it propagates whatever was wrong before. It carried `deepseek` -- the V1
`### Instruction:` format -- onto DeepSeek-Coder-V2-Lite, months after `deepseek-v2` was added for that
family and transcribed from the model's own published chat template, and it would have kept five Gemma 4
models on a template already *measured* to produce a model-answer rate of 0.000.

Derive the template from the architecture, keep per-model exceptions in one explicit table with a stated
reason each, and audit the whole queue against that mapping before launching. `DEEPSEEK` is deliberately
frozen for the V1 coder models whose evidence is already published; it is not the deepseek2 template.

## Fleet operations (EC2)

- **The EC2 vCPU quota is 192** (`service-quotas get-service-quota --service-code ec2 --quota-code
  L-1216C47A`, read 2026-10-06) — twelve `m6a.4xlarge`, not one. The increase that was pending landed
  and overshot 64. Still gate a relaunch on *no instances in `running`, `pending`, `shutting-down`
  or `stopping`*, not just `running`, or `RunInstances` fails with `VcpuLimitExceeded`: an instance
  releases its vCPUs only when it is gone. **Read the quota, do not trust this line** — it said 16
  for weeks after the increase, and on 2026-10-06 that sent a run looking for RunPod capacity when
  176 vCPUs were free. `scripts/fleet/run-shards.sh` still defaults `VCPU_QUOTA=64`; pass it
  explicitly.
- **Other projects share the account.** A box tagged `project=ribbon-catches-bloom` is not the
  model campaign's; check tags before assuming a running instance is yours, and never terminate one
  you did not launch.
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

## The SQL workload needs its own policy, not the grounded-RAG one

Ran on 2026-09-30 and every model failed the contribution gate while answering correctly. The reason is
structural: `sqlcoder-7b-2` replies `SELECT c.email_normalized, c.customer_id FROM customers c WHERE ...`,
which is a text-to-SQL model doing exactly its job, and the grounded-RAG screen asks for a cited
natural-language claim supported by a retrieved document. The two cases it "passed" it passed by echoing
the context verbatim, citation included, which is worse than failing.

General chat models fare better on the same corpus but still fall back, because the corpus's questions do
not elicit citations the way the general corpus's do.

So a text-to-SQL qualification wants its own policy -- correctness of the emitted query against a schema,
not citation grounding -- and running that corpus under the RAG gate measures the wrong thing. Do not
loosen the grounding gate to accommodate it.

## Publishing a qualified model is four gates, not one merge

Qualifying a model and publishing it are separate achievements. Everything below was discovered by
running the gates locally against a branch that looked ready to merge; every one of them would
otherwise have failed after the merge, when the fix is expensive.

**1. Every newly qualified model needs a default-configuration smoke record.**
`tools/qualification-smoke-gate.mjs` in modeljars refuses any entry that is new or whose evidence
changed unless it carries `defaultConfigurationSmoke`. The record asserts a run under the *public
library defaults*:

- no `-Dmodels.*` property of any kind (`tuningSystemProperties` must be empty)
- `backendDiagnostics.environment["native-quantized-decode"] == "true"` -- the library default is
  now the Rust decode path; `models.native.quantizedDecode` was removed, so a `"false"` here means
  the shim fell back and the run is not measuring what it claims
- `warmups == 0`, `iterations == 1`, `generationControls.promptCache == "longest-common-prefix"`
- `correctAnswerRate == 1` **and** `abstentionAccuracy == 1`, with `failures` empty

The qualification runs cannot stand in for this: they use a tuned decode thread count and qualify
at `modelAnswerCorrectRate >= 0.90`, not at a perfect pipeline score. They used to also force
`quantizedDecode` on, which meant every certified tier described a route an ordinary host did not
get; that setting is gone and the measured path and the shipped path are now the same one. `scripts/run-controlled-rag-qualification.sh` already emits exactly this record; the
fleet equivalent must reproduce its jq predicate rather than invent one.

The gate fetches the report over plain HTTP from
`raw.githubusercontent.com/integrallis/models/<modelsRevision>/<report>`, so `modelsRevision` in
`catalog/qualifications.json` must name a commit that is **reachable on `main`** after the models
release merges. Pointing it at a feature-branch commit works until the branch is squashed away.

**2. The reports the catalog names must exist at the paths it names.** Entries carry `report` and
`reportSha256`; the files themselves live in the fleet's result bucket until someone lands them.
Match them by SHA-256, never by filename — that way the evidence and the claim about it cannot
disagree. Check the *branch worktree*, not the stale `main` checkout, or you will "discover" that
already-landed evidence is missing.

**3. `catalog/model-profiles.json` is generated and gated.** `npm run catalog:profiles:check` fails
on a stale file. Regenerate with `npm run catalog:profiles` after any qualification lands.

**4. Markers reach Maven Central only by explicit dispatch.** A push to `main` runs
`model-artifacts` and publishes markers to **GitHub Packages only** — its `maven-central` job is
guarded by `inputs.target == 'maven-central'` and therefore never runs on push. The Pages deploy,
meanwhile, hard-verifies every marker's POM and JAR against `repo1.maven.org` before it will
publish the site. So the order is:

1. merge the catalog to `main` (GitHub Packages)
2. `workflow_dispatch` **model-artifacts** with `target=maven-central`
3. `workflow_dispatch` **finalize-central** to validate and publish those USER_MANAGED deployments
4. only then `workflow_dispatch` **pages**

Dispatching pages before Central has synchronized fails the deploy on a 404 that looks like a
missing model but is only a missing publish step. Verify the artifacts, not the workflow status.

**Run the gates locally before merging.** All four are runnable on a laptop in under a minute
each: `./gradlew verifyReleaseMetadata` in models, and in modeljars `npm test`,
`npm run catalog:verify-components`, `npm run catalog:verify-compositions`,
`npm run catalog:profiles:check`, plus the smoke gate against `origin/main` as the previous file.

## An unmeasured change must never ride a release branch

A work-proportional matrix-partitioning rewrite was committed to the 0.3.51 release branch with its
own message admitting the gain "still has to be measured on x86". It hung the native kernel. A job
sized from its weight bytes asks for fewer partitions than the pool has threads, and the matrix
dispatch path stored that count into `shared.partitions` — the worker pool's **activation gate**,
where a worker parks while its own index is at or beyond the value and is woken again only when the
ACTIVE thread count rises. One small projection therefore parked every worker above its partition
count permanently, and the next job needing them waited on threads that would never run. Small model
on a wide box: the exact case this catalogue exists to serve.

It was caught by an A/B on one host with one variable, after **1183 tests and thirty green CI checks
passed over it**. Three rules follow, and none of them is optional:

1. **`shared.partitions` is the activation gate, never job state.** A job's partition count travels
   in the generation word and every worker reads it back with `job_partitions(...)`. The dispatch
   path must not write the gate at all.
2. **Test the plumbing, not only the arithmetic.** All six tests written for the partitioning policy
   checked `partitions_for_matrix`'s return value and not one of them published a job through the
   pool, so every one of them passed with the hang present. A kernel-pool change is only tested by a
   test that dispatches. (Same lesson as the decoder plumbing surfaces: graph, planner and adapter
   are different surfaces and a graph test covers one of them.)
3. **A release branch carries the release and nothing else.** A performance idea belongs on its own
   branch, behind its own A/B, with the old library as one arm. If it is not measured it does not
   ship, however plausible the mechanism.

## Published evidence must be measured on the library that ships

The release also moved the default grounding policy from v21 to v23, while fifty of the sixty-five
published qualifications had been measured under v2–v22. A catalogue entry that says QUALIFIED on the
strength of a policy the shipped library no longer uses is not describing the shipped library, and
`correctAnswerRate` is a pipeline metric, so the pipeline that produced it is part of the claim.

The fix is not to rewrite the historical `groundingPolicy` field — that records what the
qualification actually measured and must stay true. It is to give **every** published model a
default-configuration smoke produced by the shipped library, so each entry carries a current proof
that the model answers its whole workload correctly under library defaults. Run the smoke across the
whole catalogue whenever the released library changes the grounding policy, the decoder, or the
kernel — not only across the entries that are new.


## Read the documented procedure before writing a script

The repositories already automate what the campaign needs, and the procedures are written down. A
whole day of failed catalog publishes came from not reading one sentence that was already in
`modeljars/docs/github-setup.md`: the `Model artifacts` workflow's reserved `all` value *bootstraps a
complete catalogue* and `verify` must be run before either publication target. Dispatching `all` for
an incremental publish made Central reject all 44 already-published markers, and the finalize step
correctly refused the batch.

**Before any publish, release, or fleet task: read `CONTRIBUTING.md`, `docs/github-setup.md`,
`RELEASING.md` and this file; read the workflow YAML for its required inputs and environment gates;
run the documented verify or dry-run target; run every local gate.** They all finish in under a
minute. Prefer the existing workflow to a script of your own.

When a script is genuinely needed, three mistakes made in one day are worth naming so they are not
made again:

- **Check the exit status of the command, not of a pipe.** `gh pr merge … | tail -2 || die` reports
  the status of `tail`, so a refused merge was swallowed and the next stage published from the wrong
  catalogue. (The same shape once made a kernel upload report success while the copy was denied.)
- **Never key a wait on a commit message or on list ordering.** A pipeline that waited for a fixed
  commit subject never matched a later commit, and `gh run list --limit 1` returned a run from the
  previous month. Key on content, and pin run ids explicitly.
- **`waiting` is not a failure.** A GitHub run in `waiting` is holding for a deployment-environment
  approval. Counting it against a timeout aborted a healthy run after two hours of correctly waiting
  for a person. Also poll matrix JOBS, not the run: a run has been seen reporting
  `completed/success` while one job was still `in_progress`.

The step-by-step catalog publish lives in modeljars' `CONTRIBUTING.md` under "Publishing newly
qualified models, step by step". Follow it rather than reconstructing it.
