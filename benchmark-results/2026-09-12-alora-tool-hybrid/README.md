# Activated-adapter tool hybrid qualification

This experiment tests whether one Qwen3 base plus a tool-specialist activated LoRA can provide a
real hybrid model while reusing exact KV-cache blocks. It does not treat semantic routing,
duplicated independent caches, or approximate cache translation as cache sharing.

## Candidate topology

- Base candidates are pinned independently. The three rejected initial runs used
  `Qwen/Qwen3-0.6B` revision `c1899de289a04d12100db370d81485cdf75e47ca`; the active candidate uses
  `Qwen/Qwen3-1.7B` revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`.
- Specialist: an aLoRA trained only for tool selection and argument generation.
- Invocation sequence: the base tokenizer's `<|im_start|>assistant\n` token sequence
  `[151644, 77091, 198]`.
- Prefix ownership: system instructions, tool schemas, and the user turn execute with the base
  weights. The specialist first applies at the invocation sequence.
- After a tool executes, the prose branch starts from the same immutable base-prefix blocks and
  recomputes only the short tool-call and tool-result suffix with the base model. Adapter-tainted
  blocks are never read by the base branch.

Training uses external Python libraries only to produce and independently evaluate the adapter.
The releasable runtime must load and execute the base, adapter, cache fork, and tool loop inside the
JVM. No Python service or external inference process is permitted in production.

That boundary is unchanged for a hybrid. Java owns both branches, activation, physical KV sharing,
and the tool loop. A failed performance gate may justify only a measured Models-owned Rust/FFM
kernel with a Java reference and fallback; it may not justify importing or launching an external
inference engine.

## Predeclared gates

These gates are fixed before training or evaluation results are inspected:

1. **Artifact identity:** the adapter records the exact base model/revision, tokenizer hashes,
   invocation token IDs, target modules, rank, scaling, training-data revision/hash, and formatter
   revision. A mismatch fails closed.
2. **Exact prefix:** every layer's K and V values before the invocation sequence are bit-identical
   between base and activated-adapter executions. The Java cache branches must reference the same
   physical immutable prefix blocks; copying equal values does not count as sharing.
3. **Tool syntax:** 100% of evaluated tool responses are structurally parseable and schema-valid.
4. **Tool choice:** at least 85% exact AST accuracy on a pinned, held-out BFCL simple/multiple slice
   and no worse than the unadapted base. On the existing 14-case Models suite, tool selection and
   schema validity must be exact, argument-value agreement must be at least 90%, and the Spring AI
   zipcode regression must pass every check. The existing six-turn virtual-conversation protocol
   must pass all six turns.
5. **Irrelevance:** at most 5% false tool calls on a pinned BFCL irrelevance slice.
6. **Tool loop:** after executing a selected tool, the base branch produces a natural-language
   answer using the structured result in every result-synthesis step of the Java gates. Raw JSON is
   not an answer.
7. **Base behavior:** with no tools, the hybrid's selected tokens are identical to the base model
   on the existing chat oracle; a disabled adapter must be a true no-op.
8. **Quality:** the complete published Java artifacts repeat gates 3–7 on a clean host. Python
   results are an oracle, not production qualification.
9. **Performance:** measure recompute and shared-prefix paths at 256, 1,024, and 4,096 prefix
   tokens. Sharing is enabled by default only above the measured crossover and must improve 4K
   handoff TTFT by at least 20%. Report adapter memory, KV memory, peak RSS, prefill, TTFT, decode,
   and end-to-end latency; do not omit a losing arm.
10. **Packaging:** ModelJars composition metadata must bind the exact released Models version,
    base artifact, adapter artifact, hashes, invocation boundary, cache policy, evidence schema,
    and results. Catalog and site gates must resolve and execute the published artifacts before the
    composition can be listed.

Failure of any correctness gate rejects the candidate. Thresholds are not relaxed after results
are known. Performance work starts only after correctness passes.

## Data and contamination controls

The training candidate is the CC-BY-4.0 `edbuildingstuff/bfcl-ft-data` Java/JavaScript-augmented
configuration because its card records deterministic construction and deduplication from the BFCL
evaluation set. The source is pinned at revision
`a7ceb3b1e1605f609fda6f2befec704f575290de`; the downloaded
`train.javajs.messages.jsonl` is 281,295,398 bytes with SHA-256
`915679b445c64676d7212e8953a0c3379c1024235e3c8168d1af2d2ef2ae7973`. The formatter used by
the two rejected training runs is `train_alora.py` with SHA-256
`43e329817c6e7d7c16790316249c0c6c57d97c5ee7e27dcc132cd657e8ff9d08`; the third run uses the
fail-closed preparation-schema check in formatter SHA-256
`b0a2bbb6c1fabaafa33953853bb592dde74641e333754db0d9ff02fd42417b4e`. Qualification uses the
independently pinned Apache-2.0 `ShishirPatil/gorilla` BFCL repository at revision
`c15b2a151662cac9839c96d7dfb1493b5329c975`. The evaluator verifies the SHA-256 of all three BFCL
v3 question files and both answer files before loading the model. A hand-written Java/Spring tool
set that is not used for training supplies the product-level integration gate.

No evaluation example may enter training. The preparation script must compare normalized query,
tool schema, and expected-call hashes across the two corpora and fail on overlap.

The pinned source cards declare both training datasets CC-BY-4.0 and the Qwen3 base Apache-2.0.
`evidence/qwen3-17b-r32-full-v9/license-audit.md` records V9's exact source revisions, document
hashes, and required attributions. A passing candidate must carry its adjacent `NOTICE` in the
adapter artifact and publication metadata; this repository note alone is not sufficient
compliance.

Before reading adapter results, the evaluation slice is fixed at 100 BFCL v3 simple cases, 100
BFCL v3 multiple cases, and 100 BFCL v3 irrelevance cases. Cases are selected by the lowest unsigned
SHA-256 of `20260912:<case id>`; file order is not used as a quality filter. Exact-call scoring
requires the same call count and tool names, no extra arguments, and a ground-truth-permitted value
for every argument. An omitted optional argument is accepted only when BFCL explicitly includes its
empty-value sentinel. The evaluator records every selected ID, prompt, raw completion, parsed AST,
schema result, and expected call so the aggregate can be independently recomputed.

## Current status

Protocol and physical cache mechanics are implemented; no adapter has passed qualification. The
experimental JVM implementation now includes a
strict safetensors adapter loader, activation-boundary enforcement, physically shared immutable KV
prefixes, request-scoped base/adapter branches, and Spring AI and LangChain4j tool loops. Synthetic
tests prove those contracts, and a property-gated real-weight JVM test loads the exact Qwen3 GGUF
and adapter to verify physical prefix sharing and exact base continuation. These mechanics tests do
not qualify the adapter's tool behavior.

The runtime also has an explicit independent-recomputation arm for the predeclared performance
comparison. It evaluates the same base prefix separately for the base and tool sessions, activates
the adapter only after that prefix, and verifies that the two sessions share zero physical KV
storage. A recomputed first turn can later freeze its retained clean base lineage and fork a longer
turn over shared blocks. This supplies a like-for-like baseline for finding the 256/1,024/4,096-token
crossover instead of comparing the shared path to a different model or prompt. The complete Python
experiment suite currently passes 90 tests, and the affected Models API, pure-Java backend, runtime,
Spring AI, LangChain4j, benchmark, and formatting suites pass together. The Java real-weight
crossover gate now also starts below the measured sharing threshold, retains conversation state,
crosses that threshold on a later turn, and then proves that both branches reference the same
immutable KV arrays while preserving exact base output.

Runtime adapter metadata schema 4 now fails closed unless every training source and its role,
revision and hash, the preparation schema and manifest, prepared split hashes, formatter hash,
the exact selected-row hashes and counts, tokenizer file hashes, base GGUF hash, adapter hash,
target graph, and invocation tokens are present and valid. The loader remains read-compatible with
the older schema 2 and 3 formats. The
packaging script verifies those values against the source manifests and files before producing a
runtime directory; the Java API exposes the same provenance for downstream ModelJars
qualification.

The first adapter, `qwen3-06b-tool-alora-r32-v1`, is rejected. It was trained for one epoch over
12,000 examples at rank 32 and alpha 64. On the fixed 25-case-per-category diagnostic slice it
scored 72% exact tool-call accuracy, 98.67% parseable syntax, 97.33% schema validity, and a 44%
false-tool-call rate. The unadapted base scored 70% exact and 8% false calls. This misses the fixed
85%, 100%, 100%, and at-most-5% gates respectively, so the run was not promoted to the full
qualification slice or packaging. The training source contained only 7.75% no-call examples; the
prepared v1 sample contained 8.63%, which explains the large irrelevance regression and is recorded
as a corpus-design failure rather than excused as model variance.

The second experiment changes the sampling protocol, not the gates: its deterministically selected
12,000-example train split and 1,000-example validation split each contain exactly 25% no-call
examples, remain disjoint from the pinned BFCL evaluation queries, and train for two epochs with the
same rank and alpha. Its train and validation SHA-256 values are
`7c73514edb920d9db4023532e833933f85e7cce3b3de1f17ecbcc80f60c8ffdb` and
`8ab6d919e89b06fe7f51715f7e1c602d70d1760a38e603520c7006bf219895e5`.
Its adapter SHA-256 is
`b7b3397b629494fca4ae6ae59d9c0cae9c5dcc1288f1cb65ea0ae4f2e770a490`.

That second adapter is also rejected. On the fixed 25-case-per-category diagnostic slice it scored
98.67% parseable syntax, 97.33% schema validity, 78% BFCL-compatible exact tool accuracy versus 72%
for the base, and a 36% false-tool-call rate versus 8% for the base. The original diagnostic report
recorded 76% versus 70% because the local evaluator compared nested dictionary answers literally;
the pinned BFCL checker recursively compares their permitted member values and normalizes string
punctuation. A test-first compatibility correction changes only `simple_94` in each arm and does not
alter the rejection. The adapter still misses four fixed gates, so no 100-case qualification,
packaging, catalog entry, or release was attempted.

Inspection of the selected no-call population found that numeric balance alone was insufficient:
random selection mostly supplied obvious, unrelated query/tool pairs. A schema-2 lexical hard-
negative preparation was rejected before training because its source contained contradictory empty
call labels. An unfiltered Hammer function-masking preparation was then rejected because multi-call
prompts and duplicate-capability tools could leave an applicable function behind. A 24% filtered
preparation was also rejected when query-equivalent tools such as email validators survived.

The third training input fixes the remaining catalog-dilution defect: query similarity is measured
against each function independently, so an applicable function cannot hide inside unrelated tool
text. With the fixed called-function cutoff of 0.45 and per-function query cutoff of 0.125, 2,569
high-confidence no-call candidates remain. The frozen split therefore declares a 19% no-call share
rather than weakening either filter. Its 12,000-row train and 1,000-row validation SHA-256 values
are `3fe555c1e4a68b65b6715341cd1d1cbf9995e549cbdb267f15896b0a9b7edf77` and
`f12c4c34c875d929b252d075749468f2c53d919744eca711e907927d9ecd83ca`.
The preparation manifest SHA-256 is
`c7da1f5f826bb1c6f10047e09ff1e7a69e3346dad2336400e5a277ef937090b9`.
The same untouched evaluation slice and thresholds still apply. The resulting rank-32 adapter,
SHA-256 `437c6e1e098768d6d40e256044fdcbcd8d26f2e785f0b4734b9d1d56aa203106`, is also rejected. On the
fixed 25-case-per-category diagnostic it scored 98.67% parseable syntax, 94.67% schema validity,
76% exact tool accuracy, and a 32% false-call rate. The unadapted base scored 64%, 64%, 72%, and 8%
respectively. One fixed syntax failure makes the full 100-case run mathematically unable to meet
the predeclared 100% gate, so no full run, Java behavior gate, performance qualification,
packaging, catalog entry, or release was attempted for this adapter.

The real-weight JVM mechanics gate now passes four cases with the rejected v1 adapter: projection
oracle agreement, one physically shared base/adapter prefix, consecutive tool-selection turns, and
a completed conversational turn followed by another shared-prefix selection. The last case exposed
an important template lifecycle defect: after an assistant response, canonical ChatML rendering
removes generation-only control tokens, so the next prompt can diverge after the immutable prefix
instead of strictly appending bytes. The runtime now reconciles and recomputes only that mutable
suffix while retaining the same physical immutable KV blocks, and refuses any reconciliation that
would replace those blocks. Synthetic tests cover both the valid rerender and the fail-closed case.

The prior Vultr A40-8Q training hosts were deleted after their adapters, manifests, diagnostic
records, and hashes were copied. The multi-candidate host
`33011d59-2d07-4aa4-b611-5d3baee78381` was deleted at 2026-09-13T10:54:30Z and a provider-wide
listing confirmed it absent. Its complete lifecycle is retained under
`evidence/hammer19-per-function-v3/host-lifecycle.md`.

The next bounded candidate is the exact Qwen3 1.7B revision
`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`. Its unadapted fixed 25-per-kind screen reached 86%
exact calls, but its 12% false-call rate and 65.33% raw syntax/schema rates do not qualify it. A real
one-step A40-8Q preflight then completed training, evaluation, and adapter serialization after a
test-first fix disabled gradient checkpointing for the independent evaluation pass. The frozen
rank-16 training configuration and the copied base-screen evidence are recorded under
`evidence/qwen3-17b-r16-v4/` and `evidence/qwen3-17b-base-screen/`. The qualification thresholds
remain unchanged.

The resulting rank-16 v4 adapter is rejected. Its adapter SHA-256 is
`dcaeab65a51ebf9d98c97f7203e5890cb3e24ae592ba1f2358cbc1d8e46895bd`. On the fixed 25-case-per-
category diagnostic it reached 100% parseable syntax and 88% exact tool calls versus the base's
65.33% and 86%, respectively. It still missed two fixed gates: schema validity was 98.67%, and its
24% false-call rate was above both the base's 12% and the 5% ceiling. No full run, Java behavior
gate, packaging, catalog entry, or release was attempted. The report and all 150 base and adapter
records are retained under `evidence/qwen3-17b-r16-v4/`.

The v5 experiment changes only the training population: 4,000 training rows and 500 validation
rows are deterministically resampled from the same verified preparation at exactly 50% call and
50% no-call. The parent split hashes remain bound into the new manifest, and all BFCL queries
remain excluded. The frozen configuration and hashes are recorded before training under
`evidence/qwen3-17b-balanced50-v5/`. The correctness thresholds are unchanged.

V5 was rejected at 98.67% syntax, 98.67% schema, 82% exact calls versus the base's 86%, and 20%
false calls. Continuing v4 on 800 easy no-call examples produced v6: syntax and schema reached
100% and false calls reached zero, but valid exact calls collapsed to 18%. Four adapter
interpolations between v4 and v6 were then screened only on a separate, pinned BFCL live development
set; none reduced false calls without unacceptable loss of valid calls, so none saw qualification.

The v8 experiment replaced the easy negatives with 600 training and 100 validation cases
from pinned BFCL live irrelevance data, after excluding every query shared with the static
qualification inputs. Those hard negatives are mixed with 2,400/400 positive tool calls while
preserving the parents' train/validation boundaries. It continues the exact v4 adapter at a reduced
learning rate. Every source, split, script, and initial-adapter hash was frozen before training in
`evidence/qwen3-17b-hard-mixed-continuation-v8/preflight.md`; the fixed gates remain unchanged.

V8 was rejected. It reached 100% syntax, 100% schema, and 86% exact tool calls, equal to the
unadapted base's 86%, but made false tool calls on 20% of the fixed irrelevance cases versus the
base's 12% and the fixed 5% ceiling. Its adapter SHA-256 is
`5a7541ef7074a76b3b8a8018172729cc7eb10518192218b04873dea723e83d94`. The training manifest,
complete 150-record diagnostic, report, and hashes are retained under
`evidence/qwen3-17b-hard-mixed-continuation-v8/`. No full run, Java behavior gate, packaging,
catalog entry, or release was attempted.

V9 is the first full-corpus Qwen3 1.7B experiment. It starts from the unmodified pinned base,
uses rank 32/alpha 64 for one epoch, and consumes all usable rows from the reproducible 12,000-row
preparation rather than continuing from any rejected adapter. Before training, a new unseen
qualification window was frozen at 100 BFCL simple, 100 multiple, and 100 irrelevance cases after
excluding every previously exposed development query. The case-set and query fingerprints are
bound by `evidence/qwen3-17b-r32-full-v9/qualification-window.json`; the full configuration and
hashes are in that directory's `preflight.md`.

The bounded V9 A40-8Q host is instance `04543d5f-f823-4d62-b9e3-e89a3950642b`, created at
2026-09-13T11:13:13Z with a USD 1.75 ceiling and a hard deletion deadline of
2026-09-13T17:00:00Z. Training started at 11:19:11Z. A local provider-identity watchdog was armed
to delete only that exact instance at the deadline; normal completion copied and verified the
evidence, deleted the host earlier, and unloaded the watchdog. Its complete lifecycle is recorded in
`evidence/qwen3-17b-r32-full-v9/host-lifecycle.md`.

V9 is rejected by the fixed development smoke. Its full-corpus rank-32 adapter reached 88% exact
tool calls versus the base's 86%, but only 98.67% syntax, 98.67% schema, and a 24% false-tool-call
rate. It therefore fails three unchanged gates: 100% syntax, 100% schema, and at most 5% false tool
calls. The unseen 300-case window was not scored; Java behavior, performance, packaging, catalog,
and release work were not attempted for this rejected adapter. The report, all 150 records,
training manifest, preparation manifest, and training log are retained under
`evidence/qwen3-17b-r32-full-v9/` with hashes in its host lifecycle record. The host was deleted and
a provider-wide inventory confirmed zero remaining Vultr instances at `2026-09-13T14:14:22Z`.

V10 tested Hammer-style function and argument masking against V9's specific failure mode. Six V9
false calls followed suggestive API names into tools whose declared capabilities did not answer the
request. Two thirds of V10's examples therefore replace callable and argument names with stable,
opaque identifiers while preserving descriptions, schemas, queries, values, and expected calls.
The preparation also rejects 165 source rows with genuinely ambiguous duplicate callable names;
the validation population is byte-for-byte selected from the same ordered V9 identities, while 21
invalid V9 training rows are deterministically replaced. Two independent preparations reproduced
the 12,000/1,000 splits and manifest exactly. Configuration, hashes, hypothesis, and unchanged stop
rules were frozen in `evidence/qwen3-17b-function-mask-v10/preflight.md` before provisioning. The
run stopped after its first optimizer step because 16-character hash identifiers inflated token
counts and dropped approximately 8% of the usable corpus. It produced no evaluated candidate.

V11 replaces those long names with deterministically shuffled short ordinals. Two regenerations
are byte-identical, and pinned-tokenizer preflight proves the exact V9 usable counts, no-call counts,
multiple-call counts, and validation source-line hash. V11 is separately frozen under
`evidence/qwen3-17b-short-mask-v11/preflight.md`; no threshold or evaluation case changed.

V11 completed but was rejected by the exposed screen: 97.33% syntax, 94.67% schema, 86% exact
calls, and 20% false calls. The sealed 300-case set was not opened. Its evidence was copied and
hash-verified, the exact A40 instance was deleted, the deadline watchdog was unloaded, and Vultr
inventory returned zero instances.

V12 tested the strongest V9 weights unchanged under an explicit capability-applicability system
contract. A new pre-score oracle first found that Models preserved raw schema whitespace while
Qwen's published Jinja template canonicalized it. The test-first correction now matches the
published prompt without moving developer-supplied schema into trusted control segments. The
oracle also rejected a draft policy that spelled Qwen control delimiters in untrusted caller text;
the delimiters were removed rather than weakening the injection boundary. The accepted oracle then
matched all 1,072 prompt bytes and 228 token IDs between Hugging Face and the real Java Qwen3 1.7B
Q8_0 path.

V12 still failed the frozen exposed quality gates. It reached 75/75 syntax and schema, but only
42/50 exact calls versus the 43/50 floor and the same-policy base's 45/50. It made 5/25 false calls
versus the 1/25 ceiling. Neither the live-development screen nor the sealed qualification set was
opened. Evidence is under `evidence/qwen3-17b-applicability-contract-v12/`; the exact A40 instance
was deleted after hash verification and provider-wide Vultr inventory returned zero.

V13 was frozen as a causal two-arm continuation from the exact V9 adapter over the exact V8 hard-
mixed corpus. The control retained ordinary completion loss; the treatment added only a class-
balanced loss at Qwen's first contextual call/no-call token. Local preflight regenerated every
corpus byte, reproduced all 3,311 post-tokenization row identities, found zero overlap with the 735
static BFCL evaluation queries, and proved the auxiliary gradient touched only the two decision
logits at the causal prediction position.

V13 was rejected before treatment. The corrected 32-step control reached 77/81 no-call decisions
and 362/390 call decisions, making the frozen requirement of at least nine additional correct no-
call decisions mathematically impossible. The treatment was stopped before its first optimizer
step; no generation screen or sealed case was opened. The control also exposed and then verified a
fix for an Accelerate autocast wrapper that made live-training and plain-reload logit hashes
incomparable despite bit-identical adapter tensors. Exact results and the rejection rationale are
under `evidence/qwen3-17b-applicability-decision-v13/`. No V13 candidate exists.

V14 was frozen after that rejection and before any V14 model output. It treated the immutable V13
control as a development-only screen baseline and measured the class-balanced decision loss that
the treatment optimized, while retaining class non-regression and exact artifact identity gates.
It passed every execution and artifact gate but failed quality: call decisions improved from
362/390 to 370/390, while no-call decisions regressed from 77/81 to 75/81, balanced accuracy fell
from 0.939411 to 0.937322, and applicability loss rose from 0.183260 to 0.187295. V14 was rejected;
no endpoint, generation, sealed, JVM, or packaging gate was opened. Its evidence and preregistration
are under `evidence/qwen3-17b-applicability-decision-v14/`. The A16 was deleted and Vultr inventory
returned zero instances.

V15 moved the applicability decision into the production Java Q4 runtime and proved physical
base/specialist KV-prefix sharing on all 75 exposed cases. Its held-out screen passed, but the
combined exposed result reached only 45/50 calls against the fixed 47/50 floor; it was rejected
before dual generation. V16 replaced the two-logit margin with an affine head over the activated
hidden state. Its BF16 validation reached 389/390 calls and 79/81 no-calls, but the fixed-zero head
failed to transfer to the production Q4 representation after two false calls in its first three
Java observations.

V17 calibrated that unchanged head on a frozen production-Q4 partition. Calibration passed, but
the untouched screen stopped when three positive misses made its call floor unreachable. V18 then
used all exposed static cases as calibration and froze a disjoint BFCL-live screen. After the
Java/Vectors cold-path determinism defect was fixed and independently verified, final calibration
passed at 47/50 calls, 24/25 no-calls, and 75/75 physically shared prefixes. The live screen was
rejected after four positive misses in its first six positive cases made 47/50 unreachable. The
threshold must not be retuned on those exposed live results. Evidence is retained under the V15
through V18 directories.

A subsequent in-process MiniLM semantic gate was also rejected. It removed every false call but
reduced valid-call recall to 40/50, adding a second model without solving the quality problem. Its
complete 75-case diagnostic is under `evidence/qwen3-17b-semantic-applicability-diagnostic/`.

V19 then scored the exact base and activated call/no-call decisions from the same physically shared
prefix. Calibration selected `specialist + 1.0 * base`, but the rule reached only 40/50 correct
calls and 25/25 no-calls across the exposed development set. That is five fewer correct calls than
V15, so the dual-margin decision family is rejected without opening fresh qualification data. All
75 observations physically shared their prefix. The report, complete log, frozen preflight, and
host lifecycle are under `evidence/qwen3-17b-dual-decision-diagnostic/`.

V20 froze a two-branch rule from the already-exposed margins and ran real Java tool generation on
all 75 development cases. It is rejected: 47/50 call decisions missed the 48/50 floor and 74/75
schema-valid results missed the 75/75 floor. It did meet 24/25 no-call decisions, 43/50 exact
generated calls, 75/75 strict syntax, and 75/75 physically shared prefixes. Live Q4 margins moved
enough to change a predicted call into an abstention; another generated result disagreed with the
source-declared JSON schema even though the upstream evaluator called the same output valid. The
complete in-process report and analysis are under `evidence/qwen3-17b-hybrid-generation-v20/`.
The sealed 300-case window remains unopened, no hybrid was qualified or published, and no remote
infrastructure was provisioned for V20.

Exact Java-vs-oracle equivalence, held-out tool quality, clean-host framework runs, and the
performance crossover gates still require a candidate that first passes the development smoke.
The preceding raw-copy, ridge, and CacheBridge Qwen3 0.6B-to-1.7B translations also failed exact
retrieval and remain rejected.
