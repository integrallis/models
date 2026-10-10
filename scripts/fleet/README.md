# Qualification fleet worker

`qual-worker-two-arm.sh` is the EC2 user-data a qualification worker runs. It lives here because the
evidence behind every catalog entry comes out of it, and for the whole 2026-09 campaign it existed only
in a scratch directory: the script that produced the numbers was not reviewable, not diffable, and not
recoverable.

## Why two arms

Qualification is comparative. `RagProductionQualificationPolicy` compares a candidate against baselines
measured on the **same host** and the **same workload**, and returns `NO_COMPARABLE_BASELINE` when there
are none. The single-arm worker this replaces produced fourteen candidate reports and zero verdicts,
which is how a campaign came to report models as qualified that had never been assessed.

Per model the worker now:

1. downloads the GGUF and runs the candidate arm (`rust-ffm`);
2. `ollama create`s **that same file** so `artifactSha256` matches, and runs the comparator arm;
3. runs `RagQualificationCli` and uploads candidate, comparator and verdict.

## The settings that must match, and why threads are explicit

`sameWorkload` compares workload, corpus sha, case ids, prompt template, retrieval top-k, max output
tokens, context length, **threads**, grounding policy, minimum retrieval score, and the matched
generation controls. Any difference excludes the comparator with `benchmark workload differs`. A run
that passed `--threads 8` to the candidate and let the comparator default to the host's core count was
rejected for exactly that, so both arms are given `--threads $THREADS` from one variable.

## What a verdict can say

- `QUALIFIED` — contribution gate and relative gate both passed.
- `FAILED_MODEL_CONTRIBUTION_GATE` — under a 1/3 model-answer rate, or under 90% correct among the
  answers the model did contribute. The pipeline answered; the model did not.
- `FAILED_RELATIVE_GATE` — slower than the comparator beyond the policy's floor and ceiling.
- `NO_COMPARABLE_BASELINE` — the comparator was excluded; `exclusions` in the verdict names why.

## The manifest is the knob, not the script

`fleet-shard-NNN.json` is a list of jobs, each `{id, uri, tpl, gb, arch, wl, dt, mt}`:

| field | meaning |
| --- | --- |
| `tpl` | prompt template, from what the architecture already qualified with |
| `wl` | **the workload corpus**, derived from the model's declared capabilities |
| `dt` | decode threads, or omit for the pool default |
| `mt` | max output tokens, 256 when omitted |

Two of these were learned by getting them wrong, both on 2026-10-09.

**`wl` must match the capability.** A shard built with `wl: "general"` for every model returned
`FAILED_MODEL_CONTRIBUTION_GATE` for a math specialist and a translation model, while the same base
models were already qualified in the catalog on `math` and `multilingual`. Re-running
`eurollm-1.7b-instruct Q4_K_S` on `multilingual` with nothing else changed returned **QUALIFIED**.
Retrieval was perfect in the failing runs -- recall, MRR, factCoverage and both citation metrics all
1.0 -- so the corpus was the only variable. Put one workload per shard so a box loads one corpus.

**`mt` must fit the model.** `fin-r1-7b` returned `truncatedAnswerRate 0.778` and
`modelAnswerCorrectRate 0.0` at the old hard-coded 256: seven of nine answers were cut off
mid-sentence, so the metric measured the cap rather than the model. It was hard-coded in two places
and is now one per-job variable handed to **both** arms, because `sameWorkload()` compares max
tokens and a mismatch excludes the comparator.


## The payload, the kernels JAR and the version label are required inputs

`QUAL_PAYLOAD`, `QUAL_KERNELS` and `QUAL_BACKEND_VERSION` must all be set in the bootstrap's
environment. The worker refuses to start without them, and refuses to start if any of them
disagree.

Both used to be hard-coded in `qual-worker-two-arm.sh`, and both went stale. The committed file
said `models-rag-bench-0.3.50-v23.tar` and `BACKEND_VERSION="models@0.3.50+v23-08d9b5e1cef8"`
while the 2026-10-09 campaign was running a `0.3.56-dev` payload and stamping `0.3.56-dev` on
every report. The worker that ran was an edited copy, so the committed one — the entire reason
this script lives in the repository rather than a scratch directory — named a build nobody had
run.

A wrong `BACKEND_VERSION` is not cosmetic. It is a false provenance claim written into evidence
that goes on to back a catalog entry, and it is unfalsifiable once the box is gone.

So the worker parses the version out of the label and requires it to appear in the payload object
name:

| `QUAL_BACKEND_VERSION` | `QUAL_PAYLOAD` | |
| --- | --- | --- |
| `models@0.3.56+v24-7ac4153` | `models-rag-bench-0.3.56-v24.tar` | starts |
| `models@0.3.56+v24-7ac4153` | `models-rag-bench-0.3.50-v23.tar` | `VERSION_PAYLOAD_MISMATCH` |
| `local` | anything | refused: a label naming no version cannot be checked |

### The kernels JAR was the worse case

`backend-native`'s published JAR carries **classes only** — zero `META-INF/models/native` entries
— so the `.so` comes *exclusively* from a separate platform JAR. The worker used to fetch that
under the fixed name `models-kernels-linux-x86_64.jar`, and the object on S3 under that name dated
from **2026-09-28** while the Java side was 0.3.56.

So every run measured a September Rust kernel against a current library, and said nothing about
it. Both declare `abi=6`, so it loaded cleanly and nothing looked wrong:

| | sha256 | size |
| --- | --- | --- |
| released 0.3.56 | `064cfae7…` | 485,960 |
| the one in use | `ce25a986…` | 480,536 |

The kernels JAR is now a required, version-checked input like the payload, and the worker logs the
`.so` digest from the JAR's own `native.properties` so a report can be checked against the kernel
that produced it rather than trusted. A versionless `models-kernels-linux-x86_64.jar` is refused.

The released platform JARs are artifacts of the release run itself
(`release-native-<platform>`), which is where to get one rather than rebuilding it.

The check is a named function bracketed by `# >>> BEGIN payload_label_version` markers.
`qual-worker-guards-test.sh` extracts that block and exercises **the shipped implementation**
rather than a copy, because the worker is fetched from S3 as one self-contained file and cannot
source a helper. If a marker is renamed the test fails loudly rather than silently testing
nothing.

```
bash scripts/fleet/qual-worker-guards-test.sh
```
## Building shard files: `build-shards.py`

`fleet-shard-NNN.json` used to be written by hand, and both fields that were learned the hard way
above were learned by getting them wrong by hand on the same day. `build-shards.py` builds them
from recorded evidence instead:

```
python3 scripts/fleet/build-shards.py \
    --ids ids.txt --reports ./prior-reports --catalog ../model-jars/catalog/models.json \
    --out-dir /tmp --start-shard 640 [--mt-override id=2048] [--wl-override id=finance]
```

**It copies, it does not derive.** For each requested model it reads `wl`, `tpl`, `dt` and `mt`
out of a prior candidate report for *that same model* and reuses them exactly, printing the path
it read each one from. Where no prior report exists it prints the precedent it found and **fails**,
writing nothing.

That restraint is deliberate, and it is not caution for its own sake. The obvious design —
derive `wl` from the model's capabilities and `tpl` from its architecture — does not survive the
data. Measured against the catalog on 2026-10-09:

- `medical-reasoning` maps to workload `general`, not `healthcare`
- `math` splits between `general` and `math`; `reasoning` is mostly `general`
- `chat` and `text-generation` appear against all eight workloads
- architecture `llama` has qualified under **eight** different prompt templates, `qwen2` under three

So a model's declared capabilities do not determine its workload and its architecture does not
determine its template. A script that picked anyway would be guessing with a script's authority,
which is worse than a person guessing, because nobody would look again.

### `dt` is not `--threads`

Two different knobs, and conflating them was this tool's first bug — found by running it against
real reports rather than by reading them:

| job field | becomes | recorded in the report as | fleet value |
| --- | --- | --- | --- |
| `dt` | `-Dmodels.native.kernels.decodeThreads` | `backendDiagnostics.environment.native-kernel-decode-threads` | 8 |
| *(not a job field)* | `--threads`, to **both** arms | `settings.threads` | 16 |

The first version read `settings.threads` and emitted `dt: 16`, silently undoing the decode-thread
setting the measured runs had actually used. `dt` is omitted entirely when the prior run used the
whole pool, so the job inherits the worker's default rather than pinning a number nobody pinned.

### Why a confirmation run reuses the settings

Re-measuring a model against a released build is only a comparison if everything except the build
is held fixed. Copying the recorded settings is what makes that true; an override is per field and
is marked `OVERRIDDEN` in the output so it cannot pass for a copied value.

Shard numbers are immutable — a number that has been launched names a payload and an S3 result
prefix — so the tool refuses to overwrite an existing shard file rather than reusing a number.

Tests: `python3 -m unittest discover -s scripts/fleet -p 'build_shards_test.py'` (stdlib only).

### Declaring a candidate that has never run here

A model with no prior report cannot have its settings copied, and `build-shards.py` refuses it by
default. To run a new candidate, declare **both** fields that decide whether the measurement means
anything:

```
python3 scripts/fleet/build-shards.py --ids slate.txt --reports <prior> \
    --catalog ../model-jars/catalog/models.json --out-dir /tmp --start-shard 650 \
    --wl-override new_model_id=summarization --tpl-override new_model_id=chatml
```

`--wl-override` and `--tpl-override` are required **together** for such a model. Declaring one and
letting the other fall back to a default would be the same guess the tool exists not to make, so
one without the other is still a refusal and the message says which is missing.

To inform that choice the refusal prints the precedent it found, strongest first:

- **same base model** — models whose `dimensions` are *identical*, which means the same base at the
  same quantization. This is the sharpest signal there is: `nexus-science` matches `nexus-legal` in
  every dimension, and legal is already qualified, so what legal ran with is the precedent that
  matters. Architecture alone would have pointed at the most common template across fifteen
  unrelated `qwen2` models.
- **same architecture, among models that have QUALIFIED** — read from the catalog's
  `qualifications.json`, which records the template that earned each verdict across every campaign.
  Only `qualified: true` rows count; what a *rejected* model ran with is not precedent for
  anything.
- **what was tried in the reports being read** — counted separately, because tried is not earned.

Declared jobs are printed as `DECLARED on the command line`, counted separately in the summary, and
noted as comparable only to a later run that holds the same declaration. `--mt-override` and
`--dt-override` are optional on top; without them the cap falls back to the worker default of 256
and no decode-thread pin is written.
## Workload coverage: `workload-coverage.py`

```
python3 scripts/fleet/workload-coverage.py --catalog ../model-jars/catalog \
    [--check-doc docs/WORKLOAD-COVERAGE.md] [--fail-on-zero]
```

Reports which RAG workloads have qualified models. The workload list is **parsed out of
`RagWorkload.java`**, not written in the script, so a workload the enum gains cannot go missing
from the report; a constant the parser cannot read is printed as a warning rather than skipped.

It keeps apart two states that both look like a zero:

- **no corpus** — the documents or cases resource is missing or empty, so no model *could* qualify
  there. That is "no data".
- **no qualified model** — the corpus has cases and nothing has passed the gate yet. That is a gap
  a campaign closes.

`--check-doc` verifies that `docs/WORKLOAD-COVERAGE.md` still states what this tool derives: the
pasted table, the quoted case questions, and the answerable/unanswerable split. Markdown
blockquote markers are stripped first, because a quoted question wraps across lines. `--fail-on-zero`
turns the report into a gate.

The candidate slate for the open gaps lives in `docs/WORKLOAD-COVERAGE.md`.
