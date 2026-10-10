# RAG workload coverage, and the candidate slate for the gaps

Regenerate the table below with:

```
python3 scripts/fleet/workload-coverage.py --catalog ../model-jars/catalog
```

The workload list is parsed out of `RagWorkload.java`, not written down here, so this report cannot
quietly omit a workload that the enum gains.

## Where coverage stands

```
workload        qualified   docs  cases  state
------------------------------------------------------
coding                  7     12      9  
finance                 1     12      9  thin: one model
general                51     12      9  
healthcare              1     12      9  thin: one model
legal                   2     12      9  
math                    1     12      9  thin: one model
multilingual            1     12      9  thin: one model
sql                     0     12      9  NO QUALIFIED MODEL (corpus is usable; this is a closable gap)
summarization           0     12      9  NO QUALIFIED MODEL (corpus is usable; this is a closable gap)
transportation          1     12      9  thin: one model
```

One row in `qualifications.json` is a **rejection**, not a qualification: the manifest header
counts `qualifiedModels` and `rejectedModels` separately and `entries` is their sum.
`h2o-danube3-500m` sits there on `general` with verdict `FAILED_MODEL_CONTRIBUTION_GATE`. It is
reported below the table and **not** counted as coverage — counting it inflated `general` from 51
to 52 and the catalog's qualified total from 101 to 102. A rejection is evidence; it is not
coverage.

Two states that both show as a zero are kept apart deliberately. **No corpus** means no model
*could* qualify there, which is "no data". **No qualified model** means the corpus has cases and
nothing has passed the gate yet, which is a gap a campaign closes. `sql` and `summarization` are
the second kind: both carry 12 documents and 9 cases, same shape as every workload that already has
qualified models.

For `summarization` the gap has a concrete consequence, stated in the enum itself: *"The router
cannot select a model for a summarization request until a qualification records that capability."*
Nothing is missing but a passing verdict.

## What these two workloads actually ask for

Read the corpora before picking candidates, because the workload names are misleading.

`sql` does **not** ask for SQL. Its cases ask natural-language questions *about a schema*, answered
from a one-sentence document, with the required facts cited:

> document — `Table orders has primary key order_id. Paid rows use status = 'PAID', and created_at
> stores UTC timestamps.`
> case — *"Which column is the orders primary key, and which status identifies paid rows?"*
> requiredFacts — `order_id`, `PAID`

`summarization` asks for grounded coverage of one source, and its **answerable** cases carry
**three** required facts each where the other workloads carry two:

> document — an incident postmortem paragraph
> case — *"Summarize Beacon incident 4417: how long were writes rejected, what triggered it, and
> how was it recovered?"*
> requiredFacts — `38 minutes`, `migration`, `rolling`

Every workload is 9 cases: **8 answerable plus 1 deliberately unanswerable**, which carries no
required facts and tests abstention instead. That ratio is worth knowing before reading any
result from this harness, because `extractiveFallbackRate = 0.889` is exactly 8/9 — it means the
extractive fallback fired on *every answerable case* and the model contributed nothing on any of
them. That is the number four models posted identically in the max-output-tokens sweep, and it is
what makes that null unambiguous rather than merely unflattering.

So both workloads test instruction-following, faithful extraction and citation discipline. Neither tests SQL
generation or abstractive summarization. This matters because the catalog already shows the same
trap in the other direction: the `medical-reasoning` capability maps to workload `general`, and
`huatuogpt-o1-7b` is QUALIFIED on `general` while **failing** on `healthcare`.

## The precedent worth copying

Every narrow domain was closed by a small domain-tuned model, not a large generalist:

| workload | model | size | template |
| --- | --- | --- | --- |
| finance | `king3djbl_nexus_finance_gguf_q4_k_m` | 0.99 GB | `chatml-no-think` |
| healthcare | `king3djbl_nexus_medical_gguf_q4_k_m` | 0.99 GB | `chatml-direct` |
| legal | `king3djbl_nexus_legal_gguf_q4_k_m` | 0.99 GB | `chatml-direct` |
| transportation | `umarfarookm_umartransit_1b_q4_k_m` | 0.99 GB | `chatml` |
| math | `qwen2_5_math_1_5b_instruct_q4_k_m` | 0.99 GB | `chatml-direct` |
| multilingual | `eurollm_1_7b_instruct_q4_k_m` | 1.05 GB | `chatml` |

Six domains, every one closed by a roughly 1 GB model. That is the shape to look for, and it
matches the founding principle that ModelJars optimises and qualifies smallest-first.

## The slate

Every architectural fact below was **range-fetched from the artifact's own GGUF header** with
`npm run catalog:triage -- <url>`, not read off a model card.

### Accepted candidates

| candidate | workload | size | license | arch | quant types | why |
| --- | --- | --- | --- | --- | --- | --- |
| `mradermacher/Mixture-Summarizer-Qwen3.5-2B-GGUF` Q4_K_M | summarization | 1.27 GB | **MIT** | `qwen35`, 320 tensors, 24 blocks, 2048 embed | Q4_K 79.6%, Q6_K 20.3%, F32 | a summarization specialist on an architecture with six qualified entries already |
| `King3Djbl/nexus-science-GGUF` q4_k_m | general | 0.99 GB | **Apache-2.0** | `qwen2`, 338 tensors, 28 blocks, 1536 embed | Q4_K 84.9%, Q6_K 15.1%, F32 | byte-for-byte the shape of three already-qualified siblings |
| `King3Djbl/nexus-security-GGUF` q4_k_m | general | 0.99 GB | **Apache-2.0** | `qwen2`, 338 tensors, 28 blocks, 1536 embed | Q4_K 84.9%, Q6_K 15.1%, F32 | same family, same shape; adds a security-assistant domain |
| already catalogued: `sqlcoder_7b_2_q5_k_m` | sql | 4.78 GB | CC-BY-SA-4.0 | `llama`, 16384 context | Q5_K 83.0%, Q6_K 17.0%, F32 | the only SQL specialist in the catalog; see the caveat below |

**No capability gaps.** All four use only Q4_K, Q5_K, Q6_K and F32 — every one of which the runtime
already serves. That is worth stating because four models were unloadable earlier on 2026-10-09
over three Q4_1 tensors each, and checking the headers first is what makes this claim cheap.

The `nexus-science` and `nexus-security` candidates run on `general`: the enum declares no science
or security workload, so they add catalog breadth rather than workload coverage. That is the honest
description of what they buy.

### Prediction, registered before the run

`sqlcoder-7b-2` is expected to **fail** the model-contribution gate on `sql`, for two reasons read
out of the file rather than guessed:

1. **It declares no chat template at all.** Zero `chat_template` keys in the header; `arch=llama`,
   `tokenizer.ggml.model=llama`. It is a base completion model whose real prompt format is defog's
   own `### Task / ### Database Schema / ### Answer`. Every `tpl` this harness offers wraps the
   prompt in a chat format this model was never trained on.
2. **It emits SQL.** Asked "which column is the orders primary key", a text-to-SQL model answers
   with a query. The grounding policy wants a cited natural-language statement containing
   `order_id` and `PAID`, so `modelAnswerRate` should collapse the way huatuogpt's did on
   healthcare.

If it fails, that is a publishable null and the argument for closing `sql` with a generalist or a
coder model instead. If it qualifies, the reasoning above was wrong and that is the more
interesting result. **Either way the prediction is recorded first**, so the outcome cannot be
reframed after the fact.

A generalist arm is therefore run alongside it on `sql`. The strongest prior is a coder model
already qualified on `coding` — schema questions sit closer to code than to prose — with
`qwen2_5_coder_1_5b_instruct_q4_0` (1.07 GB, `chatml`) as the pick, on the same
smallest-first reasoning as the table above.

### Rejected candidates, with reasons

| candidate | why not |
| --- | --- |
| `bartowski/Llama-Chat-Summary-3.2-3B-GGUF` | `creativeml-openrail-m`. OpenRAIL-M carries behavioural use restrictions; not publishable in a commercial catalog without a licensing decision that is not mine to make. |
| `RichardErkhov/abdulmannan-01_-_qwen-2.5-3b-finetuned-for-sql-generation-gguf` | **license unstated.** A marker cannot be published for an artifact with no license. |
| `mradermacher/natural-sql-7b-GGUF` | CC-BY-SA-4.0 ShareAlike, and it is the same class of text-to-SQL specialist as the SQLCoder already catalogued — it would test the same hypothesis twice at 7B. |
| `RichardErkhov/amang1802_-_Llama3.2-1B-summary-length-exp*` | research checkpoints from a length-ablation series, not a released model; several near-identical variants with no stated license. |

`King3Djbl/nexus-coder-GGUF` is deliberately **not** on the slate: `coding` already has seven
qualified models, so it would add the least of any option in that family.

## Running it

Build the shard files from recorded settings where a prior report exists, and state the choice
explicitly where one does not — `build-shards.py` refuses to guess `wl` or `tpl`, and these four
candidates have no prior report:

```
python3 scripts/fleet/build-shards.py --ids slate.txt --reports <prior> \
    --catalog ../model-jars/catalog/models.json --out-dir /tmp --start-shard 650 \
    --wl-override <id>=sql --wl-override <id>=summarization
```

Nothing here can land in the catalog until it is measured against a **released** build. The
verdicts from the 2026-10-09 campaign carry `models@0.3.56-dev`, and a dev build is not publishable
evidence.
