# Granite 4.1 3B activated-adapter hybrid qualification preflight

Frozen on 2026-09-15 before any adapter output was generated. Every threshold below is fixed;
a failing gate rejects the specific adapter, not the family, and no threshold is retuned after a
result is known.

## Candidate

- **Hybrid shape:** one Granite 4.1 3B base that owns chat and native tool calling, plus IBM's
  upstream activated-LoRA specialists sharing the base's immutable KV prefix by physical array
  identity. The specialists activate at the assistant marker, so the whole system, documents, and
  conversation prefix executes with base weights and is reusable by every member.
- **First component:** `answerability` (rank 16, alpha 32, seven projections, 124,593,064 bytes).
  Its output is one of two labels within six tokens, which makes its task gate cheap and exact.
- **Second and third components under the same protocol, each with its own frozen screen:**
  `query_clarification` (rank 32, alpha 64) and `query_rewrite` (rank 32, alpha 32). Neither
  is evaluated until the answerability component has passed or failed every gate.
- **Why this candidate and not the Qwen3 1.7B tool specialist:** twenty-two trained Qwen
  adapters (V1 to V22) proved the mechanics and failed the call/no-call quality floor. These
  adapters need no training and their publisher reports task accuracy; that report is *read*,
  not measured, and carries no weight here.

## Pinned identities

| Item | Identity |
|---|---|
| Base GGUF | `ibm-granite/granite-4.1-3b-GGUF` revision `ab4701481089b58a082ef63cc1cee738887293ff`, `granite-4.1-3b-Q4_K_M.gguf`, 2,099,501,664 bytes, SHA-256 `662b0626cd58f443baea23559b469df6576a81d349649c59413b36a9fb32eb29` (already the ModelJars catalog artifact) |
| Base tokenizer and chat template | `ibm-granite/granite-4.1-3b` revision `c0650403e44e78ec0262dab1c90914c65b196c4e`; `tokenizer.json` SHA-256 `e2bad66439538cb4d5a7580680932432ed9ece9d3b8577e675512bdf11599253`, `chat_template.jinja` SHA-256 `fed2756d2d24e127b951dcf139d0b03ab7db8ef23a456128ebc9c2db4901d476` |
| Adapter library | `ibm-granite/granitelib-rag-r1.0` revision `2f0b2c79c6731068625aca8045c2eb2e8912b353`, Apache-2.0 |
| `answerability` weights | SHA-256 `765e85650d39b89e055634c8da0c2d469c3bfa436df271df17f516ad7466d6ca`; `adapter_config.json` SHA-256 `79409b5fc9702a59d89b55d827cb9fa5d1c41fe1af01ddafceae4349ae227698`; card `answerability/README.md` SHA-256 `010d3765af42b4c69a906bca1762b236b1db9e1f13592c4c7b78dc96db8381b8` |
| `query_clarification` weights | SHA-256 `e1a8bdf9dda6667e745847b492ac47e46a2abc54f57035e11d5a4a0f66c82157` |
| `query_rewrite` weights | SHA-256 `fa6a30d7518ee4f1e2c00fd62e29662d7c217d15b90bbc5e213657b9e08a236c` |
| Invocation | `<|start_of_role|>assistant<|end_of_role|>` = tokens `[100264, 78191, 100265]`, declared by every adapter as `alora_invocation_tokens` |
| Runtime | Models `exp/granite-alora-first-candidate` on top of `feat/alora-cache-sharing`; the exact revision is recorded with each result |
| Oracle | llama.cpp `b9960-a935fbffe`, benchmark use only, never a runtime dependency |

The adapters declare seven target modules including MLP projections. That is compatible with the
Java loader, and it means nothing after the marker is reusable by the base branch; only the
pre-marker prefix is shared, exactly as the runtime already enforces.

## Predeclared gates

1. **Java conformance of the base.** `PureJavaBackend` loads the pinned GGUF as family `granite`;
   `encode("The quick brown fox")` is `[791, 4062, 14198, 39935]` with no BOS; the eight greedy
   continuation IDs are `[35308, 927, 279, 16053, 5679, 1210, 578, 734]`, captured from the
   oracle on 2026-09-15 before this file was frozen; `encodeControl` of the marker is
   `[100264, 78191, 100265]` and decodes back to the marker text. Any difference fails.
2. **Adapter identity.** The fail-closed packager pins base artifact, base revision, tokenizer
   file hashes, adapter hash, rank, alpha, target modules, marker text and tokens, publisher
   revision, card and config hashes. The Java loader must reject an altered marker, base, or
   weight hash, and the Java tokenizer must encode the marker text to the declared tokens.
3. **Mechanics with real weights.** The base and activated branches reference the same physical
   immutable prefix arrays; the shared and independently recomputed branches produce identical
   token IDs at 256, 1,024, and 4,096 prefix tokens; a loaded but unactivated adapter is a true
   no-op (greedy IDs identical to gate 1 and to the plain base on the chat oracle prompt);
   activation-boundary rewind and rerender keep the same blocks.
4. **Task quality, answerability, two public datasets, both required.** The slices are frozen by
   `prepare_answerability_suites.py` before any adapter output and bound by SHA-256 in
   `qualification-window.json`. Selection is stratified by label because MT-RAG's human turns are
   709 answerable to 55 unanswerable, and an unstratified slice would let "always answerable"
   pass a plain accuracy floor; that correction was made before any output existed.
   - MT-RAG human generation tasks (IBM/mt-rag-benchmark at `2c618bb98db3c8526433e22d8a2f7320f10a7470`,
     `mtrag-human/generation_tasks/RAG.jsonl`, Apache-2.0): every `UNANSWERABLE` task (55) plus
     55 `ANSWERABLE` tasks selected by the lowest unsigned SHA-256 of `20260915:<task_id>`;
     `PARTIAL` and `CONVERSATIONAL` are excluded; the task's retrieved `contexts` are the
     documents, in file order, and the `input` turns are the conversation.
   - SQuAD 2.0 dev (`dev-v2.0.json`, SHA-256
     `80a5225e94905956a6446d296ca1093975c4d3b3260f1d6c8f68bc2ab77182d8`, CC-BY-SA-4.0, outside
     the adapter's training domain): 100 impossible and 100 possible questions selected by the
     same rule over question ids; the paragraph is the single document; label is `answerable`
     when `is_impossible` is false.
   Thresholds: every output must be exactly `answerable` or `unanswerable` (100%, structure
   gate); balanced accuracy at least 0.80 on each dataset; on each dataset no worse than the
   unadapted base asked the same question through the same documents prompt with the instruction
   "Answer with exactly one word: answerable or unanswerable"; every case physically shares its
   prefix. The prompt bytes and token IDs of the first case must match the published Jinja
   template rendered by Transformers before any score is read, as V12 required.
5. **Long context.** Eight conversations whose document block exceeds 4,096 tokens: the specialist
   answers at least 6 of 8 correctly with physical sharing, and the base branch, continued from
   the same blocks after the specialist turn, answers an early-fact probe byte-identically to a
   native base run, retaining every native-correct answer.
6. **Performance.** Shared versus recomputed handoff at 256, 1,024, and 4,096 prefix tokens with
   token-exact outputs; sharing must improve 4,096-token handoff TTFT by at least 20%; unique
   inference-state bytes, JVM memory, native-memory tracking, and process RSS are reported for
   every arm, including losing ones. The pure-Java arm is the reference for every tier; a
   Models-owned Rust/FFM kernel arm is measured beside it, keeps the Java fallback, and must be
   token-identical to the Java arm at every tier.
   **Amendment, 2026-09-15, before any gate 4 or 5 result existed:** gates 4 and 5 may execute on
   the Rust/FFM kernel arm because the workstation's pure-Java prefill would put the 310-case
   window at tens of hours. The Rust arm is admitted for those gates only after its outputs are
   token-identical to the pure-Java arm on the first ten cases of each suite in both arms, with
   that comparison retained beside the reports; any divergence returns those gates to pure Java.
   The kernel is a matrix-product shim under the unchanged Java transformer, adapter delta, and
   cache code, and it is never an external inference runtime.
7. **Packaging.** ModelJars two-stage publication: the hidden component marker binds the complete
   file bundle, the report at an immutable Models revision, and the measured crossover; the
   visible composition is published only after a clean Java 25 host resolves the released Models
   artifacts and both markers from Maven Central and repeats gates 3, 5, and 6.

## Host plan

- Gates 1 to 3 run on the operator workstation (Intel i7-9750H, 32 GiB) because a 3B Q4_K_M
  base is small enough; the exact host is recorded with each result.
- Gates 4 to 6 run on one bounded cloud host with a spend ceiling, SSH restricted to the
  operator's `/32`, and deletion after evidence copy, following the 2026-09-14 pattern.
- Python (Transformers, PEFT) is an oracle for prompt bytes and for a reference label run; it is
  never in the product execution path.

## Execution record

- 2026-09-15: base GGUF, tokenizer files, and three adapter directories downloaded and hashed;
  every hash matches the Hugging Face LFS identity listed above. Oracle tokens captured. The
  generalized packager and its nine unit tests were written before any adapter was packaged.
- 2026-09-15, gate 1 and gate 2 boundary, operator workstation (Intel i7-9750H, macOS, Temurin
  25.0.3, Panama enabled): `./gradlew :backend-java:granite41AloraIntegrationTest` with the pinned
  GGUF and the packaged answerability adapter. Both tests passed: family `granite`, prompt tokens
  `[791, 4062, 14198, 39935]`, eight greedy IDs identical to the oracle, marker
  `[100264, 78191, 100265]` round-trips, and the adapter opened at the marker with
  `physicallySharesPrefix=true` over a positive shared prefix. Wall time 12.3 s for the base
  conformance test and 100.0 s for the adapter test including model and adapter load.
- 2026-09-15: qualification window frozen by `prepare_answerability_suites.py` (six unit tests
  pass): `mtrag-human-rag` 55 answerable + 55 unanswerable of 709/55 eligible;
  `squad-v2-dev` 100 + 100 of 5,928/5,945 eligible. `RAG.jsonl` SHA-256
  `5d5201da9fabd072fd8f6b8d051bfaafa7ef031e76722a4920c66e94cede1873` at revision
  `2c618bb98db3c8526433e22d8a2f7320f10a7470` (re-fetched at that revision and identical);
  window content SHA-256 `1d88443775ac8c32dd52239a43b6deae5cd3793eb639ad9e21193196335e9d67`,
  file SHA-256 `2c98968950d192099605b13e6ed73db86281bfc29d455886af7d5d8ec3e19781`. No adapter
  output existed when the window was frozen.
- 2026-09-15, prompt-byte oracle, first MT-RAG case, specialist arm: the Java runner's rendered
  bytes matched the Transformers rendering exactly (SHA-256
  `9f8b4aa1e76baa715c1a975156b0b954a5117aabb94948cc29498dd2f85e85d0`), but the token IDs
  diverged at position 37 (Java 1,604 tokens, oracle 1,601). Cause: Granite 4.1 declares
  `<documents>` and `</documents>` as special tokens (ids 100282 and 100283), Transformers and
  llama.cpp with special parsing emit them as single IDs, and the runner had placed them inside a
  text segment. Correction, made before any scored output: the two template markers are rendered
  as control segments while every document body and message stays a text segment, which is also
  the stricter injection boundary. The frozen window contains no literal special-token string in
  any message or document, so the corrected rendering is expected to be byte- and token-identical
  to the oracle for every case; that is verified over all 310 cases and both arms below rather than
  on the first case only. The one-case inference smoke that ran under the old rendering was
  stopped and is not evidence.
- 2026-09-15: Rust/FFM activated loaders added (`PureJavaBackend.loadActivatedAdapter` with an
  injected kernel, `RustFfmBackend.loadActivatedAdapter`) and the runner gained `--backend`; the
  macOS x86_64 kernel library built locally with SHA-256
  `53a8233ee8202b085f72de41552868cecc8b55fe634a29529a455c7548b741e3`. No Rust result exists yet.
- 2026-09-15, full-window prompt oracle after the marker fix: only 2 of 110 MT-RAG and 135 of 200
  SQuAD prompts were token-identical. Decoded spans showed two classes: digits split as
  `18|50` instead of `185|0`, and ` \"I` merged across the space. Cause: the GGUF declares
  `tokenizer.ggml.pre = dbrx`, a name absent from the Java pre-tokenizer table, whose fallback
  is no pre-tokenization at all. The published tokenizer.json carries exactly the Llama-3 split
  regex with `ignore_merges` false. Correction: `dbrx` now maps to that pattern beside
  `smaug-bpe`, with a unit test on `1850s "Islamist" won 75% in 2024`. Gate 1 had passed because
  its oracle prompt held no digits or punctuation; a second oracle prompt with digits, quotes, a
  percent sign, and a thousands separator (23 tokens, eight greedy IDs from llama.cpp b9960) is
  now part of gate 1. This defect predates the candidate and affected every Granite 4.x prompt on
  the Java path.
- 2026-09-15, third class after the dbrx mapping: 13 MT-RAG prompts still differed on no-break
  spaces, because Java's `\s` is ASCII-only unless `UNICODE_CHARACTER_CLASS` is set while the
  published Rust regex and llama.cpp treat U+00A0 as whitespace. Every BPE pre-tokenizer pattern
  now compiles with that flag; a unit test covers `risk  at   home`. The full
  backend-java unit suite passed afterwards (610 tests, 0 failures).
- 2026-09-15, prompt-byte oracle, complete: with the marker, dbrx, and whitespace corrections the
  Java runner is byte- and token-identical to Transformers 4.57.1 for all 110 MT-RAG and all 200
  SQuAD 2.0 cases in both arms (620 prompts). The machine-readable comparisons are in
  `prompt-oracle/`. Gate 4's prompt precondition is therefore satisfied for every case, not only
  the first.
- 2026-09-15, runner contract correction: the first two SQuAD cases in hash order were run on the
  Rust arm as a smoke while the unit suite was executing. Both completions were correct JSON
  string literals (`"unanswerable"` and `"answerable"`), which is exactly what the adapter's
  published `io.yaml` declares (`response_format: {"type": "string", "enum": [...]}`). The
  runner had demanded bare words, so it scored both as unstructured. The structure check now
  parses the completion as a JSON string equal to a label, per the published contract. This
  changes how outputs are read, not any threshold; the two exposed cases stay in the frozen
  window because excluding them would bias it. The Rust-arm timings from that smoke are
  discarded because the host was shared with the unit suite.
- 2026-09-15, Rust/FFM versus pure-Java identity smoke, first two SQuAD cases in hash order,
  specialist arm, operator workstation: both arms produced the identical completions
  (`"unanswerable"` then `"answerable"`, both correct), the identical shared-prefix counts (269 and
  386 tokens), and `physicallySharesPrefix=true` on every case. Clean wall times were 220.8 s and
  343.7 s on the Rust arm against 251.9 s and 343.4 s on pure Java. Two cases are not the ten per
  suite the amendment requires; the amendment stays unsatisfied.
- 2026-09-15, why both arms are this slow: the shared-prefix path prepares the base prefix through
  the session prefill, which walks the prompt one token at a time, so neither arm batches the
  prefill and the native matrix kernel has nothing to accelerate. The plain sessionless path on
  the same host measured 2.08 prefill tokens per second batched (132-token prompt, pure Java)
  against about 1.1 on the session path. A session-aware batched prefill is therefore the next
  engineering item before gate 4 can run at any useful cost; it must stay token-identical to the
  per-token path, and the Granite 4.1 integration test's greedy oracle is the check for that.
- 2026-09-16, batched session prefill (commit `fce92d1`): the single-session prefill now runs
  through the ragged session-batch path. Identity evidence: nano backend test, the Granite 4.1
  integration greedy check after a batched prefill (matches llama.cpp), and the full backend-java
  suite (611 tests). Two-case SQuAD smoke, specialist arm, identical outputs on both backends:
  Rust 30.0 s and 34.2 s per case (was 220.8 s and 343.7 s), pure Java 133.0 s and 186.8 s (was
  251.9 s and 343.4 s). The turn's own prefill and decode are seconds; the wall time is the
  shared-prefix preparation, now batched.
- 2026-09-16, gate 4 host plan (written before creation): Hetzner `cpx62`, `fsn1`, 16 shared
  x86-64 vCPU, 32 GiB RAM, Ubuntu 24.04, live gross price EUR 0.2452 per hour (the API refused
  `cpx51` in `fsn1` as an unsupported location for that type); one host, ceiling 24 hours
  (EUR 5.89) with `delete-by` 2026-09-17T04-00Z; provider firewall permitting
  TCP/22 only from the operator workstation's observed /32; SSH key `vectors-bench-v2`
  (id 114663461, MD5 60:4e:57:38:96:48:c5:45:73:10:9d:37:f7:4f:74:cf). The host runs
  `host-run.sh` at Models commit `fce92d106a77c34fd4aad4320d2876e9745c2b14`: bootstrap,
  hash-checked artifacts, the ten-case identity precondition on both backends for both suites
  and arms, then the full 310-case window on the Rust arm only if identity passes. An unrelated
  server from another campaign (`modeljars-bench-cantor-cr-20260915-b`) was present in the
  account at creation time and is not touched by this run.
- 2026-09-16T03:15:53Z: Hetzner server `166122805` (`modeljars-granite41-hybrid-20260916`,
  `cpx62`, `fsn1`, IPv4 `167.233.210.176`) created with provider firewall `11630670` applied
  (TCP/22 from the operator /32 only); host key ED25519
  `SHA256:R3AvbuQNuiOSTMItK6ms3t9MeTrocBNVKr2640Ug7qI` recorded in this directory's `known_hosts`.
  Baseline: 16 vCPU, 30 GiB free, 575 GB free disk, no stray inference processes. `host-run.sh all`
  started under nohup; a local watchdog deletes exactly this server and firewall at the deadline
  if they still exist.
- 2026-09-16, publication prerequisites identified while the host runs (not blockers for the
  gates, but for the release): ModelJars downloads a multi-file bundle from one pinned
  `/resolve/<revision>/` base and, for `hf://` sources, only from huggingface.co, so the packaged
  adapter bundle (weights, `models-activated-lora.json`, NOTICE, LICENSE) needs a Hugging Face
  repository under our control at a pinned revision before a component marker can be built. The
  ModelJars branch cannot compile until a Models release carries the activated runtime, so the
  chain is: gates pass here, draft PR #180 (which contains #176) merges, Models releases to Maven
  Central, then the ModelJars component and composition entries publish. The ModelJars component
  gate now accepts `specialistKind = upstream-rag-specialist` with the answerability evidence
  shape, and `assemble_component_report.py` produces that report from the raw host evidence.
- 2026-09-16, base public qualification plan (written before creation): ModelJars refuses a
  component whose base is not itself publicly qualified, and Granite 4.1 3B has no public
  qualification, so the base runs the controlled RAG qualification harness
  (`scripts/run-controlled-rag-qualification.sh`, policy `production-rag-model-contribution-v6`,
  workload `general`, prompt template `granite`, Rust/FFM backend, llama.cpp b10012 and Ollama
  v0.32.0 as native controls only) on a second bounded Hetzner `cpx62` in `fsn1` at EUR 0.2452
  per hour, ceiling 12 hours (EUR 2.95), `delete-by` 2026-09-16T18-00Z, provider firewall
  admitting TCP/22 from the operator /32 only, Models commit `4ce4fd1` (the 0.3.38 release
  preparation on the candidate branch). The first host keeps running gate 4 undisturbed.

**Base run, attempt 1 (2026-09-16T04:13Z):** the harness rejected `--prompt-template granite` before
the first case because `RagPromptTemplate` had no Granite envelope; the ModelJars catalog names the
template but the RAG bench had never rendered it. Added `GRANITE` (single-token role markers, text
passed through, `<|end_of_text|>\n` after every turn, same bytes as `ChatTemplate.GRANITE`) with a
renderer test. Attempt 2 runs at the commit that carries it; the base entry's `backendVersion` pins
that commit, not 4ce4fd1.

**Base run, attempt 2 (2026-09-16T04:21Z, Models 5a0e376, `--prompt-template granite`):** the
library-default correctness smoke FAILED on quality: 9/9 attempts generated, 8/9 correct,
abstention accuracy 1.0. The miss is `auto-glass-deadline`: retrieval recall 1.0 (the single
retrieved document contains both required facts) and the model replied `INSUFFICIENT_CONTEXT`. A
llama.cpp b10012 control on the identical raw prompts (scratch run, same host, not evidence)
reproduced the abstention on the same case and matched 8/9 elsewhere, so this is the model's
behaviour under the generic CONTEXT/QUESTION/ANSWER envelope, not a Java-path defect. Probe on the
failing case through llama.cpp with prompts rendered by the Transformers oracle: generic envelope
→ `INSUFFICIENT_CONTEXT`; Granite documents envelope with the harness instructions leading the
system turn → `30 calendar days, 75 dollars [claims-auto-glass]`; documents envelope without the
instructions → correct facts, no citation. Decision, made before the next run: add
`granite-documents` as a declared template (evidence in the trained `<documents>` block, bare
question as the user turn, instructions unchanged), pin its bytes to the oracle in a unit test,
and run attempt 3 under it. The generic-envelope failure stays on record; the ModelJars base entry
names `promptTemplate: granite-documents`. Also measured in attempt 2: Rust arm default-smoke
decode 5.7 tok/s and 38 tok/s prefill on 16 threads versus 54 tok/s decode on the llama.cpp
control; the performance phase enables `models.native.quantizedDecode` itself, but the base may
still miss the 0.8 decode-throughput ratio. That is a measurement to take, not a reason to skip it.

**Base run, attempt 3 (2026-09-16T04:34Z, Models bc1e978, `--prompt-template granite-documents`):**
every correctness gate passes and the performance gates fail. Library-default smoke 9/9 attempts,
correct 1.0, abstention 1.0. Performance phase (quantized native decode on, 27 attempts): correct
1.0, abstention 1.0, model answer rate 0.78, model answer correct rate 1.0. Rust arm p50 decode
11.7 tok/s, p50 prefill 37 tok/s, p95 TTFT 3672 ms, p95 end-to-end 6350 ms → absolute tier
OFFLINE (USABLE needs p95 TTFT ≤ 2000 ms). Controls on the same host and prompts: Ollama v0.32.0
48.1 tok/s decode, 504 tok/s prefill, p95 TTFT 776 ms (PRODUCTION_READY); llama.cpp b10012 48.3
tok/s decode, 241 tok/s prefill, p95 TTFT 1175 ms (USABLE). Verdict FAILED_ABSOLUTE_GATE, no
qualifying comparator (decode ratio 0.24 against both). Bundle copied to
`host-evidence/base-attempt3/` (measured, not publishable).

Is it the host or the model? Qwen2.5 3B Instruct Q4_K_M under the same harness settings on this
host: 16.7 tok/s decode, 63.5 tok/s prefill, p95 TTFT 1266 ms (its certified run on an AWS EPYC
9R14 host gave 29.4 / 83.3 / 1085 ms, so this shared-vCPU host is roughly 1.75× slower on decode).
Granite on our path is 0.70× Qwen on decode and 0.58× on prefill at near-identical parameter count
(40 × 2560 / 8192 FFN / 40 heads of 64 / 8 KV heads / 100k vocab, tied Q6_K embedding, versus
36 × 2048 / 11008 / 16 heads of 128 / 2 KV heads / 152k vocab), while Ollama runs Granite faster
than it ran Qwen. The Java path therefore carries a Granite-specific cost of roughly 3× against
llama.cpp that is not present for Qwen. Next: JFR execution samples of the Rust arm for both models
on this host to locate the cost before any kernel work. No base qualification can be claimed until
the tier and comparator gates pass on a run recorded in this file.

**Rust admission precondition, MT-RAG (host 1, Models 4ce4fd1, 2026-09-16T04:54Z):** NOT identical.
On the first ten MT-RAG cases the pure-Java arm and the Rust FFM arm agree on 9 and differ on
`f1121a39…<::>2` (label answerable; Java says unanswerable, Rust says answerable). Every other field
matches (ids, order, structure, sharing). Per-case wall time: Java 144–685 s, Rust 62–445 s
(`host-evidence/identity10-mtrag-human-rag-specialist-{pure-java,rust-ffm}.json`). The flip is a
borderline decision moving with float accumulation order; the answerability head reads one
next-token distribution, so any last-ulp difference at the boundary changes the label. Under the
frozen amendment the Rust arm is therefore not admitted for gates 4–5 on the strength of this
screen. Decision, taken before any window result is read: the window runs on BOTH arms at one
frozen Models commit — pure Java as the reference the amendment names, Rust FFM as the backend that
ModelJars would actually ship (every base entry in the catalog is rust-ffm) — and the report states
the per-case agreement rate between the arms alongside each arm's balanced accuracy. Qualification
is decided on the pure-Java arm per the frozen rule; the Rust arm is published only if it also
clears every threshold. The frozen commit must be one at which the base performance gate passes,
because ModelJars requires a single `modelsRevision` across the base, component, and composition
evidence, so the window is not started until that commit exists.

**Where the base performance gap comes from (host 2 scratch runs, 2026-09-16T04:45–05:08Z, all
measured, none publishable):**
- Vectorized swiGlu and residual FMA for Granite (Models be61bd8): decode 11.7 → 11.7 tok/s,
  prefill 37 → 39, p95 TTFT 3672 → 3258 ms. The JFR Java-side percentages were real but the wall
  clock is elsewhere; kept because the oracle tests pass and it removes a scalar loop.
- `models.purejava.batchedAttentionScores/Values=true` (exact two-row scoring, four-row FMA value
  accumulation, same float order as the per-row path): decode 13.8, prefill 48, p95 TTFT 2877 ms.
- Threads 8 instead of 16 on this 16-vCPU shared host: decode 16.0, prefill 34; threads 4: 12.6 / 27.
- Qwen2.5 3B Instruct Q4_K_M on this host, same harness: ours 16.7 tok/s decode and 63.5 prefill;
  llama.cpp b10012 64.0 and 298. Ratio 0.26, the same as Granite's 0.24–0.33. Its certified run
  (AWS EPYC 9R14, 16 dedicated vCPU) had ratio 0.88 against Ollama.
Conclusion: on this shared-vCPU cpx62 the Java runtime trails llama.cpp by roughly 4× for both
models, so the gap is host-shaped, not Granite-shaped; Granite is not an outlier on our path.
The qualification protocol compares candidate and controls on the same host, so the host class
decides the verdict. Next: repeat the base qualification on a dedicated-vCPU host of the class the
certified entries used, with the two batched-attention properties passed as recorded tuning for
the performance phase (the library-default smoke stays untuned), and read the ratio there before
any kernel work is considered.


**Base run, attempt 4 plan (written before the host exists):** dedicated-vCPU host of the class the
certified entries used: Hetzner `ccx43` (16 dedicated AMD vCPU, 64 GiB), Ubuntu 24.04 x86-64,
`fsn1`, provider firewall reused from host 2 (TCP/22 from the operator /32 only), SSH key
`vectors-bench-v2`, label `delete-by` 2026-09-16T20-00Z with a local watchdog. Runs
`base-host-run.sh` at Models `be61bd8` with `RAG_TUNED_JAVA_OPTS` =
`-Dmodels.purejava.batchedAttentionScores=true -Dmodels.purejava.batchedAttentionValues=true` for
the performance phase only; the library-default smoke stays untuned as the harness enforces. The
verdict is read from `qualification.json` on that host and copied into `host-evidence/` whatever
it says.
Substitution at creation time: the account's dedicated-core limit refused `ccx43`
(`resource_limit_exceeded`), so the host is `ccx33` (8 dedicated AMD vCPU, 32 GiB), Hetzner id
166128429, host key SHA256:aJRKPYLpCoIqW18+h6r2Gt12RGjYP4/GmKRKtAtlebE (`known_hosts_ded`). The
certified Qwen entry ran on 16 processors; the ratio gates compare candidate and controls on the
same host, so the core count changes absolute numbers, not the comparison.

**Base run, attempt 4 result (host 3, ccx33 8 dedicated Milan vCPU, Models be61bd8,
`granite-documents`, tuned batched attention):** FAILED_ABSOLUTE_GATE, every arm OFFLINE.
Correctness: smoke 9/9 correct 1.0; performance phase 27/27 correct 1.0, abstention 1.0, model
answer rate 0.78. Rust arm p50 decode 12.1 tok/s, p95 TTFT 3579 ms, p95 e2e 6218 ms. Ollama
13.4 tok/s, p95 TTFT 2209 ms, OFFLINE. llama.cpp 14.8 tok/s, p95 TTFT 4235 ms, OFFLINE. So on this
host the Java runtime is at parity with both native controls (decode 0.91× Ollama, 0.82×
llama.cpp; TTFT better than llama.cpp), and the absolute tier fails for all three because the host
cannot serve a 3B model under 2 s TTFT. Bundle in `host-evidence/base-attempt4/`.

Reading attempts 3 and 4 together: llama.cpp is 3.2× faster on the 16-vCPU Genoa host than on the
8-core Milan host, while our runtime is essentially unchanged between them (11.7–13.8 vs 12.1
tok/s). Our decode is bound by per-token latency that extra cores, AVX-512, and memory bandwidth
do not shorten: roughly 280 native projection dispatches per token plus single-threaded Java
attention over 40 heads. The batched prefill kernel likewise streams each weight block once per
batch row group and does not tile across the batch the way a GEMM does, which is what the 37–48
tok/s prefill against llama.cpp's 240 says. These are runtime engineering items, not shims: (1)
parallel per-head attention on the existing worker pool, (2) one fused native dispatch per layer
for single-token decode, (3) batch-tiled K-quant GEMM for prefill. Until at least (1) and (2)
land, the base cannot clear the comparator gate on a host where the controls clear the absolute
gate, and ModelJars refuses the component without a qualified base.


**Base run, attempt 5 plan (written before the instance exists):** the host class every certified
RAG entry used — AWS `c7a.4xlarge` (16 vCPU AMD EPYC 9R14, 32 GiB), us-east-1, Ubuntu 24.04
(`ami-025d99823a4caad37`), on-demand, 80 GiB gp3 root, existing key pair
`modeljars-qualification-20260830` (matches the operator's ed25519 key) and security group
`modeljars-qual-20260830` (TCP/22 from the operator /32 only), tag `delete-by` 2026-09-16T12-00Z
with a local watchdog. Same `base-host-run.sh` at Models `be61bd8`, `granite-documents`, and the
recorded tuning the certified Qwen2.5 3B entry used: `RAG_NATIVE_THREADS=8` plus
`models.purejava.batchedAttentionScores/Values=true` for the performance phase; the smoke stays
untuned. Purpose: decide whether the base qualifies under the existing protocol on the protocol's
own host class before any runtime work is scheduled. The verdict is copied whatever it says.
Host 2 scratch, certified-profile settings (`models.native.kernels.threads=8`, batched attention
scores/values, Models be61bd8): decode 18.9 tok/s, prefill 41, p95 TTFT 3254 ms, p95 e2e 4620 ms
(native threads 4: 14.1 / 33 / 3950). That is +62% decode over the harness default of 16 native
workers on this 16-vCPU shared host, still 0.39× the Ollama control there; TTFT stays prefill-bound.

**Base run, attempt 5 result (AWS c7a.4xlarge, EPYC 9R14 16 vCPU, Models be61bd8,
`granite-documents`, `RAG_NATIVE_THREADS=8` + batched attention):** FAILED_ABSOLUTE_GATE. Smoke 9/9
correct 1.0 (untuned, 9.7 tok/s). Performance phase 27/27 correct 1.0, abstention 1.0. Rust arm
p50 decode 20.9 tok/s, p95 TTFT 2310 ms, p95 e2e 3810 ms → OFFLINE by 310 ms of TTFT (USABLE ≤
2000). Ollama 28.4 tok/s, p95 TTFT 635 ms, PRODUCTION_READY; llama.cpp 29.0 tok/s, p95 TTFT 992
ms, PRODUCTION_READY. Decode ratio 0.74 (Ollama) / 0.72 (llama.cpp) against the 0.8 floor; e2e
ratio 2.3 against the 1.5 ceiling. Bundle in `host-evidence/base-attempt5/`. On the protocol's
own host class the base is within reach: the gates need roughly −15% TTFT, +10% decode, and −35%
end-to-end, and the runtime items named after attempt 4 (parallel attention first) are the path.


**Runner defect found by the identity screen, base arm (host 1, 2026-09-16T05:52Z):** the base arm
answered the bare word `unanswerable` on all ten MT-RAG cases and the runner scored every case
unstructured, because `prediction()` applied the specialist's JSON-literal contract to both arms.
The base arm's own instruction asks for exactly one word, so its contract is the bare label. Fixed
in `ActivatedAnswerabilityQualificationCli.prediction(output, arm)` with a test; the specialist
contract is unchanged. The ten base outputs are kept in
`host-evidence/identity10-mtrag-human-rag-base-pure-java.json` (all ten said `unanswerable` for
answerable cases, so the base's MT-RAG accuracy on this slice is 0/10 under either reading). Every
identity and window arm is re-run at the frozen commit, so this changes no recorded number.


**Attention partitioning, measured on the AWS host (scratch, certified profile):** Models 10d0caf
(heads always partitioned): prefill 59 → 94 tok/s, p95 TTFT 2310 → 1439 ms, decode 20.9 → 17.9
tok/s. Models 8b1e15b (heads serial below 1024 positions, one KV view per row): decode 21.3,
prefill 90, p95 TTFT 1501 ms, p95 e2e 3103 ms, tier USABLE. Against attempt 5's controls: decode
0.75× Ollama (floor 0.8), e2e 1.89× (ceiling 1.5). Host 2 with heads always partitioned: decode
18.9 → 10.9, prefill 41 → 59 — the same shape, which fixed the threshold. Next scratch: native
worker count {8, 12, 16} × `vectors.maxBits` {256, 512}; the runtime caps Panama at 256 bits on
this AVX-512 host by default and the attention arithmetic runs on the Java side.

**AWS tuning sweep (scratch, Models 8b1e15b, one iteration, 27 requests):**

| native workers | vectors.maxBits | decode tok/s | prefill tok/s | p95 TTFT ms | p95 e2e ms |
| --- | --- | ---: | ---: | ---: | ---: |
| 8 | 256 | 21.3 | 88 | 1469 | 2934 |
| 8 | 512 | 20.4 | 89 | 1434 | 3023 |
| 12 | 256 | 20.4 | 117 | 1098 | 2673 |
| 12 | 512 | 19.7 | 125 | 1037 | 2715 |
| 16 | 512 | 19.1 | 135 | 980 | 2703 |

Prefill scales with native workers (TTFT reaches the PRODUCTION_READY bound at 16); decode does
not move above ~21 tok/s under any setting, and 512-bit Panama lanes change nothing. Against
attempt 5's controls the best configuration gives decode 0.72–0.75 (floor 0.8) and e2e 1.63–1.79
(ceiling 1.5). The per-token floor of ~47 ms is the target; next measurement is a main-thread
Java-versus-native split under JFR to attribute it before any kernel is written.

**Main-thread split under JFR (AWS, Models 8b1e15b, certified profile, one iteration):** 1644
native-method samples (the thread inside the Rust matmul downcalls) against 820 Java samples. Of
the Java samples 480 are attention (`matVecDotExact` 220, `addWeightedRowsInPlace` 130, `softmax`
97, single dots 30), 90 are the harness hashing the artifact, 54 swiGlu, the rest norms, rope, and
dispatch glue. Per ~47 ms token that is roughly 31 ms native, 9 ms Java attention, 5 ms other Java.
llama.cpp's whole token is ~34 ms on this host, so the native projections are at parity; the
deficit is the Java-side attention and glue. Closing it needs (a) a grouped-query fused attention
kernel that reads each K and V row once for the five query heads that share it (a vectors-core
addition — backend-java has no direct Panama code and pins vectors-core 0.1.7), and (b) per-phase
native worker counts (16 for prefill, 8 for decode; today one count serves both, and the sweep
shows the two phases want different values), which is a native ABI change. Both are runtime
engineering across two projects, outside what this campaign can land as a shim. The base stands at
USABLE with decode 0.75× and e2e 1.63–1.89× the controls on the certified host class.

**State at hand-off (2026-09-16T06:10Z):** Models candidate branch at `daa2e12` (+ this note) carries
every fix; hosts 2, 3 and the AWS instance are deleted after their evidence was copied; host 1
(`166122805`) is stopped and kept with its artifact store until its 2026-09-17T04:00Z watchdog.
Nothing is published or claimed qualified.

**Second certified-class instance (AWS `i-0374e4bc6139ab0b5`, c7a.4xlarge, key
SHA256:VaXNnD2G4RoWcT7TV3McuoH27gAQfWfoMOgHn53RgLQ), control at Models 4a0d5cc with the certified
tuning:** verdict FAILED_RELATIVE_GATE — the absolute gate now passes. Rust arm USABLE: decode 18.6
tok/s, p95 TTFT 1546 ms, p95 e2e 3429 ms. Ollama 24.95 tok/s / 681 ms / 1888 ms (PRODUCTION_READY),
llama.cpp 25.5 / 1095 / 2484 (USABLE). This instance is slower for every engine than the first
(Ollama 28.4 there); the ratios are what the policy reads: decode 0.75, e2e 1.82. Correctness
27/27 and 9/9 again.

Fused grouped-query attention, first version (Models 12cb99b, same instance, three iterations):
native workers 8 → decode 16.5, prefill 90, TTFT 1519, e2e 3588; workers 12 → 15.6 / 107 / 1258 /
3424. Slower than the control on decode: the value pass walked column chunks outermost and rows
innermost, revisiting each V row per chunk per head. Second version (cbb00e6) streams rows in
blocks of four; its measurement follows.

Fused attention, second version (cbb00e6, rows streamed in blocks): workers 8 → decode 16.5,
prefill 90, TTFT 1481, e2e 3567; workers 12 → 16.1 / 107 / 1270 / 3495. Same as the first
version, so the value-loop order was not the cause of the decode drop. A local probe (Apple
Silicon, 128-bit lanes, 300 rows × 64 × 5 heads) times the fused pair at 63–68 µs against 69–75 µs
for the head-by-head kernels, so the kernel itself is not slower; an interleaved A/B of the
pre-fusion and current commits on the same instance decides whether the 18.6 → 16.5 is real.

Per-phase native workers (987211a; pool 16, decode 8): decode 15.1, prefill 130, TTFT 1082, e2e
3359; pool 12 / decode 8: 15.1 / 111 / 1225 / 3473; pool 16 / decode 6: 14.7 / 122 / 1121 / 3438.
Prefill and TTFT gain as expected from the larger pool; decode loses against a pool of 8 because
every job still wakes all workers (the generation broadcast is pool-wide) and the idle ones cost
their wake and their decrement. Keeping a pool of 8 for the qualification profile; the per-phase
switch stays available but is not part of the recorded tuning until the wake path is selective.

Interleaved A/B on the same instance (three iterations each, native workers 8): pre-fusion
4a0d5cc → decode 17.8 and 18.1 tok/s, TTFT 1534/1575 ms, e2e 3451/3371 ms; fused v2 with the
per-phase commit (987211a, decode threads unset) → 16.8 and 16.9, TTFT 1602/1604, e2e 3646/3682.
The fused v2 path is a real ~6% decode regression on this host, not noise. Kernel v3 (4c617ac:
lane reductions, register-resident value accumulators, vector softmax; 616 tests and the Granite
oracle pass) is measured next; if it does not beat the pre-fusion path, the per-head path returns.

Kernel v3 (4c617ac; lane reductions, register-resident value accumulators, vector softmax) on the
same instance: workers 8 → decode 19.1, prefill 93, TTFT 1450, e2e 3152; workers 12 → 18.3 / 115 /
1183 / 3068. Against the interleaved pre-fusion control (17.8–18.1 / 1534–1575 / 3371–3451) that is
+6% decode and −8% e2e; ratios 0.76 decode, 1.63 e2e. JFR on v3, main thread: 863 Java samples
(scoreGroup 333, accumulateGroup 160, softmax 54, swiGlu 42, rmsNorm 19, rope 15) against 1807
native samples; the score loop is a latency-bound dependent FMA chain per (row, head). Kernel v4
scores four rows per head with independent accumulators and pairs heads in the value pass: local
probe 94 → 52 µs per 300-row group, values still bit-identical, 616 tests and the oracle pass.
Native pool gains selective wake so a 16-worker pool with 8 active on decode no longer wakes the
idle eight; measured next together with v4.

**JIT-tier determinism defect found by CI (x86 runners, Models 4c617ac):** the nano-model
base-fork bit-identity test and the Spring AI embedding identity test failed in the last bits.
Cause: the vector softmax used `lanewise(EXP)`, which HotSpot serves from the vector math library
once C2 compiles the method and from scalar `Math.exp` before that, so identical inputs differed by
JIT tier; ARM has no such library, which is why the local suite passed. Fixed in f4d53f2 (scalar
`Math.exp`, which HotSpot keeps identical across interpreter and compiled code; vector maximum and
normalisation stay). Confirmed on the x86 host 1 (EPYC Genoa): 50 forward-pass and 3 kernel tests
pass at f4d53f2. Rule for this campaign: attention arithmetic must not depend on JIT tier, so no
vector transcendental intrinsics in the forward pass.

**Kernel v4 + selective wake (Models fe3bbbe, instance 2, three iterations each):**

| pool | decode workers | decode tok/s | prefill tok/s | p95 TTFT ms | p95 e2e ms |
| --- | --- | ---: | ---: | ---: | ---: |
| 8 | 8 | 19.4 | 92 | 1445 | 2956 |
| 16 | 8 | 20.2 | 140 | 1000 | 2694 |
| 12 | 8 | 19.6 | 113 | 1198 | 2905 |
| 16 | 10 | 20.0 | 130 | 1060 | 2783 |

Against this instance's Ollama control (24.95 tok/s, p95 e2e 1888 ms): pool 16 / decode 8 gives
decode 0.81 (floor 0.8) and e2e 1.43 (ceiling 1.5) — the first configuration that clears both
relative gates, with the absolute tier at USABLE (TTFT 1000 ms). Against llama.cpp (25.5, 2484):
decode 0.79, e2e 1.08. The selective wake is what makes the 16/8 split pay: with all workers woken
the same split gave 15.1 tok/s.

**Base run, attempt 6 plan (written before it runs):** same instance `i-0374e4bc6139ab0b5`, the
final Models commit that also carries the two JIT-tier determinism fixes (tier-stable exponential
and fixed-tree lane reduction; kernel arithmetic otherwise as measured above), harness invocation
via `base-host-run.sh` with `RAG_NATIVE_THREADS=16` and `RAG_TUNED_JAVA_OPTS` =
`-Dmodels.native.kernels.decodeThreads=8 -Dmodels.purejava.batchedAttentionScores=true
-Dmodels.purejava.batchedAttentionValues=true` for the performance phase; smoke untuned. The
previous bundle on this instance (control at 4a0d5cc) is moved aside as `rag.control-4a0d5cc`. The
verdict is copied whatever it says; the margins above are thin enough that instance variance can
decide it, and a FAIL is re-run once on a fresh instance of the same class before any conclusion.
Attempt 6 runs at Models `82a7433` (kernel v4, selective wake, tier-stable exponential and
fixed-tree lane reduction, nano test corrected to different tokens). Confirmed on the x86 host 1
before launch: 614 backend-java, 38 backend-native, and 4 embedding-adapter tests pass.

**Base run, attempt 6 result (Models 82a7433, instance 2, pool 16 / decode 8):**
FAILED_RELATIVE_GATE. Correctness 9/9 and 27/27, abstention 1.0. Rust arm USABLE: decode 18.9
tok/s, prefill 132, p95 TTFT 1052 ms, p95 e2e 2846 ms. Ollama 25.4 / 674 / 2122
(PRODUCTION_READY); llama.cpp 25.8 / 1028 / 2410 (USABLE). End-to-end passes against both (1.34,
1.18); decode fails against both (0.75, 0.73). The scalar exponential restored for tier
stability in f4d53f2 cost the margin the vector exponential had given (20.2 in the fe3bbbe
scratch). Bundle in `host-evidence/base-attempt6/`. Next commit: a tier-stable vector
exponential built from lanewise arithmetic only (range reduction and polynomial; local probe 29 →
7 µs per five 300-row rows); attempt 7 follows once the x86 suites and the Granite oracle pass.

**Attempt 7 plan (written before it runs):** Models `7a3f3e7` (attempt 6 code plus the tier-stable
vector exponential; x86 suites 615 + 38 pass on the instance, Granite oracle run locally before
launch). Same instance, same harness invocation and recorded tuning as attempt 6 (pool 16, decode
8, batched attention properties). If it passes, `7a3f3e7` is the frozen commit for every remaining
gate; the identity screen and both window arms restart at it (host 1 Rust, host `166144162`
pure Java), and attempt 6's bundle stays on record as the failed run it was.
