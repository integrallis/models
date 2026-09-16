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
   every arm, including losing ones. A Models-owned Rust/FFM kernel may be measured as a separate
   arm only after the pure-Java arm passes gates 1 to 5, only with a Java reference and fallback,
   and only if it is token-identical to the Java arm at every tier.
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
