# Granite 3.2 aLoRA query-rewrite qualification preflight

- Experiment: verify an upstream IBM Granite 3.2 8B query-rewrite aLoRA as the first candidate
  composed model with an exact, physically shared base KV prefix.
- Provider / plan: Hetzner Cloud, `cpx62`, `fsn1`, 16 shared x86-64 vCPU, 32 GiB RAM, 640 GB disk.
- Price / ceiling: USD 0.2452 per hour; one host, maximum four hours (USD 0.99) including setup.
- Creation deadline: 2026-09-14T22:30:00Z. Deletion deadline: 2026-09-15T02:30:00Z, or immediately on
  a failing correctness gate.
- Provisioned resource: Hetzner server `165969464` (`modeljars-granite-alora-20260914`), provider
  firewall `11624612`, IPv4 `5.75.251.129`. SSH is restricted to the operator workstation's observed
  `/32`; the server and firewall are both temporary and must be deleted together after evidence copy.
- Correctness gates: exact artifact and upstream-adapter identities; tokenization and first-marker
  boundary match IBM's aLoRA implementation; Java shared and recomputed branches produce identical
  token IDs; Java output matches an independent reference run; the documented query-rewrite cases
  meet the frozen task criteria.
- Performance gates: shared prefix references identical immutable storage, reports less unique KV
  state than recomputation, and measures the 256 / 1024 / 4096-token crossover without an external
  runtime in the product process.
- Evidence destination: this directory. The remote checkout and all model bytes are disposable after
  report hashes are copied and verified locally.

## Execution record

- The original Hetzner `cpx62` host and its dedicated firewall were deleted after the first 256-token
  control established physical sharing. Its retained log is `cpx62-256-baseline.log` (SHA-256
  `2e6836ee0e172560a0b8c8c9171c782ee90548df2916032f5763022ee7b06d19`). It measured recomputed
  handoff-to-first-token at 211,093.9 ms versus 119,100.7 ms shared, with unique inference state
  falling from 251,688,960 to 131,088,000 bytes.
- The four-tier run moved to AWS `c7i.4xlarge` (`i-0a51cf753c1122389`), Intel Xeon Platinum 8488C,
  16 vCPU / 30 GiB RAM, with the supported 512-bit Vector API width explicitly recorded. The
  instance and its SSH-only security group `sg-03876de3ffeb2bcd6` were deleted after evidence copy.
- `c7i4-512-prefix-sharing.log` (SHA-256
  `25df970c3f8233d5e557b59820d3c9fb09a5859739c0133f1ab9243c4b5ada61`) records physical sharing
  and token-exact output at every required tier: 256: 260,560.6 ms recomputed / 148,578.6 ms shared;
  1024: 1,073,533.2 / 553,470.3 ms; 4096: 6,597,127.7 / 3,367,132.8 ms. Unique inference state at
  4096 fell from 4,027,023,360 to 1,389,532,800 bytes.
- The first run completed all inference but failed only while serializing the `Optional` crossover
  field in its JSON evidence. Commit `84fad94` adds the JDK8 Jackson module and a regression test;
  the corrected benchmark must be re-run to produce its machine-readable report.
- The prior Java-only smoke entry is superseded. It did not preserve the upstream literal marker
  through packaging, so it is excluded from qualification evidence.

## 2026-09-15 local artifact correction and rerun

- The initially staged runtime metadata encoded `\"` around `rewritten_question`. IBM's published
  `adapter_config.json` contains literal JSON quotes. The runtime correctly rejected that altered
  marker before inference; it was a packaging error, not a model failure and it is not counted as
  a qualification run.
- The fail-closed upstream packager now pins the source model card, adapter configuration, base
  revision, base GGUF SHA-256, tokenizer source hashes, projection set, and the marker tokens from
  the loaded Granite tokenizer. Its three unit tests cover successful packaging, a wrong base, and
  an unsupported projection.
- The verified Q4_K_M base hash is
  `363f0bbc3200b9c9b0ab87efe237d77b1e05bb929d5d7e4b57c1447c911223e8`. The unmodified adapter
  hash is `ecb2d17dee8147b310d3d8e1ac925d34aac3551b3a960540b7a7f57165cdaa2e`.
- The new dedicated `granite32AloraIntegrationTest` passed against the actual base and adapter:
  the 81-token publisher marker was found at the generated prompt boundary and the base and
  activated branches physically shared their prefix.
- A fresh Java-only smoke on the local Apple-Silicon workstation passed at temperature zero:
  `PASS shared=true prefix=41 rewrite={"rewritten_question":"Who is the CEO of Microsoft?"}`.
  Wall time was 4m46s including model load, shared-prefix prefill, and decoding. This remains a
  smoke only; held-out semantic cases, an independent reference comparison, and the corrected
  machine-readable crossover report remain required before any catalog promotion.

## 2026-09-15 MT-RAG Cloud retrieval screen — rejected

- The completed 12-case Java run was joined to the identically selected, traceable MT-RAG Cloud
  suite by its deterministic case identifier. All 12 results were structured and physically shared.
- `score_mtrag_cloud_retrieval.py` applies the same pinned BM25 implementation (k1 1.2, b 0.75)
  to the final user turn, MT-RAG reference rewrite, and generated rewrite against the benchmark's
  passage-level Cloud corpus and judged relevance ids. It retains only query-term postings, so the
  corpus remains source data rather than a product dependency. The unit test covers both scoring
  and refusal of unshared/unstructured output.
- The predeclared screen requires generated Recall@10 to be at least the raw final-turn Recall@10
  and at least 90% of the reference-rewrite Recall@10. Results: raw 0.333, reference 0.417,
  generated 0.167. The generated rewrite is below both thresholds.
- **Decision: reject this adapter as a ModelJar candidate.** The artifact must not be published,
  listed in the public catalog, or described as a qualified hybrid model. The machine-readable
  screen report is retained outside the source checkout with the pinned suite, report, qrels, and
  corpus SHA-256 values embedded in it.
