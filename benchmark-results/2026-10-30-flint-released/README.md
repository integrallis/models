# Flint verification against released Models 0.5.8

The required released-library checks passed on a bounded remote m6a.4xlarge host.
Product JARs came from Maven Central and were checksum-verified; benchmark harness
JARs are explicitly source-built. See `results/inputs.json`, `results/classpath.json`,
the deployment manifest, and `validation/release-receipt.json` for provenance.

- Numerical parity passed against the pinned reference.
- Library-default smoke passed 9/9 pipeline cases with no failed generations.
  Its default 256-token budget truncated four visible answers. This is not a
  complete-answer claim: the qualified RAG configuration uses 512 output tokens.
- The public BackendConfiguration profile changed only native threads from the
  unset default (16) to 8. One warmup and three measured runs per arm produced
  identical complete 143-token outputs matching the source qualification.
  Median decode improved from 9.9654 to 11.3939 tokens/s (14.33%).
  This repeated-prompt compatibility check is not a new full RAG qualification;
  its warm-cache TTFT must not be compared with the full workload's TTFT.

The unchanged production-rag-model-contribution-v6 qualification, independent
Ollama comparison, and held-out answer review remain in
`benchmark-results/2026-10-30-flint-qualification` at Models commit
3098dd306900e4901b27b2615345b42da2c29b46. Their source-built backend version is
retained honestly rather than relabeled as a measurement of released 0.5.8.

This checkpoint was captured while the optional tool diagnostic was still
running. `results/completed.txt` marks completion of the required release checks,
not completion of the entire campaign. No tool qualification or catalog
publication is claimed. Final diagnostic results and host cleanup are recorded
separately when available. The host has an independent 90-minute shutdown bound.

Validation logs describe the catalog candidate at checkpoint time. The expected
publication timestamp guard failure confirms that SDK/site publication cannot
advertise a marker before its actual publication time is recorded.

The tool probe source is referenced by immutable worker revision and SHA-256 in
`deployment/tool-probe-source.json`. It is the exact source used by the remote
worker. A redundant Java copy was removed because the evidence-writer literal
guard interpreted its negative scorer input (`extra: true`) as an asserted
measurement. The canonical source, guard, measured reports, and frozen remote
inputs are unchanged.

## Final diagnostic and cleanup

All 42 optional diagnostic cases completed; the worker exited zero. The unchanged
original single-turn suite scored 10/14, the expanded ordered-plan prompt 9/14,
the two original multi-step tasks completed 2/2 with simulated-result continuation,
and the twelve fresh cases scored 10/12. Manual visible-answer review and the
legacy flight-date scorer limitation are recorded in
`validation/tool-diagnostic-review.json`. No tool capability is qualified.

Both the EC2 instance and its root volume are gone, verified through AWS in
`validation/final-cleanup.json`. The earlier released checkpoint is retained as
a historical checkpoint; it is not rewritten to claim that cleanup was already
complete at capture time. The later full catalog CI command passed locally,
including clean Maven/Gradle consumers; publication remains pending review.
