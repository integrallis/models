# Final-artifact Bonsai public QA

Run all 200 frozen public cases (100 SQuAD2, 100 BoolQ), with unchanged prompts, token oracle, sampling, context, scoring and descriptive regression limits. Candidate is the ordinary Models generation loop at the immutable source and actual Vectors core hash in protocol.json. Publisher executable and every shared library retain the exact hashes from the preceding public-QA archive. Both arms run sequentially on the same host for each paired single-case block. Twenty x86 GitHub hosts each handle 10 preassigned cases; no case selection, optional stopping, result-dependent retries or discarded errors. All ten blocks upload separately outside inference. A failed/incomplete block blocks the final verdict.

This new cohort uses four visible processors and four publisher threads on JDK 25.0.3; previous EC2 observations are not pooled. Record actual hosts, native library, class origins, all runtime JARs and per-case outputs. Quality is the scope: neither publisher superiority nor latency portability follows. Frozen descriptive gate: no failures, candidate truncations no greater than publisher, SQuAD2 F1 and BoolQ accuracy differences at least -0.03. Report exact raw answer disagreements independently. This bounded sample is not universal quality or statistical noninferiority.

Each host has a 300-minute hard job limit. The reference URL is a short-lived read-only signed URL kept in an Actions secret; it grants no cloud credentials. Archive hash and every extracted executable/library hash are verified before use. No compute instances are launched.

Scheduling repair: prior run38328345931 timed out five-case blocks; its retained
observations remain in the historical receipt and are never pooled into this run.
The revised cohort requires exactly20 host manifests,200 complete paired blocks,
and all200 cases in both arms. Each block has a30-minute Actions limit and a
29-minute internal deadline shared by both arms, leaving time to terminate the
publisher and upload partial outputs on failure. An execution receipt is written
before every subprocess and records timeout explicitly; even an interrupted arm
retains its header, stdout/stderr, server log, and completed observations. No
partial arm is scored as a completed case. Job timeout remains a failure, never
permission to accept a partial cohort.

Token preflight already covers only each host's assigned cases (now10), so it is
retained unchanged. Models/Vectors revisions are read from protocol.json by the
workflow; source and actual core SHA are still verified before inference. This
workflow is dispatch-only: preparing or pushing the repair does not launch it.
Final runtime freeze and renewal of the read-only publisher URL are required
before dispatch. Prompts, token oracle, Java/publisher generation code, scorer,
and quality limits are unchanged.

Prepared candidate runtime source is Vectors `ef30f0526db24970e1c662cd719842a6f748d9db`,
actual core SHA256
`6102074aefb48b76537f79b7a1dc89eee5f9e5b8f2fbdcd4d5428df8914871e6`.
Models source is `00782f64092a14fccb437a4e85bbd866b6c4b11a`; this source adds native-width Q5_1 decoding and
exact eight-lane reductions using two 128-bit vectors on narrow hosts. The old13f0 run is not final-runtime
quality evidence and will not contribute observations to this cohort.

This candidate uses degree-five repeated-squaring SiLU on [-32, 32]. Its numerical proof does
not transfer earlier model-quality results. The initial Linux component timing screen passes on EPYC7763 and Neoverse-N2.
The corresponding EPYC9V74 regression check remains pending. No quality gate or
workload was changed. A fresh full-model result is required for this composition.
