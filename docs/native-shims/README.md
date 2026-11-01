# Native shim evidence

`registry.json` inventories every CPU capability bit and exported CPU/CUDA function, links the
Java dispatch sites, and records the evidence status of each operation. All seventeen current
operation groups are **unproven**. Historical speedups and repaired correctness defects
are not certificates that a JVM limitation requires Rust.

Run `./gradlew verifyNativeShimEvidence` (also part of `complianceCheck`). The check detects:

- changed Java/Rust capability masks, reuse of a retired bit, or unregistered exports;
- missing dispatch anchors, operation ownership, or historical evidence;
- a new unproven entry outside the explicit migration checklist;
- a claimed proof without all five gates, exact source/dependency hashes, all declared host
  profiles, two datasets, executed Java and native calls, and hashed raw evidence;
- retired operations whose capabilities or symbols are still exported.

The existing migration checklist records unfinished work. Passing the inventory check does not
finish that work or prove performance. `./gradlew verifyNativeShimCampaignComplete` deliberately
fails while any entry remains unresolved. It is the campaign completion gate, not a fabricated
release certificate. A ready Java replacement remains unresolved until it is adopted and the
unused native route is retired.

A `PROVEN_REQUIRED` entry references a hashed decision JSON containing `gates` (`correctness`,
`javaInvestigation`, `jvmCause`, `necessity`, `retirement`, each `PASS`), `sources` (all files in its
source scope, each with `path`/`sha256`), and `profiles`. Every profile specifies `host`, `jvm`,
positive `javaCalls` and `nativeCalls`, at least two `datasets`, `decision: PASS`, and nonempty
`rawEvidence` references. The decision also explains `jvmLimitation`, `controlledExperiment`,
`workloadRequirement`, `retirementCommand`, and `review`.

Those fields make an experiment inspectable. A structural validator cannot judge the experiment's
causal reasoning, numerical validity, or statistical quality. Technical review remains mandatory.
Keep rejected experiments and failures; never update a historical hash to disguise changed data.
If a correction changes a report, retain the original and label the new assessment explicitly.

Source scopes are conservative: CPU or CUDA implementation, Java inference, dispatch, build
configuration and dependency locks. Changing an affected file invalidates a completed proof;
unrelated article edits do not. Supporting workers inherit justification from the compute
operations they serve and need no invented, independent JVM limitation.

The BF16 and SwiGLU candidate entries remain UNPROVEN. Their [inventory](bf16-swiglu-candidate-inventory.md) separates measured compiler/allocation mechanisms and component screens from missing final composed and supported-profile gates. Raw rejected experiments remain outside project repositories.

[ABI7 symbol retirement](abi7-symbol-retirement.md) lists removed compatibility wrappers and the retained generic entry points.
