# Composition qualification — Granite 4.1 3B answerability hybrid

What this directory establishes, and what it does not.

**Established here (measured on a clean host):** both member markers resolve from Maven Central and
the hybrid opens and runs through the *public* ModelJars API, producing outputs byte-identical to
the rust-ffm component qualification, physically sharing the prefix on every composite case and on
no control case.

**Not established here:**

* The numbers the composition catalog entry and the composition evidence report bind.
  `assemble_composition_report.py` takes `controlMedianMillis`, `compositeMedianMillis` and
  `improvement` from the **4,096-token tier of the crossover report**
  (`../evidence/main/gate6-crossover-pure-java.json`), not from this run. The medians this run
  records are a second, independent measurement at the frozen window's own prefix lengths
  (~250–1,300 tokens), and they are reported as such.
* `peakRssBytes` and `modelsRevision`, two of the eight fields
  `GraniteAnswerability.Qualification` takes. This run records `modelsVersion` (`0.3.42`), not the
  Models git revision the published artifacts were built from, and it does not sample resident
  memory at all. `casesPerArm`, both medians and both `uniqueInferenceStateBytes` medians it does
  supply. Whoever fills the recipe module's `QUALIFICATION` after this run needs those two from
  somewhere else, or the program needs extending first.
* Anything about the control arm's prefix length — see §2.
* The `compositionSha256` convention — see §8.

## 0. Why this run uses the core API, not the recipe module

`org.modeljars.composite:granite-answerability` is the intended **supported facade** for this
hybrid, and it should become the way this run is written once it is published. It is not published
today:

```
$ curl -s -o /dev/null -w '%{http_code}\n' \
    https://repo1.maven.org/maven2/org/modeljars/composite/granite-answerability/0.1.47/granite-answerability-0.1.47.jar
404
```

*(measured 2026-09-17)*. Its publication is gated on `graniteAnswerabilityQualified`, i.e. on the
composition entry existing in `catalog/compositions.json` — which is exactly what this run exists to
qualify. Depending on it here would be circular as well as unresolvable.

`CompositionCleanHost.java` therefore calls precisely what
`org.modeljars.composite.granite.GraniteAnswerability` calls:

| Recipe module does | This program does |
| --- | --- |
| `GraniteAnswerability.open()` | `ModelJars.openActivatedToolRuntime(BASE, SPECIALIST, ModelLoadOptions.builder().backend(ModelBackend.NATIVE).build())` |
| `GraniteAnswerability.render(request)` | the same four calls into `com.integrallis.models.runtime.chat.GraniteDocumentsPrompt` |
| `GraniteAnswerability.classify(...)` | `runtime.model().openToolTurn(prompt, strategy)` + `generateToolCall` |

Rendering is the only thing the recipe module adds over the core API, and
`GraniteDocumentsPrompt` is a public class of `com.integrallis:models-runtime:0.3.42` on Central —
the same class the rust-ffm component clean host (`../clean-host/AnswerabilityCleanHost.java`) uses
for its specialist arm.

### Backend: explicitly native

The Granite 4.1 3B Q4_K_M marker declares `backend.pure-java=false` / `backend.rust-ffm=true` and
carries only a `rust-ffm` RAG qualification; the component's own evidence in
`modeljars-0.1.47.jar!/META-INF/modeljars/component-qualifications-v1.properties` records
`backend=rust-ffm`. So the run requests `ModelBackend.NATIVE` explicitly rather than leaving the
choice to `AUTO`.

ModelJars PR #166 (merged, in 0.1.47) is what makes that possible: the activated path now calls
`selectActivatedQualification(baseDescriptor, options.backend())`. Before it, the call hard-coded
`ModelBackend.JAVA` and died on this base with *"has no qualified pure-java execution"* — which is
the blocker the earlier draft of this file recorded and which is now closed.

Because the backend is native, the JVM needs `--enable-native-access=ALL-UNNAMED` (ModelJars'
`requireNativeAccess` refuses to open without it, rather than warning) and
`com.integrallis:backend-native:0.3.42` on the classpath. Both are declared in the program:
`//JAVA_OPTIONS` and `//DEPS`. backend-native also arrives transitively through
`org.modeljars:modeljars:0.1.47`; it is declared explicitly anyway so the requirement is visible in
the program and the wrapper can fail closed on it.

## 1. Preconditions

| Input | Value | Status |
| --- | --- | --- |
| Base marker | `org.modeljars.huggingface:ibm-granite.granite-4.1-3b-gguf.q4_k_m:4.1.0-q4_k_m.2` | on Central |
| Specialist marker | `org.modeljars.github:modeljars.activated-adapters.granite-4.1-3b-answerability-alora-integrallis.f32:1.0.0-f32.2` | on Central |
| ModelJars runtime | `org.modeljars:modeljars:0.1.47` | on Central |
| Models runtime | `com.integrallis:models-runtime:0.3.42`, `com.integrallis:backend-native:0.3.42` | on Central |
| Recipe module | `org.modeljars.composite:granite-answerability` | **not published** — see §0 |
| Frozen window v2 | sha256 `dfb8cd7203418c79bae27306ffc4857f0ab02be7f9f8ddcae793af556e9cab37` | pinned raw GitHub |

## 2. What runs on the clean host

Fresh Ubuntu 24.04, run as root, with `CompositionCleanHost.java`,
`write_composition_clean_host_run.py` and `run-composition-clean-host.sh` in one directory:

```
sudo bash run-composition-clean-host.sh ./evidence
```

Configuration is in the environment, recorded verbatim in the run JSON:

```
CASES_PER_SUITE=3            # frozen-window cases per suite per arm; 1..6
ARM_ORDER=composite-first    # or control-first
MODELJARS_VERSION=0.1.47     # must match the //DEPS in CompositionCleanHost.java
MODELS_VERSION=0.3.42
```

Each of the first `CASES_PER_SUITE` cases of `squad-v2-dev` and `msmarco-v2.1-validation` is run
twice on one loaded hybrid:

* **control** — `PrefixStrategy.RECOMPUTED`: the base prefix is evaluated independently for every
  specialist call, with no physical KV sharing;
* **composite** — `PrefixStrategy.SHARED`: both branches fork from one physical KV prefix.

### What every case must produce

Expectations come from the **rust-ffm** arm's own committed evidence —
`../evidence/main/window-squad-v2-dev-specialist-rust-ffm.json` and
`../evidence/main/window-msmarco-v2.1-validation-specialist-rust-ffm.json`, both
`modelsRevision 6063076e476ba7a477e3f6dd966a77b571352d29`, `kernelPlan rust-ffm-v13`. rust-ffm is
the only backend this composition can run on, so the pure-Java tables are not the comparand even
where they agree entry for entry.

| | control arm | composite arm |
| --- | --- | --- |
| `output` | byte-identical to the rust-ffm evidence | byte-identical to the rust-ffm evidence |
| `structured` | true | true |
| `physicallySharesPrefix` | **false** | **true** |
| `sharedPrefixTokens` | **0** | the window's count for that case |

`sharedPrefixTokens` counts tokens held in *physically shared* KV storage, so a recomputed turn
reports `0` by construction — `ActivatedToolCallingModel#openRecomputedToolTurn` passes a literal
zero. The public API exposes no prefix-length accessor for a turn that shares nothing, so **this run
cannot check that the control arm saw the same prefix length**, only that it shared none of it. That
the arm switch is not a silent no-op is established by `physicallySharesPrefix` and by
`uniqueInferenceStateBytes` (below), both of which move.

Anything else exits non-zero and the writer records `pass: false`.

### What each case records

`arm`, `suite`, `id`, `promptSha256`, `output`, `label`, `structured`, `physicallySharesPrefix`,
`sharedPrefixTokens`, `uniqueInferenceStateBytes`, `millis`, `pass` — plus, per arm, the median
`millis` and the median `uniqueInferenceStateBytes`.

`uniqueInferenceStateBytes` is `ActivatedToolTurn#uniqueInferenceStateBytes()`: the inference-state
bytes reachable from both branches, counting the physically shared prefix once. It is read before
the turn closes and outside the timed interval. It is the memory side of what sharing buys, and it
feeds `GraniteAnswerability.Qualification`'s `sharedUniqueStateBytes` /
`recomputedUniqueStateBytes` once the recipe module is published. A measurement that could not be
taken is recorded as `-1` and fails the record, rather than being silently reported as zero.

Default `CASES_PER_SUITE=3` is 12 measurements. Budget the base download (2.1 GB), the adapter
download (119 MB), model load, and roughly 15–25 minutes of measurement on pilot-2 host geometry.

## 3. How `resolvedFromCentral` and `runViaPublicApi` are derived

The composition gate refuses to assemble a report unless both are exactly `true`, so neither may be
an assertion. Both are derived at runtime, and every input to the derivation is recorded beside the
result so it can be audited instead of believed.

### `resolvedFromCentral` — per member

`CompositionCleanHost#markerSource` enumerates every `META-INF/modeljars/registry.properties` on the
classpath, finds the one that declares `model.<modelId>.markerCoordinate`, and takes the jar that
resource came out of. That jar — not a coordinate string — is then:

* **hashed** on this host (`markerJarSha256`), and
* **located**: `resolvedFromCentral` is true iff the jar's own path ends in the Maven repository
  layout its coordinate implies, compared element by element
  (`<group as directories>/<artifact>/<version>/<artifact>-<version>.jar`, via `Path#endsWith`).

The `markerJarUri` it was hashed from is recorded too.

What that alone proves is that a Maven-layout resolver put the artifact in a local repository under
its exact coordinate. Three further facts close it to *Central*:

1. `fresh-check.txt` records, before anything else runs, that `~/.m2`, `~/.jbang` and `~/.gradle`
   did not exist — nothing could have been seeded;
2. `jbang-resolve.log` is the resolver's own verbose log, naming `central=https://repo1.maven.org/maven2/`
   as the only repository;
3. `write_composition_clean_host_run.py` cross-checks every `markerJarSha256` against
   `resolved-classpath.txt`, which the wrapper builds independently by hashing every jar
   `jbang info classpath` reports. A member only one of the two saw fails the record.

### `runViaPublicApi` — a property of the run, recorded on both members

`CompositionCleanHost#publicApi` derives it as the conjunction of four observations of the loaded
runtime, each recorded separately in `publicApi`:

| fact | what is observed |
| --- | --- |
| `publicApiClassFromCentral` | `org.modeljars.ModelJars` and `org.modeljars.ModelJarActivatedRuntime` were loaded from one and the same jar — via each class's protection-domain code source — and that jar sits at the Maven layout `org.modeljars:modeljars:0.1.47` implies. Its sha256 is recorded. |
| `runtimeTypeOwnedByModelJars` | the object this run drove is exactly `org.modeljars.ModelJarActivatedRuntime`, and that class declares **no public constructor**, so no code outside package `org.modeljars` can have minted it. |
| `membersResolvedByRuntime` | the runtime's own `baseDescriptor()` / `adapterDescriptor()` carry the two member marker coordinates and both artifact digests, i.e. ModelJars' registry resolved the markers — this run never constructed a descriptor. |
| `backendSelectedByCatalog` | the runtime's `baseQualification()` names the base model id and binds the same artifact digest the descriptor carries, i.e. the catalog gate ran and chose the backend. Its `reportUri` is logged. |

Holding that object is itself evidence of one more check having passed:
`ModelJarActivatedRuntime`'s constructor refuses to return unless the *loaded adapter's* embedded
base and adapter digests equal the verified descriptors' digests.

**What this cannot establish** is that the program's source called
`ModelJars.openActivatedToolRuntime` rather than some equivalent internal path. Nothing observable
at runtime distinguishes those. What closes that gap is that the program source is committed in this
directory and its sha256 is recorded in `program-sha256.txt` beside the evidence — so the claim is
checkable by reading one file, not by trusting a boolean.

`write_composition_clean_host_run.py` re-checks all of it: the entry point must be that method, the
selected backend must be `rust-ffm`, every one of the four facts must be true, `runViaPublicApi`
must equal their conjunction (a flag that does not follow from its own facts fails), and both
members must agree with it.

## 4. Outputs to commit

Everything the wrapper writes under `evidence/`, in particular:

| File | Role |
| --- | --- |
| `composition-clean-host-run.json` | the composition clean-host record |
| `published-artifacts.json` | the `--published-artifacts` input to `assemble_composition_report.py` |
| `composition-program-report.json` | per-case measurements, medians, artifact resolution, public-API derivation |
| `composition-clean-host-output.log` | `outputLog` — its sha256 and size are bound in the run JSON |
| `resolved-classpath.txt` | `resolvedClasspathSha256` is the sha256 of these bytes |
| `jbang-resolve.log` | the resolver's own record of which repository it fetched from |
| `fresh-check.txt`, `host-baseline.txt`, `toolchain.txt`, `java-version.txt`, `program-sha256.txt` | host and toolchain provenance |
| `SHA256SUMS` | checksums of every committed evidence file |

Then replace the literal `<EVIDENCE_REVISION>` in `composition-clean-host-run.json`'s
`outputLog.uri` with the `integrallis/models` commit that contains the log.

The wrapper fails closed, before the run, if `CompositionCleanHost.java` does not declare every
pinned `//DEPS` and `--enable-native-access=ALL-UNNAMED`, if it *does* depend on the unpublished
recipe module, or if either member marker, the ModelJars jar, `models-runtime` or `backend-native`
is missing from the resolved classpath.

## 5. Assembling the composition report

From `benchmark-results/2026-09-15-granite-4.1-3b-alora-hybrid/`, with the pilot-2 evidence paths
from the component report's own command plus the composition arguments:

```
python3 assemble_composition_report.py \
  --composition-id granite_4_1_3b_answerability_hybrid \
  --base-member-id ibm_granite_granite_4_1_3b_gguf_q4_k_m \
  --specialist-member-id modeljars_granite_4_1_3b_answerability_alora_integrallis_f32 \
  --specialist-kind first-party-rag-specialist \
  --published-artifacts <evidence>/published-artifacts.json \
  ... (every argument the component report was assembled with) \
  --output composition-report/composition-qualification.json
```

`--published-artifacts` must carry both members with `resolvedFromCentral` and `runViaPublicApi`
exactly `true`; the assembler exits rather than assemble a report without them.

## 6. Unit tests

```
python3 -m unittest write_composition_clean_host_run_test -v   # from this directory, 42 tests
```

## 7. Dev-machine verification (not the clean-host run)

Run on 2026-09-17 on an Intel Core i7-9750H MacBook Pro, Temurin 25.0.3, macos-x86_64 native kernel,
with `org.modeljars:modeljars:0.1.47` and both member markers resolved from Maven Central into an
otherwise-populated `~/.m2`, and the base GGUF pre-seeded into the ModelJars content-addressed cache
from an existing local copy. **This is not a clean-host record** — the machine is not fresh, the JDK
is not the pinned `25.0.4.1+1`, and no wrapper record was produced. It verifies that the program
compiles against the published artifacts, opens the hybrid through the core API, and that both arms
behave as this file says.

The marker jar digests the program measured on that host,
`fb6aba90d2f06ad70282fd99eed356030e22b0f8a2620c13bab1492df92a3a11` (base) and
`ca260264f048889afd641b84ca10a09c2b5dfe934be475d7b1639ce7308a0b4f` (specialist), equal the digests
of the same two files downloaded directly from `repo1.maven.org`. The public-API jar measured
`b8800ff6942fc17ffba258b8e03f8a2e9c72361958fc7b2a4cf37079d1852a91`, and `backend-native-0.3.42.jar`
resolved to `3e62b2e59df079428f3b90802b48674fcac22dc256617ab5e90156a4b16c95d3` — the same digest the
rust-ffm component clean host recorded.

`CASES_PER_SUITE=3`, `ARM_ORDER=composite-first`: **12/12 measurements passed, exit 0**. Every case
in both arms was byte-identical to the rust-ffm evidence; the composite arm shared physically on all
six cases with the window's token counts, the control arm on none.

| suite | id | output | shared tokens (composite) | composite ms | control ms | composite unique bytes | control unique bytes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| squad-v2-dev | 5ad247b0d7d075001a428b45 | `"unanswerable"` | 269 | 21 677 | 34 485 | 89 150 720 | 167 813 120 |
| squad-v2-dev | 57267640f1498d1400e8e074 | `"answerable"` | 386 | 26 152 | 48 358 | 89 150 720 | 167 813 120 |
| squad-v2-dev | 5737432bc3c5551400e51e9b | `"answerable"` | 357 | 24 464 | 44 536 | 89 150 720 | 167 813 120 |
| msmarco-v2.1-validation | 95542 | `"unanswerable"` | 776 | 54 156 | 98 276 | 173 057 280 | 335 626 240 |
| msmarco-v2.1-validation | 851555 | `"answerable"` | 1 082 | 70 502 | 128 140 | 340 870 400 | 671 252 480 |
| msmarco-v2.1-validation | 1067349 | `"answerable"` | 853 | 49 821 | 96 294 | 173 057 280 | 335 626 240 |

Medians: control **72 326 ms**, composite **37 986.5 ms**, latency improvement **0.4748**; median
unique inference-state bytes control **251 719 680**, composite **131 104 000**.

These are *this laptop's* numbers on a 6-core Coffee Lake, not a result for anything. They are here
because they show the two arms are genuinely different work — the control arm costs about twice the
wall time and about twice the live inference state, exactly as recomputing rather than sharing the
prefix should.

`ARM_ORDER=control-first` was exercised separately at `CASES_PER_SUITE=1` (4/4, exit 0), and
`write_composition_clean_host_run.py` was run against the real 12-measurement program report with
the real classpath fingerprint and log — it produced `pass: true` and a `published-artifacts.json`
with both members at `resolvedFromCentral: true` / `runViaPublicApi: true`. The one input a dev box
cannot supply, `fresh-check.txt`, was substituted; every other check ran against real data.

Three findings from that run are folded into the program and the writer:

1. `membersResolvedByRuntime` was false because `ModelJarDescriptor#markerCoordinate()` returns a
   `ModelJarCoordinate`, not a `String`, so the coordinate comparison could never succeed.
2. The control arm reports `sharedPrefixTokens = 0`, not the window's count. The earlier draft
   required the window's count in *both* arms, which no correct control run can satisfy. See §2.
3. The final `PASS measurements=… passed=…` line is now printed **after** the hybrid is closed. The
   record's log check reads the last non-empty line of `composition-clean-host-output.log`, and
   closing the runtime is free to log; printing the verdict before the close left a passing run one
   stray log line away from being recorded as a failure.

## 8. Convention note: `compositionSha256`

`build.gradle.kts` only requires 64 lowercase hex characters; no derivation is enforced or
documented. The value drafted earlier,
`b2dbee7f22de346d4f420824ffe652a43e3dfbb1f7177ec8359176aa68200633`, was computed over the **f32.1**
specialist coordinate; the composition now names **f32.2**, so it is stale regardless of convention.
The recipe recorded with it — sha256 over one tab-separated
`<role>\t<modelId>\t<markerCoordinate>\t<memberBundleSizeBytes>\t<artifactSha256>\n` line per member,
sorted by role — did not reproduce that digest here under any of the obvious role vocabularies
(`base`/`specialist`, `base`/`adapter`, `base`/`component`, `base`/`first-party-rag-specialist`), so
the convention is not pinned down by this directory. The value must be regenerated by whatever tool
owns it before the catalog entry is written; do not carry the old one forward.
