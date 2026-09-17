# Composition qualification — Granite 4.1 3B answerability hybrid

What this directory establishes, and what it does not.

**Established here (measured on a clean host):** both member markers resolve from Maven Central and
the hybrid opens and runs through the *public* ModelJars API, producing outputs byte-identical to
the component qualification, physically sharing the prefix on every composite case and on no
control case.

**Not established here:** the numbers the composition catalog entry and the composition evidence
report bind. `assemble_composition_report.py` takes `controlMedianMillis`,
`compositeMedianMillis` and `improvement` from the **4,096-token tier of the crossover report**
(`../evidence/main/gate6-crossover-pure-java.json`), not from this run. The medians this run
records are a second, independent measurement at the frozen window's own prefix lengths
(~250–1,300 tokens), and they are reported as such.

## 0. Preconditions

| Input | Value | Status |
| --- | --- | --- |
| Base marker | `org.modeljars.huggingface:ibm-granite.granite-4.1-3b-gguf.q4_k_m:4.1.0-q4_k_m.2` | on Central |
| Specialist marker | `org.modeljars.github:modeljars.activated-adapters.granite-4.1-3b-answerability-alora-integrallis.f32:1.0.0-f32.1` | on Central |
| Recipe | `org.modeljars.composite:granite-answerability:<release>` | **not published yet** |
| ModelJars runtime | `org.modeljars:modeljars:<release>` | 0.1.46 on Central; see the blocker below |
| Frozen window v2 | sha256 `dfb8cd7203418c79bae27306ffc4857f0ab02be7f9f8ddcae793af556e9cab37` | pinned raw GitHub |

### Blocker: the base has no qualified pure-java execution

`ModelJars.openActivatedToolRuntime(base, adapter)` calls `selectQualification(baseDescriptor,
ModelBackend.JAVA)`. The Granite 4.1 3B Q4_K_M catalog entry declares `backends."pure-java": false`
and carries only a `rust-ffm` RAG qualification, so the call fails before installing anything:

```
org.modeljars.ModelJarException: ModelJar
org.modeljars.huggingface:ibm-granite.granite-4.1-3b-gguf.q4_k_m:4.1.0-q4_k_m.2
has no qualified pure-java execution
```

Reproduced on 2026-09-17 against `org.modeljars:modeljars:0.1.46` plus both marker JARs, on a dev
machine. The component qualification itself never hit this: it loaded
`PureJavaBackend.loadActivatedAdapter` from the Models library directly, bypassing ModelJars'
catalog gate. The composition run must not bypass it — running through the public API is exactly
what the composition gate requires — so this must be resolved before the clean host can run. See
the PR description for the two candidate resolutions.

## 1. What runs on the clean host

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

* **control** — `Prefix.RECOMPUTED`: the base prefix is evaluated independently for every
  specialist call, with no physical KV sharing;
* **composite** — `Prefix.SHARED`: both branches fork from one physical KV prefix.

Both arms must produce the exact output and `sharedPrefixTokens` recorded in
`../evidence/pj1/window-squad-v2-dev-specialist-pure-java.json` and
`../evidence/pj3/window-msmarco-v2.1-validation-specialist-pure-java.json`; the composite arm must
physically share on every case and the control arm on none. Anything else exits non-zero and the
writer records `pass: false`.

Default `CASES_PER_SUITE=3` is 12 measurements. On the pilot-2 host geometry the specialist arm
took 24–115 s per case, so budget roughly 15–25 minutes of measurement plus the base download
(2.1 GB) and model load.

## 2. Outputs to commit

Everything the wrapper writes under `evidence/`, in particular:

| File | Role |
| --- | --- |
| `composition-clean-host-run.json` | the composition clean-host record |
| `published-artifacts.json` | the `--published-artifacts` input to `assemble_composition_report.py` |
| `composition-program-report.json` | per-case measurements, medians, artifact resolution |
| `composition-clean-host-output.log` | `outputLog` — its sha256 and size are bound in the run JSON |
| `resolved-classpath.txt` | `resolvedClasspathSha256` is the sha256 of these bytes |
| `SHA256SUMS` | checksums of every committed evidence file |

Then replace the literal `<EVIDENCE_REVISION>` in `composition-clean-host-run.json`'s
`outputLog.uri` with the `integrallis/models` commit that contains the log.

## 3. Assembling the composition report

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

## 4. Unit tests

```
python3 -m unittest write_composition_clean_host_run_test -v   # from this directory, 28 tests
```

## 5. Convention note: `compositionSha256`

`build.gradle.kts` only requires 64 lowercase hex characters; no derivation is enforced or
documented, and the withdrawn Qwen entry's value could not be reproduced from any obvious one. The
value in the drafted catalog entry is therefore a *declared* identity computed the same way
`artifactBundleSha256` computes a bundle identity — sha256 over one tab-separated line per member,
sorted by role:

```
<role>\t<modelId>\t<markerCoordinate>\t<memberBundleSizeBytes>\t<artifactSha256>\n
```

which for this hybrid gives
`b2dbee7f22de346d4f420824ffe652a43e3dfbb1f7177ec8359176aa68200633`. If maintainers have a different
convention, that value changes.
