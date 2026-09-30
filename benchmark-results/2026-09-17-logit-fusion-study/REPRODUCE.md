# Reproducing the logit-fusion control study

This guide lets a third party rerun every gate, pilot arm and full arm of the study on a clean
Ubuntu 24.04 host or a Mac, and check each number against the evidence we publish. The protocol,
with its hypotheses, arms, gates, decision rules and stop rule, is [PREREGISTRATION.md](PREREGISTRATION.md).
It was frozen before any fused output existed.

Provenance labels follow the working agreement. **[measured]** means we produced it on our own
harness. **[read]** means it comes from a document. **[believed]** means it is an estimate waiting to
be measured. As of this guide no study result exists: no gate has run on the pinned weights and no
pilot exists. The only real-weight runs so far are a mechanics smoke on a developer laptop with
non-study GGUFs (G0, G1 on 3 items at 64 tokens, G2 on 2 items at 12 steps); they check that the
code paths work and are not study evidence.

Everything inference-related runs in Java (Models `models-bench`, pure-Java backend by default, or
the optional Rust/FFM backend). Python is used only for three things: preparing the data, the
independent NumPy reference for gate G2, and an optional Hugging Face `transformers` cross-check.
None of them is in the measured path.

Contents:

1. [What is pinned](#1-what-is-pinned)
2. [Hardware](#2-hardware)
3. [Install: Ubuntu 24.04](#3-install-ubuntu-2404)
4. [Install: macOS](#4-install-macos)
5. [Check out and build Models](#5-check-out-and-build-models)
6. [Prepare the data](#6-prepare-the-data)
7. [Download the models](#7-download-the-models)
8. [Gates](#8-gates-g0g6)
9. [Pilot, S2](#9-pilot-s2)
10. [Tuning, calibration and freezing, S3](#10-tuning-calibration-and-freezing-s3)
11. [Full runs, S4 to S7](#11-full-runs-s4s7)
12. [Summaries, the stop rule and the results log](#12-summaries-stop-rule-results-log)
13. [Independent cross-checks](#13-independent-cross-checks)
14. [The host runner, `host-run.sh`](#14-the-host-runner-host-runsh)
15. [Report JSON: where each field comes from](#15-report-json-where-each-field-comes-from)
16. [Expected outputs per command](#16-expected-outputs-per-command)
17. [Pilot wall-time table (empty)](#17-pilot-wall-time-table)
18. [Open decisions](#18-open-decisions-flagged-not-amendments)

---

## 1. What is pinned

| Thing | Pin | Where |
|-------|-----|-------|
| Models code | `<MODELS_COMMIT>`, the exact 40-hex commit | Every report records it as `environment.modelsCommit`. Use the commit from the report you want to reproduce. For a fresh run, use the head of the study branch or the merge commit that contains this directory. |
| Protocol | `PREREGISTRATION.md` at its freezing commit | this directory |
| Datasets | Hugging Face dataset revisions plus source-file sha256, embedded in `prepare_fusion_data.py` | `data-manifest.json` holds the expected output sha256, item counts and `idsSha256` |
| Models | Hugging Face repo, revision, file, sha256 and size per GGUF | `models.json` |
| Arms | label → spec, thinking mode, model set, temperature, dependencies | `arms.json` |
| Prompts | bundled `fusion/prompts-v1.json` in `models-bench` | Each report records `config.decoding.promptsSha256`, plus a `renderedPromptSha256` per item |
| Frozen tuning | `frozen.json`, produced at S3 and committed before any test-set tuned run | Test runs must pass `--frozen` and `--frozen-sha256` |
| Python deps | `requirements-data.txt` (pyarrow), `requirements-reference.txt` (numpy), `requirements-hf.txt` (torch, transformers, numpy), all exact versions | this directory |

Dataset revisions [read from the Hugging Face API on 2026-09-17]:

| Dataset | Repo @ revision | Source file | Source sha256 |
|---------|-----------------|-------------|---------------|
| GSM8K test | `openai/gsm8k` @ `740312add88f781978c0658806c59bc2815b9866` | `main/test-00000-of-00001.parquet` | `ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59` |
| GSM8K train (500-item dev split, seed 20260917) | same | `main/train-00000-of-00001.parquet` | `ea82612ea9582142387730c793eb67d3b12849002bc0b7fa6f8efafa7351419d` |
| ARC-Challenge test | `allenai/ai2_arc` @ `210d026faf9955653af8916fad021475a3f00453` | `ARC-Challenge/test-00000-of-00001.parquet` | `62f03257e737aed263f55c6abf87c7bb0028a44a6bdd2a26eb1279eb42c1d1e9` |
| MATH-500 | `HuggingFaceH4/MATH-500` @ `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be` | `test.jsonl` | pinned in `prepare_fusion_data.py` |

Model sets used by the runner. Sizes and hashes are in `models.json`:

| Set | Members (name → `models.json` key) | Purpose |
|-----|-------------------------------------|---------|
| `q8` | A → `A-q8` (Qwen3 0.6B Q8_0), B → `B-q8` (1.7B Q8_0), C → `C-q8` (4B Q8_0), BIG → `BIG-q4` (8B Q4_K_M) | Primary arms and the memory-matched control |
| `q4` | A → `A-q4`, B → `B-q4`, C → `C-q4`, BIG → `BIG-q4` | H6 quantization arms |
| `big-q8` | BIG → `BIG-q8` (8B Q8_0) | H6, the BIG-Q8 side |

`A-q4` and `B-q4` have `status: proposed-substitute`. Qwen's official GGUF repositories publish no
Q4_K_M file for 0.6B or 1.7B [read from the HF API], so these pins come from another publisher and
may use a different quantization procedure. The runner refuses to download them unless
`ALLOW_PROPOSED_PINS=1`. That flag is only to be set once the substitute has been approved and the
approval is written in `RESULTS-LOG.md`. The `notes` field in `models.json` gives the alternatives.

## 2. Hardware

- **Recommended host [believed]:** an x86-64 bench host with 16 vCPU, one arm per host, AVX2 at a
  minimum and AVX-512 if available. The Vector API species is recorded in every report
  (`environment.vectorRuntime`), so hosts with different SIMD widths cannot be silently compared.
  Controls and treatments must run on the same host type. Otherwise compute comparisons (H3, H8 and
  the practical claim) are void.
- **RAM [believed, derived from `models.json` file sizes]:** GGUF weights are memory-mapped. A
  process needs roughly the summed file size of its members, plus a KV cache per member, plus the
  JVM heap. The KV cache at the default context of 10,240 tokens is several hundred MB per member;
  it depends on layers × KV heads × head dim.

  | Arm group | Weights | Suggested RAM |
  |-----------|---------|---------------|
  | single member A / B / C | 0.64 / 1.83 / 4.28 GB | 8 / 8 / 16 GB |
  | BIG (8B Q4_K_M) | 5.03 GB | 16 GB |
  | fused A+B+C at Q8_0 | ≈ 6.75 GB | 32 GB |
  | BIG-Q8 | 8.71 GB | 24 GB |
  | fused A+B+C at Q4 set | ≈ 4.0 GB | 16 GB |

  These are estimates to keep hosts from swapping. Actual peak RSS is measured per arm
  (`timings.peakRssBytes`) and replaces this table after the pilot.
- **Disk:** about 30 GB covers all GGUFs (≈ 25 GB summed from `models.json`), data, build and
  evidence. Gate G2 dumps `20 items × 50 steps × 3 members × 151,936 × 4 bytes` ≈ 1.8 GB of member
  logits, plus the same again for fused outputs. Budget 5 GB of scratch in `$RUN_ROOT/work`.
- **Contention:** fused arms run one worker thread per member (`--member-threads`), and each member
  also uses the Vector API internally. The core contention is part of what is measured. Do not run
  anything else on the host; `host-baseline.txt` records competing processes.

## 3. Install: Ubuntu 24.04

```bash
sudo apt-get update && sudo apt-get install -y git curl jq ca-certificates coreutils python3 unzip
# JDK 25 (Temurin)
sudo mkdir -p /opt/java
curl -fsSL 'https://api.adoptium.net/v3/binary/latest/25/ga/linux/x64/jdk/hotspot/normal/eclipse' -o /tmp/jdk25.tar.gz
sudo tar -xzf /tmp/jdk25.tar.gz -C /opt/java
sudo ln -sfn "$(find /opt/java -maxdepth 1 -type d -name 'jdk-25*' | sort | head -1)" /opt/java/current
export JAVA_HOME=/opt/java/current PATH=/opt/java/current/bin:$PATH
java -version            # must report 25; the output is recorded in every report
# uv (manages the Python venvs; pinned version)
curl -LsSf https://astral.sh/uv/0.11.25/install.sh | sh
```

The Adoptium URL resolves to the latest JDK 25 GA build. Each report records the exact
`java -version` output, so a later JDK update shows up as a configuration difference.

For the Rust/FFM backend only: `sudo apt-get install -y build-essential pkg-config`, then
`curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal`. The repository's
`rust-toolchain.toml` pins the toolchain.

## 4. Install: macOS

```bash
brew install git jq uv
# JDK 25: either Homebrew …
brew install --cask temurin@25
export JAVA_HOME="$(/usr/libexec/java_home -v 25)"
# … or SDKMAN
curl -s https://get.sdkman.io | bash && sdk install java 25-tem
java -version
```

macOS runs are fine for reproducing gates and small pilots. Reports record `osName`, `cpuModel` and
the CPU feature flags (read through `sysctl`). The protocol's compute comparisons use one host type
throughout, so do not mix macOS and Linux reports within a comparison. On macOS, use `shasum -a 256`
wherever this guide says `sha256sum`.

## 5. Check out and build Models

```bash
git clone https://github.com/integrallis/models.git
cd models
git switch --detach <MODELS_COMMIT>
test -z "$(git status --short)"          # reports record modelsDirty; a dirty tree is not a reproduction
./gradlew :models-bench:installDist                          # pure-java backend
# ./gradlew :models-bench:installDist -PmodelsBenchNative=true # adds the rust-ffm backend
export FUSION=$PWD/models-bench/build/install/models-bench/bin/models-bench
export STUDY=$PWD/benchmark-results/2026-09-17-logit-fusion-study
export REV=$(git rev-parse HEAD)
export JAVA_OPTS="-Xms2g -Xmx12g -XX:+UseG1GC -XX:+AlwaysPreTouch"
```

The launcher already carries `--add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED`.
Every JVM flag and `models.*` / `vectors.*` system property is recorded in the report under
`environment.jvmInputArguments` and `environment.inferenceSystemProperties`.

## 6. Prepare the data

```bash
uv venv --python 3.12 .venv-data
uv pip install --python .venv-data/bin/python -r "$STUDY/requirements-data.txt"
.venv-data/bin/python -m unittest discover -s "$STUDY" -p 'prepare_fusion_data_test.py'   # no network
.venv-data/bin/python "$STUDY/prepare_fusion_data.py" --out-dir data --cache-dir hf-cache
```

The script downloads the pinned revisions from
`https://huggingface.co/datasets/<repo>/resolve/<revision>/<file>` and refuses any source file whose
sha256 differs from its pin. It then writes `gsm8k-test.jsonl` (1,319 items), `gsm8k-dev.jsonl`
(500 items), `arc-challenge-test.jsonl` (1,172 items), `math500-test.jsonl` (500 items) and
`data/data-manifest.json`. The dev split is
`sorted(random.Random(20260917).sample(range(len(train)), 500))`.

Verify that the outputs are byte-identical to the committed expectation:

```bash
for name in $(jq -r '.datasets | keys[]' "$STUDY/data-manifest.json"); do
  file=data/$(jq -r --arg n "$name" '.datasets[$n].output' "$STUDY/data-manifest.json")
  want=$(jq -r --arg n "$name" '.datasets[$n].outputSha256' "$STUDY/data-manifest.json")
  got=$(sha256sum "$file" | cut -d' ' -f1)
  [ "$want" = "$got" ] && echo "OK $name" || echo "MISMATCH $name want $want got $got"
done
```

## 7. Download the models

```bash
download_set() { # q8 | q4 | big-q8
  case "$1" in q8) keys="A-q8 B-q8 C-q8 BIG-q4";; q4) keys="A-q4 B-q4 C-q4 BIG-q4";; big-q8) keys="BIG-q8";; esac
  mkdir -p gguf
  for key in $keys; do
    entry=$(jq -ec --arg k "$key" '.models[] | select(.key==$k)' "$STUDY/models.json")
    status=$(jq -r .status <<<"$entry")
    if [ "$status" != official ] && [ "${ALLOW_PROPOSED_PINS:-0}" != 1 ]; then echo "SKIP $key ($status)"; continue; fi
    dest="gguf/$key--$(jq -r .file <<<"$entry")"
    [ -f "$dest" ] || curl -fL -o "$dest" "$(jq -r .url <<<"$entry")"
    want=$(jq -r .sha256 <<<"$entry"); got=$(sha256sum "$dest" | cut -d' ' -f1)
    [ "$want" = "$got" ] && echo "OK $key" || { echo "HASH MISMATCH $key"; rm -f "$dest"; }
  done
}
download_set q8
Q8="A=$PWD/gguf/A-q8--Qwen3-0.6B-Q8_0.gguf,B=$PWD/gguf/B-q8--Qwen3-1.7B-Q8_0.gguf,C=$PWD/gguf/C-q8--Qwen3-4B-Q8_0.gguf"
BIGQ4="BIG=$PWD/gguf/BIG-q4--Qwen3-8B-Q4_K_M.gguf"
```

`models.json` also carries a `memoryCheck` object, which compares the summed Q8_0 member bytes with
the BIG Q4_K_M bytes. The protocol requires that check before the run.

## 8. Gates (G0–G6)

A failed gate stops the stage. It is never a reason to adjust a result. Gates that load weights (G1,
G2) take `--backend pure-java --models-revision "$REV"`; G0 reads GGUF metadata only.

```bash
# G0: tokenizer identity from GGUF metadata (vocabulary, merges, token types, scores, special-token
# ids and the derived special-token list); the chat template hash is reported but not gated
$FUSION logit-fusion gate g0 --members "$Q8,$BIGQ4" --report gate-g0.json

# G1: e_i weights reproduce member i alone, token for token, at T=0, for every member, on 20 items
$FUSION logit-fusion gate g1 --members "$Q8" --dataset gsm8k --data data/gsm8k-test.jsonl \
  --limit 20 --max-tokens 1024 --thinking off --backend pure-java --models-revision "$REV" --report gate-g1.json

# G2: dump member logits and Java's fused outputs, then recompute independently in NumPy
$FUSION logit-fusion gate g2-dump --members "$Q8" --dataset gsm8k --data data/gsm8k-test.jsonl \
  --limit 20 --steps 50 --out g2-dump --report gate-g2-dump.json --backend pure-java --models-revision "$REV"
uv venv --python 3.12 .venv-ref && uv pip install --python .venv-ref/bin/python -r "$STUDY/requirements-reference.txt"
.venv-ref/bin/python "$STUDY/reference_fusion.py" --dump g2-dump --tolerance 1e-4 --report gate-g2.json

# G3: flags a member-argmax agreement rate of exactly 100% (a switch that never switches)
$FUSION logit-fusion gate g3 --reports pilot-gsm8k-F-uniform.json,pilot-gsm8k-F-article.json --report gate-g3.json

# G4: gold answers rendered in the answer format must score 100%, per dataset
for d in gsm8k:gsm8k-test gsm8k:gsm8k-dev arc:arc-challenge-test math500:math500-test; do
  $FUSION logit-fusion gate g4 --dataset "${d%%:*}" --data "data/${d#*:}.jsonl" --report "gate-g4-${d#*:}.json"
done

# G5: truncation and extraction-failure rates per arm; >3% truncation means rerun with a higher cap
$FUSION logit-fusion gate g5 --reports "$(ls pilot-gsm8k-*.json | paste -sd, -)" --report gate-g5.json

# G6: reasoning and stated answer extractors on the 90 bundled hand-built traces (100% required)
$FUSION logit-fusion gate g6 --report gate-g6.json
```

G7 (calibration fitted on the dev split only, with ECE before and after) is part of `calibrate` in
section 10. G3 and G5 read arm reports, so they run after the pilot and after each full run.

## 9. Pilot (S2)

The pilot runs 50 GSM8K test items per arm, for every arm in `arms.json` (all protocol arms except
FRONTIER; ORACLE is derived). **Tuning happens at S3, after the pilot**, so the pilot runs the
protocol's own specs (`tuned`, `best`) with `--pilot true`, and the Java runner makes three
substitutions. Each report records them in `config.arm.pilotSubstitution`, and the pilot summary
must state them:

- `tuned` weights become `uniform`, which makes pilot F-tuned identical to F-uniform;
- the vote tie-break `best` becomes member B, because the dev-split best member is not chosen yet;
- VOTE-conf runs with uncalibrated confidence (temperature 1 per member).

Without `--pilot true` (and without `--frozen`) the CLI refuses every arm that needs S3 outputs.

Order matters: VOTE, VOTE-think, VOTE-consist, VOTE-conf, RERANK and RERANK-think reuse the member
reports listed in `dependsOn`. Run A, B, C, A-think, B-think and C-think first.

Direct CLI, one line per arm. Non-thinking arms use `--max-tokens 1024`; thinking arms use
`--thinking on --max-tokens 8192`. The protocol fixes the thinking cap per dataset from the pilot's
p95 trace length, so 8192 is only the pilot cap.

```bash
P="--dataset gsm8k --data data/gsm8k-test.jsonl --data-manifest data/data-manifest.json --limit 50 --seed 20260917 --context-length 10240 --backend pure-java --models-revision $REV"
A=${Q8%%,*}; B=$(cut -d, -f2 <<<"$Q8"); C=$(cut -d, -f3 <<<"$Q8")
OFF="--thinking off --temperature 0 --max-tokens 1024"; ON="--thinking on --temperature 0 --max-tokens 8192"

$FUSION logit-fusion run --arm member:A   --arm-label A   --members "$A"     $OFF $P --report pilot-gsm8k-A.json
$FUSION logit-fusion run --arm member:B   --arm-label B   --members "$B"     $OFF $P --report pilot-gsm8k-B.json
$FUSION logit-fusion run --arm member:C   --arm-label C   --members "$C"     $OFF $P --report pilot-gsm8k-C.json
$FUSION logit-fusion run --arm member:BIG --arm-label BIG --members "$BIGQ4" $OFF $P --report pilot-gsm8k-BIG.json
$FUSION logit-fusion run --arm member:A   --arm-label A-think   --members "$A"     $ON $P --report pilot-gsm8k-A-think.json
$FUSION logit-fusion run --arm member:B   --arm-label B-think   --members "$B"     $ON $P --report pilot-gsm8k-B-think.json
$FUSION logit-fusion run --arm member:C   --arm-label C-think   --members "$C"     $ON $P --report pilot-gsm8k-C-think.json
$FUSION logit-fusion run --arm member:BIG --arm-label BIG-think --members "$BIGQ4" $ON $P --report pilot-gsm8k-BIG-think.json

PP="$P --pilot true"
$FUSION logit-fusion run --arm fuse:A+B:poe:tuned     --arm-label AB --members "$A,$B" $OFF $PP --report pilot-gsm8k-AB.json
$FUSION logit-fusion run --arm fuse:A+C:poe:tuned     --arm-label AC --members "$A,$C" $OFF $PP --report pilot-gsm8k-AC.json
$FUSION logit-fusion run --arm fuse:B+C:poe:tuned     --arm-label BC --members "$B,$C" $OFF $PP --report pilot-gsm8k-BC.json
$FUSION logit-fusion run --arm fuse:A+B+C:poe:uniform --arm-label F-uniform --members "$Q8" $OFF $P --report pilot-gsm8k-F-uniform.json
$FUSION logit-fusion run --arm fuse:A+B+C:poe:tuned   --arm-label F-tuned   --members "$Q8" $OFF $PP --report pilot-gsm8k-F-tuned.json
$FUSION logit-fusion run --arm fuse:A+B+C:mixture:tuned --arm-label F-mix   --members "$Q8" $OFF $PP --report pilot-gsm8k-F-mix.json
$FUSION logit-fusion run --arm fuse:A+B+C:article:0.5,0.2,0.3 --arm-label F-article --members "$Q8" $OFF $P --report pilot-gsm8k-F-article.json

# Vote arms reuse member outputs and load no weights; --members is not needed.
$FUSION logit-fusion run --arm vote:A+B+C:best --arm-label VOTE $OFF $PP \
  --member-reports pilot-gsm8k-A.json,pilot-gsm8k-B.json,pilot-gsm8k-C.json --report pilot-gsm8k-VOTE.json
for arm in vote:VOTE-think vote-consist:VOTE-consist vote-conf:VOTE-conf; do
  $FUSION logit-fusion run --arm "${arm%%:*}:A+B+C:best" --arm-label "${arm#*:}" $ON $PP \
    --member-reports pilot-gsm8k-A-think.json,pilot-gsm8k-B-think.json,pilot-gsm8k-C-think.json --report "pilot-gsm8k-${arm#*:}.json"
done
# Rerank reuses member outputs for candidates but loads every member to score them.
$FUSION logit-fusion run --arm rerank:A+B+C:tuned --arm-label RERANK --members "$Q8" $OFF $PP \
  --member-reports pilot-gsm8k-A.json,pilot-gsm8k-B.json,pilot-gsm8k-C.json --report pilot-gsm8k-RERANK.json
$FUSION logit-fusion run --arm rerank:A+B+C:tuned --arm-label RERANK-think --members "$Q8" $ON $PP \
  --member-reports pilot-gsm8k-A-think.json,pilot-gsm8k-B-think.json,pilot-gsm8k-C-think.json --report pilot-gsm8k-RERANK-think.json

SC_K=2   # placeholder [believed]; re-set from pilot core-seconds so SC-k compute is within ±10% of F-tuned
$FUSION logit-fusion run --arm sc:C:$SC_K --arm-label SC-k       --members "$C" --thinking off --temperature 0.7 --max-tokens 1024 $P --report pilot-gsm8k-SC-k.json
$FUSION logit-fusion run --arm sc:C:$SC_K --arm-label SC-k-think --members "$C" --thinking on  --temperature 0.7 --max-tokens 8192 $P --report pilot-gsm8k-SC-k-think.json

# H6 arms (the q4 set needs approved substitute pins; BIG-Q8 needs the big-q8 set)
Q4="A=$PWD/gguf/A-q4--Qwen3-0.6B-Q4_K_M.gguf,B=$PWD/gguf/B-q4--Qwen3-1.7B-Q4_K_M.gguf,C=$PWD/gguf/C-q4--Qwen3-4B-Q4_K_M.gguf"
$FUSION logit-fusion run --arm fuse:A+B+C:poe:tuned --arm-label F-tuned-Q4 --members "$Q4" $OFF $PP --report pilot-gsm8k-F-tuned-Q4.json
$FUSION logit-fusion run --arm member:BIG --arm-label BIG-Q4 --members "$BIGQ4" $OFF $P --report pilot-gsm8k-BIG-Q4.json
$FUSION logit-fusion run --arm member:BIG --arm-label BIG-Q8 --members "BIG=$PWD/gguf/BIG-q8--Qwen3-8B-Q8_0.gguf" $OFF $P --report pilot-gsm8k-BIG-Q8.json
```

On a bench host the same pilot is `MODELS_COMMIT=<sha> bash host-run.sh pilot <label>` for each
label in `arms.json`. The runner passes `--pilot true` and wires the `dependsOn` reports itself;
see section 14.

## 10. Tuning, calibration and freezing (S3)

Everything in this section uses only the dev split (`gsm8k-dev.jsonl`, GSM8K train rows). Nothing
may touch test items.

```bash
D="--dataset gsm8k --data data/gsm8k-dev.jsonl --data-manifest data/data-manifest.json --seed 20260917 --context-length 10240 --backend pure-java --models-revision $REV"
# 1. Member outputs on dev: best-member choice (non-thinking) and calibration inputs (thinking)
for m in A B C; do
  $FUSION logit-fusion run --arm member:$m --arm-label $m --members "$(tr , '\n' <<<"$Q8" | grep "^$m=")" \
    --thinking off --temperature 0 --max-tokens 1024 $D --report dev-gsm8k-dev-$m.json
  $FUSION logit-fusion run --arm member:$m --arm-label $m-think --members "$(tr , '\n' <<<"$Q8" | grep "^$m=")" \
    --thinking on --temperature 0 --max-tokens 8192 $D --report dev-gsm8k-dev-$m-think.json
done
# 2. Weight grids on the simplex, step 0.1: one per fused arm that uses tuned weights
#    Each grid point is a full fused run over the 500 dev items, written to --work-dir and reused
#    if present (66 points for three members, 11 for a pair). Ties -> closest to uniform.
$FUSION logit-fusion tune-weights --members "$Q8" --arm-members A+B+C --rule poe     --grid-step 0.1 --work-dir tune-ABC-poe     --thinking off --max-tokens 1024 --temperature 0 $D --report tuning-ABC-poe.json
$FUSION logit-fusion tune-weights --members "$Q8" --arm-members A+B+C --rule mixture --grid-step 0.1 --work-dir tune-ABC-mixture --thinking off --max-tokens 1024 --temperature 0 $D --report tuning-ABC-mixture.json
#    and likewise --arm-members A+B, A+C, B+C with --rule poe
# 3. G7: per-member confidence temperature from dev thinking outputs; ECE before and after
$FUSION logit-fusion calibrate --dataset gsm8k --member-reports dev-gsm8k-dev-A-think.json,dev-gsm8k-dev-B-think.json,dev-gsm8k-dev-C-think.json --report calibration.json
# 4. Freeze weights, calibration and the dev best member into one file (prints its sha256)
$FUSION logit-fusion freeze --tuning tuning-ABC-poe.json,tuning-ABC-mixture.json,tuning-AB-poe.json,tuning-AC-poe.json,tuning-BC-poe.json \
  --calibration calibration.json --dataset gsm8k --data data/gsm8k-dev.jsonl --data-manifest data/data-manifest.json \
  --best-member-reports dev-gsm8k-dev-A.json,dev-gsm8k-dev-B.json,dev-gsm8k-dev-C.json --report frozen.json
sha256sum frozen.json
```

**Rule:** commit `frozen.json` to this directory before any test-set run of a tuned arm (AB, AC, BC,
F-tuned, F-mix, VOTE-conf, RERANK, RERANK-think, F-tuned-Q4). Every such run must pass
`--frozen frozen.json --frozen-sha256 <sha>`. The CLI refuses a tuned arm without them, or when the
hash does not match. The CLI also checks that no test item id appears in the frozen dev ids. For
full runs, the VOTE tie-break member is the best member recorded in `frozen.json`, not B.

## 11. Full runs (S4–S7)

These are the same commands as the pilot, without `--limit` and with the frozen file for tuned
arms. The datasets, in protocol order:

| Stage | Dataset flag | Data file | Notes |
|-------|--------------|-----------|-------|
| S4 | `--dataset arc` | `data/arc-challenge-test.jsonl` | `--arc-scoring generative` (letter answer) or `--arc-scoring loglik` (fused log-likelihood over the options); run both |
| S5 | `--dataset gsm8k` | `data/gsm8k-test.jsonl` | primary generative set |
| S6 | `--dataset math500` | `data/math500-test.jsonl` | normalized answer match |
| S7 | `--dataset generic` | post-cutoff set (not yet pinned) | members, F-tuned and BIG only; directional |

Example: F-tuned on GSM8K.

```bash
F="--frozen $STUDY/frozen.json --frozen-sha256 $(sha256sum "$STUDY/frozen.json" | cut -d' ' -f1)"
$FUSION logit-fusion run --arm fuse:A+B+C:poe:tuned --arm-label F-tuned --members "$Q8" \
  --dataset gsm8k --data data/gsm8k-test.jsonl --data-manifest data/data-manifest.json \
  --thinking off --temperature 0 --max-tokens 1024 --seed 20260917 --context-length 10240 \
  --backend pure-java --models-revision "$REV" $F --report full-gsm8k-F-tuned.json
```

The protocol's robustness check (T = 0.7, three seeds) repeats the generative arms with
`--temperature 0.7 --seed 20260917`, `--seed 20260918` and `--seed 20260919`. It never enters a
decision rule.

On a bench host: `MODELS_COMMIT=<sha> FROZEN=<path> FROZEN_SHA256=<sha> bash host-run.sh arm <gsm8k|arc|math500> <label>`.

## 12. Summaries, stop rule, results log

```bash
$FUSION logit-fusion summarize --reports "$(ls pilot-gsm8k-*.json | paste -sd, -)" \
  --member-labels A,B,C --fused-label F-tuned --report pilot-summary.json
```

The summary gives, per arm, accuracy, tokens/s, peak RSS, core-seconds, truncation rate and p95
trace length. It also gives ORACLE headroom (items any member got right, minus the best member) and
the stop-rule check. The pilot stops before the full runs if **both** hold: ORACLE headroom is under
3 points, and F-tuned is at or below the best member. At the pilot, "F-tuned" is the uniform-weight
stand-in, and the summary must say so.

Record every run in [RESULTS-LOG.md](RESULTS-LOG.md) (append only) with
`$FUSION logit-fusion results-log --in results-log.jsonl --report RESULTS-LOG.generated.md`.

## 13. Independent cross-checks

- **`reference_fusion.py` (gate G2).** A NumPy float64 reimplementation of the poe, mixture and
  article rules over the float32 member logits dumped by Java. It passes when the maximum absolute
  difference is at most `1e-4` on both the raw fused score and the normalized fused log-probability.
  It shares no code with the Java path.
- **`hf_reference_fusion.py` (optional, not a gate).** A short `transformers` script that fuses
  `Qwen/Qwen3-0.6B` and `Qwen/Qwen3-1.7B` safetensors checkpoints at pinned revisions, for one GSM8K
  item. It prints fused tokens and per-token member agreement:

  ```bash
  uv venv --python 3.12 .venv-hf && uv pip install --python .venv-hf/bin/python -r "$STUDY/requirements-hf.txt"
  .venv-hf/bin/python "$STUDY/hf_reference_fusion.py" --data data/gsm8k-test.jsonl --item-id gsm8k-test-0000 --rule poe --weights 0.5,0.5
  ```

  It checks the fusion rule, not our numbers. The checkpoints are bf16 while the study uses Q8_0
  GGUFs, so token identity with the Java run is not expected. It is never in the measured path.

## 14. The host runner (`host-run.sh`)

The runner targets a fresh Ubuntu 24.04 x86-64 host, as root. It reads `models.json`, `arms.json`
and `data-manifest.json` from the checkout at `MODELS_COMMIT`, never from the launching machine.

| Env var | Default | Meaning |
|---------|---------|---------|
| `MODELS_COMMIT` | required | Models commit to clone and build; must be clean |
| `RUN_ROOT` | `/opt/modeljars-runs/logit-fusion-20260917` | root for checkout, store, data, work, evidence |
| `BACKEND` | `pure-java` | or `rust-ffm` (builds with `-PmodelsBenchNative=true`, records the native library hash) |
| `MEMBER_THREADS` | member count | worker threads for fused decoding |
| `JAVA_OPTS` | `-Xms2g -Xmx12g -XX:+UseG1GC -XX:+AlwaysPreTouch` | JVM flags (recorded) |
| `PILOT_LIMIT` | `50` | pilot items |
| `SC_K` | `2` | SC-k sample count, a placeholder until the pilot measures compute |
| `ALLOW_PROPOSED_PINS` | `0` | `1` permits `proposed-substitute` GGUFs, after approval only |
| `FROZEN`, `FROZEN_SHA256` | unset | required for full runs of tuned arms |
| `MAX_TOKENS_OFF` / `MAX_TOKENS_THINK` | `1024` / `8192` | output caps |
| `CONTEXT_LENGTH` | `10240` | per-member KV capacity |
| `ARC_SCORING` | `generative` | or `loglik` (reports get a `-loglik` suffix) |
| `PHASE_TIMEOUT` | `36h` | watchdog on every Java invocation |
| `HF_TOKEN` | unset | optional, for rate limits |

```bash
export MODELS_COMMIT=<sha>
bash host-run.sh bootstrap            # JDK 25, uv, clone, installDist, host-baseline.txt, java-version.txt
bash host-run.sh data                 # prepare + verify against data-manifest.json
bash host-run.sh models q8            # A-q8 B-q8 C-q8 BIG-q4 (hash + size checked)
bash host-run.sh models big-q8        # BIG-q8
ALLOW_PROPOSED_PINS=1 bash host-run.sh models q4   # only after the substitute pins are approved
bash host-run.sh gates                # G0, G1, G2 (+ NumPy), G4 ×4, G6
bash host-run.sh pilot A              # … one per arms.json label; dependencies first
bash host-run.sh summarize pilot gsm8k
bash host-run.sh tune                 # S3 on the dev split → evidence/frozen.json (commit it!)
FROZEN=… FROZEN_SHA256=… bash host-run.sh arm arc F-tuned
bash host-run.sh results-log
```

Monitoring: `grep -E 'PHASE-(DONE|FAILED)' $RUN_ROOT/evidence/host-run.log`. Each phase appends a
JSON line to `evidence/results-log.jsonl`. Pilot lines carry `pilotSubstitution`. Each phase then
rewrites `evidence/SHA256SUMS` over the evidence directory. Reports are written as `<mode>-<dataset>-<label>.json`,
with mode `pilot`, `full` or `dev`, and an existing report is never overwritten, so phases can be
resumed.

## 15. Report JSON: where each field comes from

All Java reports share one versioned schema, `schemaVersion: 1`. A unit test in `models-bench`
fails the build if a required field is missing.

| Field | Source |
|-------|--------|
| `schemaVersion`, `kind`, `createdAt` | report writer; `kind` is `arm`, `gate-g0`…, `tuning`, `calibration`, `frozen` or `summary` |
| `config.arm.spec`, `.label`, `.kind`, `.members`, `.rule`, `.weights`, `.tieBreakMember`, `.samples` | parsed `--arm` / `--arm-label`; `tuned` weights resolved from the frozen file |
| `config.arm.frozenPath`, `.frozenSha256` | `--frozen`, and the sha256 of that file as verified against `--frozen-sha256` |
| `config.decoding.temperature`, `.topK`, `.topP`, `.seeds`, `.maxTokens` | CLI flags (defaults applied); `seeds` lists every seed used (SC-k: seed..seed+k−1) |
| `config.decoding.thinking`, `.chatTemplate`, `.chatTemplateSha256` | `--thinking`, mapped to the Qwen3 ChatML template (thinking on: plain assistant turn; off: the empty `<think></think>` block Qwen3's template emits for `enable_thinking=false`) |
| `config.decoding.promptsSha256` | sha256 of the prompts JSON (bundled `prompts-v1.json` or `--prompts`) |
| `config.decoding.memberThreads`, `.contextLength`, `.tokenLogEvery`, `.arcScoring`, `.rerankScore` | CLI flags |
| `config.members[].path`, `.sha256`, `.sizeBytes` | `--members`; sha256 computed over the file at load time |
| `config.members[].tokenizerSha256` | hash over the GGUF metadata `tokenizer.ggml.*` vocabulary, merges, token types and special-token ids (what G0 compares) |
| `config.members[].chatTemplateSha256`, `.architecture`, `.ggufFileType`, `.vocabularySize`, `.tokenizerFields` | GGUF metadata (`tokenizer.chat_template`, `general.architecture`, `general.file_type`, token count, per-field tokenizer hashes) |
| `config.members[].diagnostics`, `.loadMillis` | the loaded backend's `BackendDiagnostics` (kernel plan, environment) and its load time |
| `config.dataset.path`, `.sha256`, `.itemCount`, `.offset`, `.limit`, `.selectedIdsSha256` | `--data` file, `--offset` / `--limit`, and the sha256 of the selected ids joined by `\n` |
| `config.dataset.hfRepo`, `.hfRevision`, `.devIdsSha256` | `--data-manifest` (`data-manifest.json` written by `prepare_fusion_data.py`) |
| `config.backend.name`, `.nativeLibraryPath`, `.nativeLibrarySha256` | `--backend`; for rust-ffm, the loaded native kernel library and its sha256 |
| `environment.modelsCommit`, `.modelsDirty` | `git rev-parse HEAD` / `git status --porcelain` in the checkout |
| `environment.modelsRevisionArgument` | `--models-revision` (the host runner passes `MODELS_COMMIT`) |
| `environment.javaVersionOutput` | output of `java -version` for the running JVM binary |
| `environment.jvmInputArguments` | `RuntimeMXBean.getInputArguments()` |
| `environment.inferenceSystemProperties` | system properties with prefixes `models.`, `vectors.`, `jdk.incubator.vector`, `java.util.concurrent.ForkJoinPool`, `java.vm.`, `java.version`, `java.vendor` |
| `environment.vectorRuntime` | `VectorUtil.runtimeCapabilities()` from vectors-core (Vector API species, kernel selection) |
| `environment.availableProcessors` | `Runtime.availableProcessors()` |
| `environment.host.cpuModel`, `.cpuFlags` | `/proc/cpuinfo` `model name` and `flags` (Linux); `sysctl machdep.cpu.brand_string` and feature keys (macOS) |
| `environment.host.logicalCores`, `.ramBytes`, `.osName`, `.osVersion`, `.arch`, `.hostname` | `nproc --all` / `sysctl hw.logicalcpu`, the JVM OS MXBean, system properties |
| `timings.wallMillis` | monotonic clock around the whole run |
| `timings.processCpuSeconds` | `OperatingSystemMXBean.getProcessCpuTime()` delta (core-seconds) |
| `timings.peakRssBytes`, `.peakRssMethod` | `/proc/self/status` `VmHWM` (Linux, `linux-VmHWM`); `ps -o rss` sampled every 2 s elsewhere (`sampled-ps-rss-2s`, can miss spikes) |
| `timings.loadMillis` | summed member load time |
| `summary.*` | derived from `items`: accuracy, truncation and extraction-failure rates, tokens/s (generated tokens over generation wall time, reused outputs excluded), core-seconds, reused core-seconds, p95 trace tokens, per-aggregation accuracy (majority, consist, conf, oracle), per-member reasoning/answer mismatch rate |
| `summary.g3` | fused arms only: steps, all-member argmax agreement rate, per-member fused-equals-argmax rate, fused-equals-no-member rate, mean fused entropy, mean KL(fused‖memberᵢ), `flagged` when agreement is exactly 100% |
| `items[].renderedPromptSha256` | sha256 of the exact rendered prompt text (template plus instruction plus question) |
| `items[].outputs[].text`, `.tokenIds` | generated text and token ids (full, unsampled) |
| `items[].outputs[].reasoningAnswer`, `.statedAnswer`, `.consistent` | reasoning-aware extractor: `<think>` split, a_r, a_u (rules validated by G6) |
| `items[].outputs[].confidence` | mean log-probability of the answer tokens under the producing member |
| `items[].outputs[].memberForwardMillis` | wall time spent in each member's prefill and forward |
| `items[].outputs[].tokenLogProbabilities` | per generated token, log-probability under the producing distribution (member, or fused at T = 1) |
| `items[].outputs[].agreement` | G3 counters for that fused output, over every step |
| `items[].outputs[].tokenLog[]` | fused outputs, every `--token-log-every` steps: member argmaxes, which member argmaxes equal the fused token, fused entropy, KL(fused‖memberᵢ), fused token log-probability |
| `items[].aggregations`, `.aggregationCorrect`, `.candidates` | vote aggregations (majority, consist, conf) and ORACLE; rerank / ARC log-likelihood candidates with per-member and fused scores |
| `items[].coreSeconds`, `.reusedCoreSeconds` | process CPU-time delta around the item; core-seconds of reused member outputs |

## 16. Expected outputs per command

Every command's last stdout line is `PASS <what>`, `FAIL <what>` or `DONE <what>`. Exit codes: 0
for PASS or DONE, 1 for FAIL, 2 for a usage error.

| Command | Writes | Passing condition |
|---------|--------|-------------------|
| `prepare_fusion_data.py` | 4 JSONL files + `data-manifest.json` | output sha256 values equal the committed `data-manifest.json` |
| `gate g0` | `gate-g0.json` (per-member field hashes) | vocabulary, merges and special tokens identical across all members; a mismatch names the field |
| `gate g1` | `gate-g1.json` (per member, per item, full token id sequences) | every sequence identical, 20 items × every member |
| `gate g2-dump` + `reference_fusion.py` | dump dir + `gate-g2.json` | max abs diff ≤ 1e-4, every item, step and rule |
| `gate g3` | `gate-g3.json` | reports agreement rates; a rate of exactly 100% is flagged |
| `gate g4` | `gate-g4-<dataset>.json` | 100% of gold answers score correct |
| `gate g5` | `gate-g5.json` | truncation ≤ 3% per arm (otherwise rerun with a higher cap) |
| `gate g6` | `gate-g6.json` | 90/90 fixtures (30 per dataset) |
| `run` | arm report (section 15) | `DONE`; accuracy is a result, not a pass condition |
| `tune-weights` | per-point reports in `--work-dir`, plus `tuning-*.json` | `DONE` |
| `calibrate` | `calibration.json` with per-member temperature and ECE before/after | `DONE` |
| `freeze` | `frozen.json`; its sha256 printed | `DONE` |
| `summarize` | summary JSON with the stop-rule evaluation | `DONE` |

## 17. Pilot wall-time table

**To be filled from pilot measurements; nothing here is measured yet.** Fill it from
`pilot-summary.json` and name the host type and commit. Until then, the only cost figures are the
protocol's [believed] envelope: low tens of fused tokens/s, and 5–10 hours per fused arm per
generative dataset.

Host type: ______ · Models commit: ______ · Backend: ______ · Date: ______

| arm | items | wall time | tokens/s | peak RSS | core-seconds |
|-----|-------|-----------|----------|----------|--------------|
| A | 50 | | | | |
| B | 50 | | | | |
| C | 50 | | | | |
| BIG | 50 | | | | |
| A-think | 50 | | | | |
| B-think | 50 | | | | |
| C-think | 50 | | | | |
| BIG-think | 50 | | | | |
| AB | 50 | | | | |
| AC | 50 | | | | |
| BC | 50 | | | | |
| F-uniform | 50 | | | | |
| F-tuned (uniform stand-in) | 50 | | | | |
| F-mix (uniform stand-in) | 50 | | | | |
| F-article | 50 | | | | |
| VOTE | 50 | | | | |
| VOTE-think | 50 | | | | |
| VOTE-consist | 50 | | | | |
| VOTE-conf (uncalibrated) | 50 | | | | |
| RERANK (uniform stand-in) | 50 | | | | |
| RERANK-think (uniform stand-in) | 50 | | | | |
| SC-k (k = __) | 50 | | | | |
| SC-k-think (k = __) | 50 | | | | |
| F-tuned-Q4 (uniform stand-in) | 50 | | | | |
| BIG-Q4 | 50 | | | | |
| BIG-Q8 | 50 | | | | |

## 18. Open decisions (flagged, not amendments)

These are places where the implementation had to choose something the protocol text does not pin
down, or words differently. Each needs an explicit decision, recorded as a dated amendment to
`PREREGISTRATION.md` before any test-set result is read.

- **MATH-500 equivalence.** The protocol names "a sympy-equivalence checker". The Java runner uses
  a port of the Hendrycks MATH `strip_string` normaliser plus exact rational equality of plain
  numbers and integer fractions, unit-tested to 100% on all 500 gold answers (G4) and on the G6
  fixtures. It does not do symbolic simplification, so it can score a correct but differently
  written expression as wrong. Options: accept the normaliser as the scorer, or add a post-hoc
  sympy re-scoring of the saved outputs as a secondary column.
- **Thinking-mode rerank.** RERANK-think takes its candidates from thinking-mode outputs but scores
  them with the non-thinking prompt (`config.decoding.rerankPromptThinking = off`), because a
  thinking prompt would require scoring a reasoning trace, not an answer.
- **Rerank score.** Candidates are scored by summed log-likelihood of the rendered answer
  continuation (`--rerank-score sum`); `mean` is available. The protocol says "log-likelihood".
- **ARC log-likelihood prompt.** The option-scoring prompt ends with the assistant prefix `Answer:`
  and scores the continuation ` <label>`; see `prompts-v1.json`.
- **Vote ties.** Ties go to the tied answer that contains the best member's vote, then the arm's
  member order. SC-k ties go to the earliest sample.
- **Best member at the pilot.** The stop-rule summary uses the best member on the pilot items; the
  protocol chooses the best member on the dev split at S3.
- **Tuning cost.** Each grid point is a full fused run over 500 dev items (66 points for three
  members). The pilot's measured cost should decide whether this is affordable as written.
