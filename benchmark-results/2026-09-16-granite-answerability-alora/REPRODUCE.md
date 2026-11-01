# Reproducing the Granite 4.1 3B answerability aLoRA (pilot 2): training, labels, qualification

This guide rebuilds, from public inputs, every number used to qualify the Integrallis answerability
aLoRA. Java is the system of record: qualification runs on the Models runtime. Python appears only
where a third-party toolchain is the reference (training with PEFT, the PEFT reference evaluation)
or for small data-preparation and label-audit scripts.

Throughout, labels mark how each fact is known:
- **measured**: recorded in the evidence files named;
- **read**: taken from a published card or terms page;
- **believed**: not verified here.

## 0. Pinned inputs

| Input | Pin |
|---|---|
| Base GGUF (runtime) | `ibm-granite/granite-4.1-3b-GGUF` @ `ab4701481089b58a082ef63cc1cee738887293ff`, `granite-4.1-3b-Q4_K_M.gguf`, sha256 `662b0626…eb29` |
| Base HF weights and tokenizer (training, reference) | `ibm-granite/granite-4.1-3b` @ `c0650403e44e78ec0262dab1c90914c65b196c4e` |
| QuAC train | `https://s3.amazonaws.com/my89public/quac/train_v0.2.json`, sha256 `ff5cca5a2e4b4d1cb5b5ced68b9fce88394ef6d93117426d6d4baafbcc05c56a` |
| SQuAD v2 train | `rajpurkar/squad_v2` @ `3ffb306f725f7d2ce8394bc1873b24868140c412`, `squad_v2/train-00000-of-00001.parquet`, sha256 `f6da32ffb482ff463ad056477740d1bb284b96a45db3a08bee6a225ca6abf291` |
| MS MARCO v2.1 train shard 0 | `microsoft/ms_marco` @ `a47ee7aae8d7d466ba15f9f0bfac3b3681087b3a`, `v2.1/train-00000-of-00007.parquet`, sha256 `c92b58a2f53fbbe3cb6a24dadeafaaa7c0cf6f231eb99b95910d6b339812cffc` |
| Qualification window | `../2026-09-15-granite-4.1-3b-alora-hybrid/qualification-window-v2.json`, file sha256 `dfb8cd7203418c79bae27306ffc4857f0ab02be7f9f8ddcae793af556e9cab37`, windowSha256 `ea9e4a0c…` (built by `prepare_answerability_suites.py`; see preflight) |
| Models runtime (qualification) | `integrallis/models` tag `v0.3.42` = `6063076e476ba7a477e3f6dd966a77b571352d29` |
| Adapter provenance commit | tag `adapter-provenance/granite-4.1-3b-answerability-pilot2` = `ec56bc99a04457df8274c84c1f4ff72dc4167d9d` |

MS MARCO usage: its terms say "non-commercial research purposes only" **(read)**. The publisher
confirmed use for this release on 2026-09-17.

## 1. Training (Python, third-party toolchain)

**Environment (measured, `pilot2/training-manifest.json`):**
- **Hardware:** Vultr A16-8Q vGPU (8 GB), driver 550.90.07.
- **Software:** Python 3.10.12, torch 2.6.0+cu124, transformers 5.17.0, peft 0.21.0, bitsandbytes 0.50.2, accelerate 1.15.0, pyarrow 25.0.1.

```bash
python3 -m venv venv && ./venv/bin/pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
./venv/bin/pip install transformers==5.17.0 peft==0.21.0 bitsandbytes==0.50.2 accelerate==1.15.0 pyarrow==25.0.1 safetensors
# sources (verify sha256 against the table above)
curl -sL -o train-src/quac-train_v0.2.json https://s3.amazonaws.com/my89public/quac/train_v0.2.json
curl -sL -o train-src/squad_v2-train.parquet "https://huggingface.co/datasets/rajpurkar/squad_v2/resolve/3ffb306f725f7d2ce8394bc1873b24868140c412/squad_v2/train-00000-of-00001.parquet"
curl -sL -o train-src/msmarco-train-00000-of-00007.parquet "https://huggingface.co/datasets/microsoft/ms_marco/resolve/a47ee7aae8d7d466ba15f9f0bfac3b3681087b3a/v2.1/train-00000-of-00007.parquet"
# prepared set (defaults: 4000 per source per label, 150 validation per source per label, 4 distractors, seed 20260916)
python3 prepare_answerability_training.py --quac-train train-src/quac-train_v0.2.json \
  --squad-train train-src/squad_v2-train.parquet --msmarco-train train-src/msmarco-train-00000-of-00007.parquet --output prepared
# compare prepared/manifest.json with prepared-manifest.json
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 ./venv/bin/python train_answerability_alora.py --prepared prepared --out runs/pilot2 \
  --load-in-4bit --max-length 2048 --batch-tokens 2048 --gradient-accumulation 8 --train-limit 6000 \
  --validation-limit 300 --epochs 1 --eval-every 150 --learning-rate 1.5e-4
```

**Expected (measured):**
- 4,924 examples used (1,076 over 2,048 tokens), 364 steps, held-out balanced accuracy 0.7752.
- `adapter_model.safetensors` sha256 `67533dff14cd0cfa9ae0d00e83a9955226426ff1f98f1fd28790b2fd8757eea6`.
- `adapter_config.json` sha256 `7b559cf39b86084d92363ad6bce1ed92a9ef3c2de78e512c65b693dd6e1f7209`.

GPU training is not guaranteed bit-reproducible across drivers or cards **(believed)**. A retrained
adapter with different hashes must go through qualification again. It does not inherit this one.
The published bundle is the artifact of record.

## 2. Reference-path window evaluation (Python, PEFT on CPU)

**Environment (measured on the reference host):** torch 2.14.0+cpu, transformers 5.17.0, peft 0.21.0, bf16, greedy decoding.

```bash
for suite in msmarco-v2.1-validation squad-v2-dev mtrag-human-rag; do
  python3 evaluate_alora_window.py --base ibm-granite/granite-4.1-3b --base-revision c0650403e44e78ec0262dab1c90914c65b196c4e \
    --adapter runs/pilot2/adapter --window ../2026-09-15-granite-4.1-3b-alora-hybrid/qualification-window-v2.json \
    --suite $suite --output pilot2/window-$suite-peft.json
done
```

Expected (measured) on original labels: SQuAD 0.860, MS MARCO 0.750, MT-RAG 0.673.

## 3. Label audit and evidence-confirmed labels

**Tools:**
- `../2026-09-15-granite-4.1-3b-alora-hybrid/audit_suite_labels.py`: blind export, adjudication, admission, re-scoring with Wilson intervals; 13 tests.
- `../2026-09-15-granite-4.1-3b-alora-hybrid/confirm_disputed_labels.py`: verbatim-quote protocol; 12 tests.

**Judges.** The judges are model agents (Claude Opus 5 and Claude Sonnet 5 at the time,
2026-09-17). Their exact prompts are in `label-prompts/`:
- `*-audit-judges.workflow.js`: the blind answerability judges;
- `confirmation-extractor.workflow.js`: quote extraction;
- `confirmation-verifiers.workflow.js`: the two verifiers.

The first 67-case MS MARCO audit used the same judge prompt text as the relabel workflow, pointing
at `msmarco-label-audit/msm-audit-blind.json`.

Model judges are not deterministic services **(believed)**. A third party re-running them should
expect some disagreement. The committed judge outputs are the record, and the admission and
control rules decide whether a re-run's labels may be used.

```bash
cd ../2026-09-15-granite-4.1-3b-alora-hybrid
# blind export (then run the judge prompts on each part)
python3 audit_suite_labels.py export --window qualification-window-v2.json --suite squad-v2-dev --out-dir <dir> --parts 3 --seed 20260918
# adjudicate committed judgements
python3 audit_suite_labels.py adjudicate --window qualification-window-v2.json --suite squad-v2-dev \
  --judge "A=../2026-09-16-granite-answerability-alora/squad-label-audit/squad-v2-dev-part*-judgeA.json" \
  --judge "B=../2026-09-16-granite-answerability-alora/squad-label-audit/squad-v2-dev-part*-judgeB.json" --out adjudicated.json
python3 audit_suite_labels.py adjudicate --window qualification-window-v2.json --suite msmarco-v2.1-validation \
  --judge "A=../2026-09-16-granite-answerability-alora/msmarco-label-audit/msm-*judgeA.json" \
  --judge "B=../2026-09-16-granite-answerability-alora/msmarco-label-audit/msm-*judgeB.json" --out adjudicated-msmarco.json
# evidence confirmation (the prepare seed is the default 20260919)
python3 confirm_disputed_labels.py prepare --window qualification-window-v2.json \
  --adjudicated msmarco-v2.1-validation=../2026-09-16-granite-answerability-alora/msmarco-label-audit/adjudicated-msmarco-suite.tool.json \
  --adjudicated squad-v2-dev=../2026-09-16-granite-answerability-alora/squad-label-audit/adjudicated-squad-suite.json \
  --judge "A=../2026-09-16-granite-answerability-alora/squad-label-audit/squad-v2-dev-part*-judgeA.json" \
  --judge "B=../2026-09-16-granite-answerability-alora/squad-label-audit/squad-v2-dev-part*-judgeB.json" --out-dir <dir>
python3 confirm_disputed_labels.py verify-quotes --out-dir ../2026-09-16-granite-answerability-alora/label-confirmation
python3 confirm_disputed_labels.py decide --out-dir ../2026-09-16-granite-answerability-alora/label-confirmation \
  --adjudicated msmarco-v2.1-validation=../2026-09-16-granite-answerability-alora/msmarco-label-audit/adjudicated-msmarco-suite.tool.json \
  --adjudicated squad-v2-dev=../2026-09-16-granite-answerability-alora/squad-label-audit/adjudicated-squad-suite.json
```

**Expected (measured):**
- Controls: 0/20 false answerable, 20/20 true answerable.
- `confirmed-msmarco-v2.1-validation.json` casesSha256 `e8425109…`.
- `confirmed-squad-v2-dev.json` casesSha256 `2f4f7897…`.

## 4. Java qualification (system of record)

**Runner:** `release-pilot2/host-run-pilot2.sh` targets Ubuntu 24.04 x86-64 as root and does the following:
- installs Temurin JDK 25 and a minimal Rust toolchain for the optional Rust FFM kernels;
- checks out Models `v0.3.42`;
- verifies the window hash;
- downloads and hash-checks the base GGUF and tokenizer;
- verifies the adapter sources against their sha256;
- fetches the model card, NOTICE and adapter config from the provenance commit;
- packages the runtime bundle.

**Measured hosts:** Hetzner cpx62 (16 vCPU AMD EPYC Genoa, 32 GB) and AWS c7a.4xlarge (16 vCPU AMD EPYC 9R14).

```bash
mkdir -p /opt/modeljars-runs/granite41-answerability-pilot2/store/adapter-src
cp adapter_model.safetensors adapter_config.json /opt/modeljars-runs/granite41-answerability-pilot2/store/adapter-src/
bash host-run-pilot2.sh bootstrap
bash host-run-pilot2.sh artifacts        # bundle sha256s: see below
bash host-run-pilot2.sh smoke            # 3 cases, both backends
bash host-run-pilot2.sh identity         # first 10 cases, both suites, specialist+base, pure-java+rust-ffm
bash host-run-pilot2.sh junit            # gates 1-3 (two adapter-agnostic tests) + the IBM-pinned test for the record
bash host-run-pilot2.sh longcontext rust-ffm
bash host-run-pilot2.sh crossover pure-java
bash host-run-pilot2.sh arm squad-v2-dev specialist pure-java     # deciding backend
bash host-run-pilot2.sh arm squad-v2-dev base pure-java
bash host-run-pilot2.sh arm msmarco-v2.1-validation specialist pure-java
bash host-run-pilot2.sh arm msmarco-v2.1-validation base pure-java
bash host-run-pilot2.sh rust-window
```

**Runtime bundle (measured):** identical on all five hosts.

| File | sha256 |
|---|---|
| `adapter_model.safetensors` | `67533dff14cd0cfa9ae0d00e83a9955226426ff1f98f1fd28790b2fd8757eea6` |
| `models-activated-lora.json` | `27de22cf55ca941832bb3d5d6e0ce46de570edfc85f00ea33a11c30b2d33ccd3` |
| `NOTICE` | `4d3b1f70e77411b1d51ba33215dfe2bd7d1f4090d384725b7f1820df7a8937b1` |
| `LICENSE` | `f3b8149a65f7ae0e2fe55de40d52b98dd8965e6b213b7c441a9596ba670c2e64` |

Re-score each completed arm on the confirmed labels, then apply the unchanged gate:

```bash
python3 ../2026-09-15-granite-4.1-3b-alora-hybrid/audit_suite_labels.py rescore \
  --adjudicated label-confirmation/confirmed-squad-v2-dev.json \
  --arm "pilot2-specialist-purejava=release-pilot2/evidence/pj1/window-squad-v2-dev-specialist-pure-java.json" \
  --arm "base-purejava=release-pilot2/evidence/pj2/window-squad-v2-dev-base-pure-java.json"
```

**Gate:** structured rate 1.0, confirmed balanced accuracy at least 0.80 and at least the base, on
pure Java. Rust is published only if it also passes, and only after the identity check on the
first 10 cases.

**Recorded so far (measured):**
- SQuAD pure-Java specialist: 0.868 (0.789–0.920), structured 200/200, all shared.
- SQuAD pure-Java base: 0.567.
- Adapter identity pure Java vs Rust: 10/10 on both suites.
- JUnit: 2/2 deciding tests passed, and the IBM-pinned test also passed.

## 5. Evidence layout

- `release-pilot2/evidence/<host>/`: each host's evidence directory. It holds `host-run.log`, per-arm JSON and logs, `adapter-bundle.sha256`, `models-activated-lora.json`, `host-baseline.txt` (lscpu, free, df) and `java-version.txt`.
- **Every window report records:** `modelsRevision`, backend, kernel plan, arm, suite, windowSha256, window file sha256, environment, per-case outputs, predictions, shared-prefix tokens and timings.
- `../2026-09-15-granite-4.1-3b-alora-hybrid/preflight.md`: every decision rule, with the time it was fixed relative to the data it governs.
