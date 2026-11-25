# Reproducing the gated-fusion study (S1 onwards)

The protocol is [PREREGISTRATION.md](PREREGISTRATION.md). S0 (the runtime backend, the `gated:` arm
and the toy-member proofs of G1-gated and G8) landed in Models; this guide starts at S1. Every
command runs from the Models checkout through the installed bench CLI:

```bash
./gradlew :models-bench:installDist -q
CLI=models-bench/build/install/models-bench/bin/models-bench
```

Provenance labels follow the working agreement: **[measured]** here, **[read]** from a document,
**[believed]** an estimate awaiting measurement.

## S1a. Pin the weights

`models.json` names every file by repository, revision and name. The official GGUFs download from
those URLs; the Math member is converted by `convert-math.sh <workdir>` (pinned safetensors
revision, pinned llama.cpp release) because no official GGUF of it exists. Record each file's
sha256 and byte count in `models.json` **before any gate runs**, and commit that change on its own.

## S1b. Gates, in order (each refuses to continue past a FAIL)

Members are named lead first; `L`, `M`, `I` bind to the paths of the pinned files. The datasets
are the September study's: run its `prepare_fusion_data.py` (it pins every source by revision and
sha256 in its `data-manifest.json`) into a local directory and point `DATA` at it; the prepared
files are not committed.

```bash
DATA=<directory prepare_fusion_data.py wrote>
MEMBERS="L=$W/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf,M=$W/qwen2.5-math-1.5b-instruct-q4_k_m.gguf,I=$W/qwen2.5-1.5b-instruct-q4_k_m.gguf"
COMMON=(--backend pure-java --context-length 4096 --thinking off --temperature 0)

# G0: byte-identical tokenizers across L, M, I
$CLI logit-fusion gate g0 --members "$MEMBERS" --report evidence/gate-g0.json

# G1-gated: never-gate == lead alone; always-gate == every-token FusionDecoder, token for token
$CLI logit-fusion gate g1-gated --members "$MEMBERS" --dataset gsm8k --data $DATA/gsm8k-test.jsonl \
  --limit 20 --max-tokens 256 "${COMMON[@]}" --report evidence/gate-g1-gated.json

# G8: catch-up by one prefill == stepping, per member, at d = 1, 2, 7, 31, within 1e-3 per logit
$CLI logit-fusion gate g8 --members "$MEMBERS" --dataset gsm8k --data $DATA/gsm8k-test.jsonl \
  --limit 20 --max-tokens 64 --deltas 1,2,7,31 --tolerance 1e-3 "${COMMON[@]}" --report evidence/gate-g8.json

# G4: the scorers on gold answers (unchanged from September)
$CLI logit-fusion gate g4 --dataset gsm8k --data $DATA/gsm8k-test.jsonl --report evidence/gate-g4-gsm8k.json
$CLI logit-fusion gate g4 --dataset math500 --data $DATA/math500-test.jsonl --report evidence/gate-g4-math500.json
```

G2 (the NumPy reference of the gated step) reuses `g2-dump` with the gated arm's per-step logits and
`hf_reference_fusion.py`; its gated variant is written at S1 beside this guide once the dump format
is fixed.

## S2. Pilot

The stop-rule arms on 50 GSM8K and 50 MATH-500 items, through the `run` command with
`--arm gated:L+M+I:poe:0.55,0.25,0.20:0.65` (G-article), `--arm gated:<best>+...:poe:uniform:0.65`
(G-best), `--arm fuse:L+M+I:poe:uniform` (F-every), and the three member arms; each report's
`aggregations` carries the gate counts. Core-seconds, consult rate and catch-up lengths from the
pilot replace every believed cost in the protocol before S3.

## Hosts

Hetzner ccx33 (the account's dedicated-core limit), Ubuntu 24.04, JDK 25, the kernel route
recorded per member in every report; the control record for each box lives beside it under
`evidence/`.
