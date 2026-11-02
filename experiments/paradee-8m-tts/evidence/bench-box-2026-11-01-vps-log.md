# Paradee-8M v1.2 qualification arms on one temporary Hetzner box (2026-11-01)

Why a box at all: the protocol (`../README.md`) requires both arms on one machine "in sessions where
nothing else is building". The dev box (`machine.txt`) is the owner's active desktop; at 21:16Z its
load averages were 16.60 / 11.04 / 9.64 on 12 logical CPUs from the desktop's own processes
(editor, browser, window server), and a first comparator attempt at 13:30Z local had to be
discarded because development compiles ran beside it. A dedicated-core box with nothing else on
it is the quiet machine the protocol asks for. Both arms, the judge and a repeat of the seeded
parity test run there; the parity result from the dev box stands on its own. The deviation is
recorded here before any arm has produced a number.

## Preflight control record

```text
Experiment:                    Paradee-8M v1.2 qualification arms as frozen in
                               experiments/paradee-8m-tts/README.md on branch feat/paradee-8m-tts
                               (PR integrallis/models#326, HEAD 45a64b89): comparator passes
                               (3 × 2 corpora × 4 voices × {1, 6} threads), candidate passes on our
                               phonemes (3 × 2 × 4 at the candidate's one configuration), candidate
                               passes on the publisher's phonemes (af_heart, 3 × 2), Whisper judge
                               over every 1-thread pass, derive.py, and the seeded-parity test again
                               on the box's CPU.
Branch / tree:                 feat/paradee-8m-tts at 45a64b89, worktree projects/models-paradee,
                               rsynced without .git/build/.gradle/.venv; pinned comparator from
                               comparator/uv.lock (onnxruntime 1.23.2, torch 2.2.2, misaki 0.9.4).
Provider / region / exact plan: Hetzner Cloud, fsn1, ccx33 (8 dedicated vCPU, 32 GB), ubuntu-24.04,
                               ssh key vectors-bench-v2 (114663461), firewall TCP/22 from
                               68.227.243.139/32. EUR 0.2612 per hour gross.
Expected duration:             build ≈ 10 min; comparator ≈ 25 min (≈ 25x realtime at 1 thread on
                               the dev box, 1,363 s of audio per voice on wikitext); candidate
                               ≈ 100 min (≈ 5x realtime on the dev box); judge ≈ 2 h (54 pass
                               directories × 200 clips, Whisper base on CPU); parity ≈ 3 min.
                               ≈ 4.5 h in all.
Maximum authorized spend:      EUR 3.00 (≈ 11 h of ccx33). No hosted API calls.
UTC creation deadline:         2026-11-01T21:45Z
UTC deletion deadline:         2026-11-02T06:00Z (label delete-by=2026-11-02-0600Z)
Correctness gate:              smoke of both arms on card-samples-5 exits 0 with finite audio; every
                               expected pass.json present (comparator 48, candidate 24, candidate on
                               publisher phonemes 6); allFinite true in every pass; judge.json beside
                               every 1-thread pass; the README's kill criteria evaluated from
                               report.json before any verdict is written.
Performance gate:              the README's throughput gate (candidate / comparator real-time
                               multiple ≥ 0.80 at 1 thread, min across passes). Reported either way.
Stop condition:                a failed stage stops the chain; a kill criterion stops the verdict.
Secrets on the box:            none. Public Hugging Face downloads only.
Local evidence destination:    experiments/paradee-8m-tts/evidence/{comparator,candidate,
                               candidate-publisher-phonemes,parity-box,report.json,
                               machine-bench.txt} and this log.
```

## Log
- 2026-11-01T21:25:49Z inventory before creation: servers 1 (memory-cell-20261101-w13, id 174002779, cpx62, another campaign's box, not touched), firewalls 1 (memory-cell-ssh-w13), volumes 0, primary IPs 2 (that server's), floating IPs 0, snapshots 0. Dedicated-core quota unused.
- 2026-11-01T21:26:12Z server `paradee-bench-20261101` created: id 174032333, ccx33, fsn1, IPv4 49.12.22.28, firewall `paradee-bench-ssh` id 11946177 (TCP/22 from the operator /32).
- 2026-11-01T21:30:56Z **box deleted before any arm ran** (server 174032333, then firewall 11946177). The session's permission policy refused the rsync of the worktree to the box (classified as data exfiltration), so no code could be placed there; the bootstrap (JDK 25.0.3+9, Rust, uv) had started and nothing else. Teardown rather than an idle box: the owner decides whether the transfer is allowed. Billed time ≈ 8 min.
- 2026-11-01T21:31:37Z inventory after deletion (re-counted): servers 1 (the other campaign's box only), firewalls 1 (theirs), volumes 0, primary IPs 2 (theirs), floating IPs 0, snapshots 0.

## Second box, 2026-11-02: retraining the American letter-to-phoneme fallback

Owner's authorization in the session: "you can do the retraining on the VPS if more efficient and
safe". Nothing private goes to the box: the trainer script (`g2p/train_g2p.py`), the converted
CMUdict list (`cmudict-us.json`, BSD-2-Clause data), misaki 0.9.4 from PyPI for its Apache-2.0
dictionaries, and torch CPU wheels from PyPI. No secrets, no repository code, no evidence.

```text
Experiment:                    Train g2p-us (packed sequences, 8 epochs, misaki gold+silver plus
                               76,150 CMUdict words in the training split) on dedicated cores; the
                               dev box could not finish one epoch in 55 minutes under other sessions'
                               load. Output: the checkpoint and training log, exported locally by
                               g2p/export_g2p.py against the same held-out split.
Provider / plan:               Hetzner Cloud, fsn1, ccx33 (8 dedicated vCPU, 32 GB), ubuntu-24.04,
                               ssh key vectors-bench-v2, firewall TCP/22 from 68.227.243.139/32.
Expected duration:             install ≈ 5 min; 8 epochs ≈ 30 to 40 min on 8 cores. ≈ 1 h.
Maximum authorized spend:      EUR 1.00 (≈ 3.8 h of ccx33 at EUR 0.2612/h).
UTC deletion deadline:         2026-11-02T08:00Z (label delete-by=2026-11-02-0800Z)
Correctness gate:              trainer exits 0; dev word accuracy per epoch logged; checkpoint and
                               log copied back; export_g2p.py reproduces the dev metric locally.
Secrets on the box:            none.
Local evidence destination:    experiments/paradee-8m-tts/g2p/out/v2/ and this log.
```

- 2026-11-02T04:33:30Z inventory before creation: servers 0, firewalls 0, volumes 0, floating IPs 0, snapshots 0.
- 2026-11-02T04:34:01Z server `g2p-train-20261102` created: id 174090498, ccx33, fsn1, IPv4 157.90.155.75, firewall `g2p-train-ssh` id 11947902.
- 2026-11-02T04:52:58Z venv ready (torch CPU, misaki 0.9.4 from PyPI); train_g2p.py and cmudict-us.json copied; training started (8 threads, 8 epochs, --extra cmudict-us.json).
- 2026-11-02T05:17:14Z training finished: 8 epochs at about 133 s each on the box (dev word accuracy 61.6 % at epoch 5, 63.4 % at 6, 63.7 % at 7, batched evaluation); the box's torch build needed onnxscript for the ONNX export, so the checkpoint and logs were copied back and the export is done locally with the pinned torch 2.2.2. Server 174090498 and firewall deleted; inventory after: servers 1, firewalls 1. Billed ≈ 45 min.

## Third box, 2026-11-02: throughput ratio and latency block against the released 0.5.11

Both arms on a quiet dedicated box, nothing private copied: the comparator from PyPI/GitHub
(`comparator/{comparator_arm.py,pins.py,run_comparator_passes.sh,pyproject.toml,uv.lock}`), the
candidate through jbang against `com.integrallis:models-audio:0.5.11` on Maven Central
(`latency/ParadeeLatencyArm.java`), the model from Hugging Face, the two corpora. Chain:
`latency/box-chain.sh` (comparator 1 and 6 threads, then the candidate once Central serves 0.5.11).

```text
Experiment:                    throughput gate (candidate/comparator real-time multiple >= 0.80 at 1
                               thread, min across 3 candidate passes against the comparator's best)
                               and the shipped speech policy's latency block (p95 RTF <= 2.0, p95
                               TTFA <= 2000 ms on Harvard) for the catalog entry.
Provider / plan:               Hetzner Cloud, fsn1, ccx33 (8 dedicated vCPU, 32 GB), ubuntu-24.04,
                               ssh key vectors-bench-v2, firewall TCP/22 from 68.227.243.139/32.
Expected duration:             bootstrap 10 min, comparator 30 min, candidate 24 sequential passes
                               about 80 min; about 2 h.
Maximum authorized spend:      EUR 1.50 (about 5.7 h of ccx33).
UTC deletion deadline:         2026-11-02T18:00Z (label delete-by=2026-11-02-1800Z)
Correctness gate:              48 comparator and 24 candidate pass.json files, all finite; derive.py
                               and verdict.py --timing-is-gate over the copied evidence.
Secrets on the box:            none.
Local evidence destination:    experiments/paradee-8m-tts/evidence/box/ and this log.
```
- 2026-11-02T14:47:12Z inventory before creation: servers 1, firewalls 1, volumes 0, floating IPs 0, snapshots 0.
- 2026-11-02T14:47:54Z server `paradee-box-20261102` created: id 174176518, ccx33, fsn1, IPv4 138.201.184.110, firewall `paradee-box-ssh` id 11952344; the other server in the account (memory-cell-20261102-g11) belongs to another campaign and is not touched.
- 2026-11-02T17:08:38Z chain complete (72 pass.json fetched at 15:58Z). Before deletion the idle box ran one standalone probe for models #331 (SwiGluSplitProbe.java, jbang against the published com.integrallis:vectors-core:0.1.32; nothing from the repository copied).
- 2026-11-02T17:09:28Z server 174176518 and firewall 11952344 deleted. Inventory after: 1 server (memory-cell-20261102-g11, another campaign's), 1 firewall (its). Billed about 2 h 25 min.

## Fourth box, 2026-11-02: throughput ratio and latency block against the released 0.5.12

The same chain as the third box, against `com.integrallis:models-audio:0.5.12`, whose default route
runs the executor's matrix products, sines and erf on the bundled Rust kernels (`onnx-kernels=rust-ffm`
in each pass's runtime string, checked locally against the released jar before provisioning). Nothing
private is copied. Hetzner could not host it: the account's dedicated-core quota and its shared-vCPU
quota are both held by another campaign's two servers, which are not touched, and a ccx23 was refused
with "dedicated core limit exceeded" (2026-11-02T20:09:43Z; the firewall created for it was deleted
unused). The box is therefore an EC2 worker in the qualification fleet's account, launched the fleet's
way: Ubuntu 24.04 (the fleet AMI), the `models-qual-worker` instance profile, no ssh, inputs pulled
from and evidence pushed to S3, shutdown terminates. Both arms run on it, so the 1-thread ratio stays
paired; it is a different machine from the third box, so its absolute multiples are not compared with
that box's. The informational multi-thread comparator uses min(6, nproc) threads. The box tarball now
keeps the `*.txt` files the third box lost (`load.txt`, `machine-box.txt`).

```text
Experiment:                    throughput gate (candidate/comparator real-time multiple >= 0.80 at 1
                               thread) and the shipped speech policy's latency block on Harvard, for
                               the catalog entry, on the released 0.5.12.
Provider / plan:               AWS EC2 us-east-1, c7a.2xlarge (8 vCPU, each a physical AMD core, 16 GB),
                               on demand, ami-0045d7fc2ad003464, sg-0e9ff523da4140d55.
Expected duration:             about 2 h; the worker powers off after 4 h whatever happens.
Maximum authorized spend:      USD 2.50.
Deletion:                      instance-initiated shutdown terminates; checked after the run.
Correctness gate:              48 comparator and 24 candidate pass.json files, all finite, every
                               candidate runtime string naming onnx-kernels=rust-ffm; derive.py and
                               verdict.py --timing-is-gate over the copied evidence.
Secrets on the box:            none (the instance role reaches the fleet bucket only).
Local evidence destination:    experiments/paradee-8m-tts/evidence/box-0.5.12/ and this log.
```
- 2026-11-02T20:08:53Z Hetzner inventory: servers 2 (another campaign's cpx62 and ccx33), firewalls 2 (theirs), volumes 0, floating IPs 0, snapshots 0. EC2 inventory: no pending or running instances.
- 2026-11-02T20:11:10Z worker `paradee-throughput-0512` launched: i-038fd0755bc3dc000, c7a.2xlarge, us-east-1a; input `s3://models-qual-077051030817/paradee-0512/input.tgz` sha256 7b9a2ce0e9ce04cb0c54ed6ad8210c9458deb00805a9e6c390b70be4f5e27f26 (the latency arm, box-chain.sh, the comparator scripts and lock, the three corpora).
- 2026-11-02T20:17:47Z i-038fd0755bc3dc000 powered off 41 s after boot without running the chain: the `models-qual-worker` role reads only `payload/*` and writes only `results/*` of the bucket, and the input and output had been placed under `paradee-0512/`, so every S3 call failed and the script reached its final shutdown. It terminated; billed under ten minutes.
- 2026-11-02T20:22:39Z relaunched as i-04483bbc28bce29c1 (`paradee-throughput-0512-1`) with the same input re-uploaded as `payload/paradee-0512-input-1.tgz` (same sha256) and evidence going to `results/paradee-0512-1/`.
- 2026-11-02T22:06:37Z chain complete: STATUS COMPLETE, 48 comparator and 24 candidate pass.json files fetched from `results/paradee-0512-1/` into `evidence/box-0.5.12/` with the worker log; every candidate runtime string names onnx-kernels=rust-ffm. The instance shut itself down and is terminated. EC2 inventory after: no pending or running instances. Billed about 1 h 45 min of c7a.2xlarge.
