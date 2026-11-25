# Gated-fusion study, stage S1: gates on real weights

Control record, written before provisioning. Owner approval 2026-11-25 ("Proceed as recommended"):
the S1 box under EUR 3. Protocol: models `benchmark-results/2026-11-25-gated-logit-fusion/PREREGISTRATION.md`
(frozen 2026-11-25, Amendment 1); commands: its `REPRODUCE.md`.

## Inventory before provisioning (2026-11-25T20:32:43Z)

servers=0 firewalls=0 volumes=0 primary-ips=0 floating-ips=0 snapshots=0 (the first write of this line counted header rows; corrected from a re-audit that counted data rows, taken after this box existed and showed only it)

## Plan

- type: cpx62 (16 shared vCPU, 32 GB, fsn1), EUR 0.2452/h gross; the dedicated slot is kept for the
  survival run. Gates are identity checks, not timings, so shared cores are acceptable here.
- image: ubuntu-24.04; ssh key vectors-bench-v2 (114663461); firewall TCP/22 from <operator-ip>/32 only
- label: delete-by=2026-11-26-0432Z (8 h); budget under EUR 3 (about 12 h at this price); the puller deletes
- what runs: models main 05cbefc6 cloned from the public repository (no transfer, no credentials);
  JDK 25; `:models-bench:installDist`; official GGUFs L and I downloaded at the pinned revisions;
  M converted by `convert-math.sh` (llama.cpp v1.0.0); the September data prepared by
  `prepare_fusion_data.py`; sha256 and byte counts recorded in `models.json` before any gate;
  gates G0, G1-gated, G8, G4 in that order, each stopping the chain on FAIL; evidence pulled before
  teardown.
- teardown: identity-guarded on the id this record names; never 147840357

## Server id and results

- server id **177684127**, name `fusion-s1`, cpx62 fsn1, created 2026-11-25T20:33:00Z, delete-by=2026-11-26-0432Z
- firewall id **12165236** (`fusion-s1-fw`), TCP/22 from <operator-ip>/32 only
- teardown guard: delete only id 177684127; never 147840357

## Log

- 2026-11-25T20:33:00Z created
- 2026-11-25T20:38:51Z setup finished (JDK, uv, models 05cbefc6 cloned from the public repo, bench CLI built, September data prepared, L and I downloaded, M converted); see the box's /root/evidence
- 2026-11-25T20:40:36Z setup pass 2: rustup (the repo's rust-toolchain.toml) was missing for :backend-cuda:compilePtx; bench CLI, data, GGUFs, conversion (see the box's /root/evidence)
- 2026-11-25T20:42:04Z setup pass 3: the PTX compile needs rust-src on nightly-2026-09-17 (the error named it); rerun of the setup script
- 2026-11-25T20:44:30Z setup pass 4: the PTX compile also needs the llvm-bitcode-linker component on that nightly; rerun
- 2026-11-25T20:45:53Z conversion rerun through the uv venv (the box has no system pip; convert-math.sh assumed one, to be fixed in the repo)
- 2026-11-25T20:46:27Z gates launched (G0, G4 gsm8k, G4 math500, G8, G1-gated; pure-java backend, context 4096, T=0, 20 items)
- 2026-11-25T21:46:06Z G0 attempt 1 FAIL (measured): M vs L differ in exactly one field, tokenizer.ggml.add_eos_token: absent in the official Coder and Instruct GGUFs, written as false by llama.cpp v1.0.0's converter into the Math GGUF; vocabulary, merges, tokens, types and special tokens identical. Response under the protocol: fix the member artifact's metadata (remove the key the official files do not carry, with gguf-py's gguf-new-metadata), re-pin by sha256, and rerun the chain from G0; the gate is not changed. attempt-1 report kept as gate-g0.attempt1.json.
- 2026-11-25T21:46:41Z M re-pinned (key removed); gates relaunched from G0
- 2026-11-25T22:58:22Z gates done: G0 PASS (attempt 2), G4 gsm8k 1319/1319, G4 math500 500/500, G8 PASS (240 comparisons, worst max |dlogit| 0.0), G1-gated PASS (40 comparisons); evidence pulled; server 177684127 and firewall 12165236 deleted; re-audit: servers=c6sample 
