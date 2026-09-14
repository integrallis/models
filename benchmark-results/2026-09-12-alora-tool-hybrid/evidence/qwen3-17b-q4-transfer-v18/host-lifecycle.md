# V18 production-Q4 transfer host lifecycle

- Experiment: exposed calibration followed by a separately committed BFCL-live screen for
  `qwen3-17b-q4-transfer-v18`
- Provider / region / exact plan: Hetzner Cloud `ash`, `cpx51`
- Hardware: 16 shared x86 vCPUs, 32 GB RAM, 360 GB disk, Ubuntu 24.04 x86-64
- Live price at preflight: USD 0.4479/hour
- Expected use: under five hours for setup, repeated calibration after defects, evidence freeze,
  live screen if permitted, evidence transfer, and teardown
- Maximum authorized spend: USD 2.23; `4.98`-hour hard deletion ceiling
- Created no earlier than: `2026-09-14T04:30:53Z`
- Hard deletion deadline: `2026-09-14T09:30:53Z`
- State before creation: zero Hetzner servers, volumes, floating IPs, primary IPs, firewalls, and
  snapshots; zero Vultr instances
- Resolved server type: `cpx51` ID `26`
- Resolved image: Ubuntu 24.04 x86 image `161547269`
- Verified SSH key: Hetzner key `114663461` (`vectors-bench-v2`), whose MD5 fingerprint
  `60:4e:57:38:96:48:c5:45:73:10:9d:37:f7:4f:74:cf` matches the established local Ed25519
  benchmark identity
- Exact Models revision: `8616c7d03bf23c77d5486eda754a2f89b5a4250a`
- Model, adapter, head, calibration source, live screen, thresholds, and gates: exactly as frozen in
  `preflight.md`
- Correctness stop conditions: delete immediately if an artifact identity, calibration gate,
  committed-calibration verification, live-screen gate, or physical shared-prefix identity fails
- Performance claims: prohibited; shared vCPUs make this a correctness gate only
- Sealed qualification data: prohibited until V18's calibration, live screen, and exposed dual
  generation pass
- Local evidence destination: this directory

The exact server/firewall IDs, address, host key, measured hardware, run hashes, phase results,
deletion actions, elapsed cost, and closing provider inventory are appended as they occur.

## Provisioned host

- Server ID / name: `165761108` / `modeljars-alora-v18-java-20260914`
- IPv4: `5.161.100.228`
- Firewall ID / name: `11619238` / `modeljars-alora-v18-java-ssh`
- Created: `2026-09-14T04:32:15Z`
- Measured runtime: OpenJDK `25.0.4`, Ubuntu Linux `6.8.0-138-generic`, AMD EPYC-Rome, 16
  processors, 32,859,291,648 physical bytes
- Automatic cleanup: local LaunchAgent `org.integrallis.modeljars-alora-v18-watchdog` checks the
  exact IDs every 60 seconds and enforces the `2026-09-14T09:30:53Z` deadline.

The first watchdog epoch was discovered to resolve to `2026-09-14T14:30:53Z`, despite the
documented `07:30:53Z` deadline. It was corrected before the original deadline passed. The corrected
deadline includes a two-hour extension for an honest calibration rerun and caps total compute at
approximately USD 2.23 instead of the erroneous roughly USD 4.48 window.

The host checked out code revision `8616c7d03bf23c77d5486eda754a2f89b5a4250a` and passed a clean
`:models-bench:check`. An initial launcher invocation used a module-relative records path and failed
before loading or scoring the model. Its diagnostics were moved outside the evidence directory and
are not qualification evidence. The corrected invocation used the absolute, hash-verified source.

## Phase 1

- Started: `2026-09-14T04:45:53Z`
- Completed: `2026-09-14T05:28:27Z`
- Result: PASS, threshold `2.4962309929993705`, calls 47/50, no-calls 24/25, balanced
  accuracy 0.95, physical shared-prefix identity 75/75
- Calibration report SHA-256:
  `973aa08df9a4bde2969453ffd3d041e2eef79de6ef1be68468f37dd0c5f929e6`
- Calibration log SHA-256:
  `9b1352791d421f17355035e360f2bae8137d2f47d995683b23e936eae06e977a`
- Clean-host check SHA-256:
  `c25d165a87e9dbf44b28f14f439d6aad44d42cb2f04dbda192f99859df3f2ff6`

The evidence was copied to the local worktree and independently recomputed before Phase 2. Phase 2
did not start before the Phase 1 evidence and hashes were committed.

## Pre-score corrections and determinism audit

The first Phase 2 launcher stopped with zero cases scored because the frozen source contained one
valid prior `assistant` message and the qualification loader accepted only `system` and `user`.
Revision `9f1e5c60fda30ee1ec036d5d1decb9d76cd76527` added that source role test-first. A clean
`:models-bench:check` rerun passed with SHA-256
`2f98ead556c8f8dbd50a94087dd230ed4aeef0d0f37a28ff23783de89a918bc0`.

Because the code revision changed, the full exposed calibration was repeated rather than reusing
the committed result. It completed at `2026-09-14T06:16:37Z` and again reported threshold
`2.4962309929993705`, calls 47/50, no-calls 24/25, balanced accuracy 0.95, and physical sharing
75/75. Its report and log hashes were
`93359d5ca1ca0fd08351c90f41afe78d79e496d2fb6961b4dfc0d63f44a12a1c` and
`73c9bbd1ee088915b08a377ac2b40ec569751dd489b84e549ca7ed8640b48b9e`.

An exact comparison found that only the first cold score differed from the first process. Targeted
local experiments reproduced first-call-only Activated-LoRA variation, ruled out physical sharing
and parallel GGUF execution, and found the base path stable. The full finding is recorded in
`cold-activated-determinism.md`. Revision `4ebbc471553d5956970091d72f08fd79ad835a4a` fixes the
F32 adapter warmup in Java and adds a real-weight regression. The passing pre-fix calibration is
retained only as diagnostic evidence; Phase 2 remains unscored and Phase 1 must run again.

## Stabilized diagnostic and teardown

A further full calibration diagnostic under revision
`4ebbc471553d5956970091d72f08fd79ad835a4a` completed successfully: calls `47/50`, no-calls
`24/25`, balanced accuracy `0.95`, and physical sharing `75/75`. Its report SHA-256 is
`314d5bf277feafdbc5ed0a7fb1e843713b9c88c090acda5a89067ef0b5bc8817`. This is diagnostic
evidence only. The implementation still depended on the temporary Models prewarm and therefore
cannot unlock Phase 2.

The host became idle while the underlying Vectors deterministic-F32 fix awaited CI and release.
Server `165761108` and firewall `11619238` were verified by exact ID and name, deleted at
`2026-09-14T07:23:09Z`, and independently verified absent. The local watchdog was unloaded.
The elapsed allocation was 2 hours 50 minutes 54 seconds, an upper-bound compute estimate of USD
1.28 at the recorded hourly price. Closing Hetzner inventory contained zero servers and zero
firewalls. A clean host will be reprovisioned only after the released Vectors dependency and final
Models revision are available.

## Final Vectors 0.1.21 qualification run

All 28 checks on Models revision `55a624ca68bb62115563dfdd5b0f982149ab8396` passed before
capacity was created. The repository resolves Vectors 0.1.21 from Maven Central and retains no
Activated-LoRA prewarm.

- Server ID / name: `165788606` / `modeljars-alora-v18-final-20260914`
- Firewall ID / name: `11620007` / `modeljars-alora-v18-final-ssh`
- Created: `2026-09-14T08:41:42Z`
- Hard deletion deadline: `2026-09-14T13:41:42Z`
- Plan / price ceiling: Hetzner `cpx51` in `ash`, USD 0.4479/hour, at most five hours / USD 2.24
- Measured runtime: OpenJDK `25.0.4`, Ubuntu Linux `6.8.0-138-generic`, AMD EPYC-Rome, 16
  processors, 32,859,291,648 physical bytes
- SSH host-key fingerprint: `SHA256:d4WZMXiZ4YFxwglN5Akzxhup16GE3eplp6Diy1MvP/A`
- Automatic cleanup: local LaunchAgent
  `org.integrallis.modeljars-alora-v18-final-watchdog` checks the exact server and firewall IDs
  every 60 seconds and enforces the deadline.

The opening inventory contained zero Hetzner servers and zero firewalls. The firewall permits SSH
only from the operator workstation and was verified as attached to the exact server. The clean
checkout and all model, adapter, head, and exposed-record hashes were verified before scoring. A
clean `:models-bench:check` passed after installing Java 25 and the repository-pinned Rust build
toolchain. Rust was used only to compile Models' existing owned native-kernel module as required by
the benchmark distribution; V18 constructs `PureJavaBackend` directly and cannot use that module
or an external inference runtime for scoring.

The final Phase 1 calibration ran from `2026-09-14T08:54:59Z` through
`2026-09-14T09:34:26Z`. It passed at threshold `2.4962309929993705`: calls `47/50`, no-calls
`24/25`, balanced accuracy `0.95`, and physical shared-prefix identity `75/75`. The independently
verified report and log SHA-256 values are
`74588d3431a7d81e9e9df2ce2eab748df7a8533ffdcd101793fff0be0f8d542b` and
`9e87a39cf1acc5c301f85f20aaf5dec5028701e9e2c2e95e5f88e8f69c34c459`. Independent
recomputation from the 75 observations reproduced all four counts and balanced accuracy. The
formerly unstable first cold score is now `2.1539079427156906`; all repeated-path scores remained
stable. Phase 2 was not started before this report, log, hash manifest, and lifecycle update were
committed.

Phase 2 started at `2026-09-14T09:39:54Z` with the committed calibration hash and unchanged
threshold. It was stopped after observation 31 made the predeclared gate mathematically impossible:
no-calls `24/25`, calls `2/6`, four positive misses, and physical sharing `31/31`. Reaching the
required `47/50` calls was no longer possible even if every remaining positive passed. The partial
log SHA-256 is `fffb0b94044bfb096a9be764031431c33b05f118b3a3b004c5eac2dd05adcdee`.
No threshold, model, adapter, head, prompt, or gate was changed, and the remainder of the live
window was not scored.

The exact server and firewall were deleted and verified absent at `2026-09-14T10:15:37Z`; the
watchdog was unloaded. Closing inventory contained zero servers, firewalls, volumes, floating IPs,
primary IPs, and snapshots. The allocation lasted 1 hour 33 minutes 55 seconds, an upper-bound
compute estimate of USD 0.71 at the recorded hourly price.
