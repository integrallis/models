# V18 production-Q4 transfer host lifecycle

- Experiment: exposed calibration followed by a separately committed BFCL-live screen for
  `qwen3-17b-q4-transfer-v18`
- Provider / region / exact plan: Hetzner Cloud `ash`, `cpx51`
- Hardware: 16 shared x86 vCPUs, 32 GB RAM, 360 GB disk, Ubuntu 24.04 x86-64
- Live price at preflight: USD 0.4479/hour
- Expected use: under two hours for setup, calibration, evidence freeze, live screen if permitted,
  evidence transfer, and teardown
- Maximum authorized spend: USD 1.35; three-hour hard deletion ceiling
- Created no earlier than: `2026-09-14T04:30:53Z`
- Hard deletion deadline: `2026-09-14T07:30:53Z`
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
  exact IDs every 60 seconds and enforces the `2026-09-14T07:30:53Z` deadline.

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
