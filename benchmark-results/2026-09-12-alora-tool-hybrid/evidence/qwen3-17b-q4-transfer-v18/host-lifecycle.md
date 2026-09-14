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
