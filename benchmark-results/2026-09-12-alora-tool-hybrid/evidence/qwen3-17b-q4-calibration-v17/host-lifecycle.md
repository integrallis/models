# V17 production-Q4 calibration host lifecycle

- Experiment: separate 30-case calibration and 45-case untouched screen for
  `qwen3-17b-q4-calibration-v17`
- Provider / region / exact plan: Hetzner Cloud `ash`, `cpx51`
- Hardware: 16 shared x86 vCPUs, 32 GB RAM, 360 GB disk, Ubuntu 24.04 x86-64
- Live price at preflight: USD 0.4479/hour
- Expected use: under 90 minutes for setup, calibration, evidence freeze, screen, evidence transfer,
  and teardown
- Maximum authorized spend: USD 0.90; two-hour hard deletion ceiling
- Created no earlier than: `2026-09-14T03:06:15Z`
- Hard deletion deadline: `2026-09-14T05:06:15Z`
- State before creation: zero Hetzner servers, volumes, floating IPs, primary IPs, firewalls, and
  snapshots; zero Vultr instances, as proved by the V16 closing audit at `2026-09-14T02:49:32Z`
- Resolved server type: `cpx51` ID `26`
- Resolved image: Ubuntu 24.04 x86 image `161547269`
- Verified SSH key: Hetzner key `114663461` (`vectors-bench-v2`), matching the established local
  benchmark identity
- Exact Models revision: `9dd6683` after resolution to a full 40-character SHA on the host
- Model, adapter, head, records, partition, thresholds, and gates: exactly as frozen in
  `preflight.md`
- Correctness stop conditions: delete immediately if an artifact identity, calibration gate,
  calibration-artifact verification, screen gate, or physical shared-prefix identity fails
- Performance claims: prohibited; shared vCPUs make this a correctness gate only
- Sealed qualification data: prohibited until both V17 phases and exposed dual generation pass
- Local evidence destination: this directory

The exact server/firewall IDs, address, host key, measured hardware, run hashes, phase results,
deletion actions, elapsed cost, and closing provider inventory are appended as they occur.
