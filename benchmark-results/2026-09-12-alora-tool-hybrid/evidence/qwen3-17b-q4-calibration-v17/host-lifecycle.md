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

## Actual lifecycle and result

- Server: Hetzner `165754862`, `modeljars-alora-v17-java-20260914`
- Temporary address: `5.161.100.228`
- Firewall: Hetzner `11619101`, `modeljars-alora-v17-java-ssh`
- Created: `2026-09-14T03:07:42Z`
- Measured CPU / memory: 16 logical `AMD EPYC-Rome Processor` CPUs and 32,859,287,552 bytes
  physical memory
- Java: Eclipse Adoptium `25.0.4.1`, OpenJDK 64-Bit Server VM
- Exact executed Models code revision:
  `9dd66837e1770adc10e355ee1cc6ca9b07dbead5`
- Calibration: PASS, 20/20 calls, 9/10 no-calls, 30/30 physically shared; selected threshold
  `2.7213982172616653`
- Calibration artifact SHA-256:
  `97c81ca7dcb3eab4114f0164c25506a9e6332c2339083cc1c72f3925ee18487d`
- Calibration log SHA-256:
  `c72bb3294809af254b44201be8c02c6297653b591057442cf4642e7effb85149`
- The calibration artifact was frozen in commit `ad8a999` before the screen began.
- Untouched screen: REJECT after 43/45 observations. All 15 observed no-call cases were correct,
  but three of the first 28 call cases missed. With only two calls remaining, the maximum possible
  screen result was 27/30, below the frozen 28/30 call floor. All 43 observations physically shared
  their immutable prefix.
- Screen log SHA-256:
  `6cbb45b23453240ba72bccf2141c9df2cda655305d001f9cd32f9c4447667973`
- Stop condition: the run was terminated as soon as passing the frozen screen gate became
  mathematically impossible. The sealed qualification window and generation gates were not opened.
- Deletion requested: `2026-09-14T04:01:24Z`
- Closing inventory audit: `2026-09-14T04:02:05Z`
- Verified closing state: zero Hetzner servers, volumes, floating IPs, primary IPs, firewalls, and
  snapshots; zero Vultr instances
- Approximate metered lifetime: 54 minutes; approximate compute cost at the preflight price:
  USD 0.40
- Local watchdog and ephemeral SSH known-host material were retired after provider deletion was
  verified.
