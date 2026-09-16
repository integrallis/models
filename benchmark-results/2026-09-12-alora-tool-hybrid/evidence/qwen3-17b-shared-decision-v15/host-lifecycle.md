# V15 sequential decision-screen host lifecycle

- Experiment: exact sequential Java scoring of the frozen 75-case Qwen3 1.7B activated-adapter
  decision screen after the local ragged-batch path failed its bit-identity gate
- Provider / region / exact plan: Hetzner Cloud `ash`, `cpx51`
- Hardware: 16 shared x86 vCPUs, 32 GB RAM, 360 GB disk, Ubuntu 24.04 x86-64
- Live price at preflight: USD 0.4479/hour, hourly billing
- Expected use: about three hours including setup, scoring, evidence transfer, and teardown
- Maximum authorized spend: USD 1.80
- Created no earlier than: `2026-09-13T23:20:29Z`
- Hard deletion deadline: `2026-09-14T03:20:29Z`
- Correctness stop condition: delete immediately if artifact hashes, frozen source identity, physical
  KV-prefix identity, or the fixed call/no-call screen gate fails
- Performance work: prohibited; this host executes the fixed sequential correctness screen only
- Local evidence destination: this directory
- State before creation: Hetzner and Vultr inventories both returned zero instances
- Resolved image: Ubuntu 24.04 x86 image `161547269`
- Verified SSH key: Hetzner key `114663461` (`vectors-bench-v2`), matching the local Ed25519 MD5
  fingerprint `60:4e:57:38:96:48:c5:45:73:10:9d:37:f7:4f:74:cf`
- Exact Models revision: `2c8851e32c3d721db2f1b2345caeccfb4e4d09c3`
- Model SHA-256: `061b54daade076b5d3362dac252678d17da8c68f07560be70818cace6590cb1a`
- Adapter SHA-256: `f1a4a1bc15e3942ed5a3fbe2c4750f39a8b24804e20566242ec740c5ce851c21`
- Frozen records SHA-256:
  `944d137ba1e325e0f3daa922a73108de0da5ccf0292bcf4dbe4a565aa11fcb4c`
- Plan-resolution note: the first creation request for `ccx53` was rejected with
  `resource_limit_exceeded: dedicated core limit exceeded`; it created no server. The provider
  inventory remained empty. The available `cpx51` was selected before any scored work. Because its
  vCPUs are shared, this run is correctness evidence only and cannot satisfy a performance gate.
- Created: `2026-09-13T23:22:53Z`
- Server ID / label: `165735073` / `modeljars-bench-alora-v15-20260913`
- Address: `5.161.100.228`
- First observed SSH host-key fingerprint: Ed25519
  `SHA256:MdShnIGJBGdH40HRLFL2292PI04z6aR5jLTfzT3Kve8`, retained in the experiment-specific
  `known_hosts`; global host-key checking was not bypassed
- Network gate: Hetzner firewall `11618700` (`modeljars-alora-v15-ssh`) admits SSH only from the
  qualification workstation's observed `/32`; no inbound application port is exposed
- Host identity: Linux 6.8.0-138-generic, 16 single-thread AMD EPYC-Rome vCPUs, one NUMA node,
  AVX2/FMA, 30 GiB visible RAM, no swap, and no competing model process before the run
- Toolchain: Temurin 25.0.3+9, Gradle 9.4.1, Rust 1.96 only for compiling the repository's narrow
  native module
- Pre-run repository gates: `:models-runtime:check :models-bench:check` passed on the host
- Uploaded model, adapter, and source-record SHA-256 values matched the preflight values before
  execution
- Runtime: exact sequential Java scoring, 16 Vector workers, 256-bit Panama vectors; remote model
  process PID `6132`, launched under detached Gradle process `6029`
- Watchdog: local launchd job `org.modeljars.alora-v15-watchdog` was armed to validate and delete the
  exact server at the hard deadline
- First scored run: all 75 sequential observations completed on Models revision
  `2c8851e32c3d721db2f1b2345caeccfb4e4d09c3`, but report serialization failed on the optional
  adapter training-selection provenance. The partial report and full log were copied and matched
  remote SHA-256 values before teardown.
- Report-only correction: commit `2fab0dc6e4a52ea6f0b98e257f9f6e62946984a2` added the Jackson JDK
  8 datatype module and a regression test. `:models-bench:check` passed locally and on the host.
  No model, adapter, data, prompt, partition, score, threshold, or gate changed.
- Corrected run completed: `2026-09-14T01:01:36.920114918Z`; 39 minutes 20 seconds; threshold
  `4.8179874`; calibration 17/20 calls and 9/10 no-calls; held-out screen 28/30 calls and 15/15
  no-calls; all 75 observations physically shared their prefix.
- Overall V15 decision result: 45/50 calls and 24/25 no-calls. The frozen 47/50 positive floor is
  impossible, so V15 was rejected before dual generation and the sealed 300 cases remained closed.
- Copied evidence SHA-256 values: corrected report
  `799b1645ec510b6765ad50057ef379bd549e9dabc127516289189484e159643a`, corrected log
  `9b075c9df6f1f242d9a893a1540c853f5f51f84d1722a81ed46ade034ebea2a5`, first partial report
  `fe58482a9b02922436ecc164b8573645d1feebfb642f831dfb678edb45d99f65`, and first log
  `44fbe6be7dd32b9b8b7065758a1496fef0e1bca0e3452a10fe2a4e4e927957d7`.
- Server deletion action `655180087629089` started `2026-09-14T01:04:22Z` and succeeded at
  `2026-09-14T01:04:33Z`, 1 hour 41 minutes 29 seconds after creation. At the quoted rate this is
  about USD 0.76 of elapsed usage and at most USD 0.90 if rounded to two whole hours.
- The initial firewall deletion reached Hetzner while the server deletion was settling and returned
  HTTP 422. A read-only check then showed firewall `11618700` with the exact expected identity and
  no attachments; its deletion returned HTTP 204.
- Final audit at `2026-09-14T01:05:24Z`: the exact server and firewall both returned HTTP 404;
  Hetzner reported zero servers, volumes, floating IPs, primary IPs, firewalls, and snapshots;
  Vultr reported zero instances.
- After provider absence was proven, the launchd watchdog was unloaded and its exact plist and
  script were moved to macOS Trash. They are recoverable from Trash; they are no longer active.
