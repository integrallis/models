# V16 Java development-gate host lifecycle

- Experiment: frozen 75-case production-Java Q4 hidden-applicability decision gate for
  `qwen3-17b-hidden-applicability-v16`
- Provider / region / exact plan: Hetzner Cloud `ash`, `cpx51`
- Hardware: 16 shared x86 vCPUs, 32 GB RAM, 360 GB disk, Ubuntu 24.04 x86-64
- Live price at preflight: USD 0.4479/hour
- Expected use: under 90 minutes for setup, complete sequential scoring, evidence transfer, and
  teardown
- Maximum authorized spend: USD 0.90; two-hour hard deletion ceiling
- Created no earlier than: `2026-09-14T02:30Z`
- Hard deletion deadline: `2026-09-14T04:30Z`
- State before creation: zero Hetzner servers and zero Vultr instances
- Resolved image: Ubuntu 24.04 x86 image `161547269`
- Verified SSH key: Hetzner key `114663461` (`vectors-bench-v2`), matching the established local
  benchmark identity
- Exact Models revision: `f21b108d2214a1509a274442c93476d2636252bc`
- Model SHA-256: `061b54daade076b5d3362dac252678d17da8c68f07560be70818cace6590cb1a`
- Adapter SHA-256: `f1a4a1bc15e3942ed5a3fbe2c4750f39a8b24804e20566242ec740c5ce851c21`
- Applicability-head file SHA-256:
  `67b1f5fd231f924554d02daf450ce049fbccd24c084f7730d4ce2724ca234d1e`
- Frozen records SHA-256:
  `944d137ba1e325e0f3daa922a73108de0da5ccf0292bcf4dbe4a565aa11fcb4c`
- Correctness stop conditions: delete immediately if any artifact identity, Java arithmetic,
  physical shared-prefix identity, 47/50 call floor, or 24/25 no-call floor fails
- Performance claims: prohibited; shared vCPUs make this a correctness gate only
- Sealed qualification data: prohibited until this gate and the following exposed dual-generation
  gate both pass
- Local evidence destination: this directory

The exact server/firewall IDs, address, host key, measured hardware, run hashes, deletion actions,
elapsed cost, and closing provider inventory are appended as they occur.

## Actual run

- Server: `165752751`, `modeljars-alora-v16-java-20260914`, IPv4 `5.161.100.228`
- Firewall: `11619048`, `modeljars-alora-v16-java-ssh`; SSH only from the qualification
  workstation's observed `/32`
- Created: `2026-09-14T02:35:09Z`
- SSH host-key fingerprint:
  `SHA256:2R2bRrrwJY2l2wtLq17VfZ8QFNWZHBVJauunKv7bcGM`
- Measured CPU: 16 online single-thread vCPUs reported as AMD EPYC-Rome under KVM, AVX2/FMA
- Measured memory: 32,859,295,744 bytes, with no swap
- Measured root filesystem: 362,533,470,208 bytes
- JDK: Eclipse Temurin 25.0.4.1
- Rust toolchain: 1.98.1, used only because the repository's existing aggregate build produces the
  native-platform artifact; the scored CLI explicitly instantiated `PureJavaBackend`
- Fresh-image correction: Ubuntu did not contain a C linker. The first repository check stopped at
  `backend-native:cargoBuildRelease` before inference. `build-essential` was installed, after which
  `:models-runtime:check :models-bench:check` passed. This did not change the frozen experiment.
- Uploaded model, adapter, head, records, and Git revision all matched the preflight identities.
- Java log SHA-256:
  `ec3346dac3ec0b9abf60d8548413c81b32e16de8e421f9ee810cf7c213a6f619`

The full 75-case command started as PID 6108. The first three completed observations all proved
physical shared-prefix identity. The first two no-call cases were false positives, however. Since
the frozen gate permits at most one false call among all 25 no-call cases, V16 was already
mathematically unable to pass. PID 6108 was validated by its command line and terminated at that
point. No threshold or artifact was changed, no partial report was promoted, and the sealed
300-case qualification data remained unopened.

## Teardown

- Exact server deletion requested: `2026-09-14T02:48:52Z`
- Server `165752751` subsequently returned HTTP 404.
- The exact detached firewall `11619048` was then deleted.
- Closing audit at `2026-09-14T02:49:32Z`: zero Hetzner servers, volumes, floating IPs, primary
  IPs, firewalls, or snapshots; zero Vultr instances.
- Billable lifetime: 13 minutes 43 seconds, approximately USD 0.10 at the recorded hourly rate.
- The LaunchAgent watchdog was unloaded only after provider absence was proven. Its exact plist and
  script were moved to Trash and remain recoverable.
