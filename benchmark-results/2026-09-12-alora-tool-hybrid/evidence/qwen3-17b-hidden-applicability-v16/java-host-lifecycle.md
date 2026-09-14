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
