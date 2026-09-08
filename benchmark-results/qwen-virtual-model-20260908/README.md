# Qwen virtual-model qualification — 2026-09-08

## Infrastructure preflight

- Experiment: compare one Qwen3 1.7B Q8_0 control with a Qwen3 0.6B Q4_0 chat / Qwen3
  1.7B Q8_0 tool-loop virtual model.
- Provider / region / exact plan: Hetzner Cloud, `ash` (Ashburn, Virginia), `ccx33`;
  8 dedicated x86 vCPUs, 32 GB RAM, 240 GB disk.
- Expected hourly and minimum charge: USD 0.266/hour, hourly billing.
- Maximum authorized spend: USD 1.064 (four hours).
- UTC creation deadline: 2026-09-08T10:30:00Z.
- UTC deletion deadline: 2026-09-08T14:30:00Z.
- Correctness gate: all six protocol turns pass in every one of three fresh control and three fresh
  hybrid JVM processes.
- Performance gate: hybrid median end-to-end generation time improves on control by at least 5%.
- Local evidence destination: this directory.

The run order is counterbalanced: control 1, hybrid 1, hybrid 2, control 2, control 3, hybrid 3.
Model loading is reported separately and excluded from the performance comparison. The two pinned
model files, exact Models revision, protocol digest, JVM/vector settings, hardware identity,
per-turn cache metrics, and process IDs are recorded by the harness.

## Execution environment

- Hetzner server: `165117087`, `modeljars-bench-hybrid-20260908`, public address
  `5.161.100.228`.
- Host key: ED25519 `SHA256:IWM9ELVfm/wBC33bmGE2JMtviXSO2y9j9T/zyomURF4`.
- Hardware: eight dedicated x86 vCPUs presented as four-core/eight-thread AMD EPYC Milan, 32 GB
  RAM, 240 GB disk, AVX2, one NUMA node.
- OS/JVM: Ubuntu 24.04, Linux `6.8.0-138-generic`, Eclipse Temurin `25.0.4.1`.
- Runtime: Panama Vector API, 256-bit active vectors, eight GGUF threads, persistent executor,
  default Vectors toggles.
- Models revision: `4fcd96eba6bce15e74a89ed14b528f9ba5dd803a`.
- Protocol SHA-256: `d770c2aef09bed0c6522756cd9c8ef480585a59af6b6f439123179d57630fd3d`.
- Lifecycle: provisioned at approximately `2026-09-08T10:13Z`; deletion requested at
  `2026-09-08T10:36:48Z` and absence verified immediately afterward. The server existed for less
  than one hour, so its compute-charge upper bound is USD 0.266.
- Decommission audit: the server ID is absent, and no experiment-owned volume, floating IP,
  primary IP, or snapshot remains.

The remote qualification and comparison source files matched the local frozen source hashes:

```text
36d854895b810f9d0867404e06de1fa0b56d5fcd2a0c63e58ac59e6a7fa94c41  VirtualModelQualificationCli.java
9f0d766bcb5d5f2027d4e889ed757c4138bf2be45baa7afc6d720c7453c92307  VirtualModelQualificationComparisonCli.java
```

## Result

All six turns passed in all six fresh JVMs. The virtual model therefore passed the semantic
composition gate: the tool-capable member handled the complete tool loop, both members preserved
their own cache lineages, and the opaque `mem-2048` result survived both switches.

| Arm | Run 1 | Run 2 | Run 3 | Median |
| --- | ---: | ---: | ---: | ---: |
| Qwen3 1.7B control | 83,672 ms | 75,672 ms | 75,538 ms | 75,672 ms |
| Qwen3 0.6B / 1.7B virtual model | 90,921 ms | 91,728 ms | 91,223 ms | 91,223 ms |

The virtual model was **20.55% slower**, failing the predeclared requirement for at least a 5%
improvement. The smaller member reduced median cold chat from 30,293 to 10,760 ms and warm chat
from 9,021 to 2,307 ms. Those savings were erased when the idle 1.7B member had to process the
canonical history: its first tool turn rose from 10,147 to 44,369 ms. Returning to its cache later
also cost 14,862 ms versus 6,966 ms in the control. Peak process RSS increased from approximately
2.47 GB to 3.19 GB because both weight sets and both session states were resident.

**Decision:** reject this pair as a publishable composite. No ModelJars marker or hybrid model is
created. The next performance experiment must reduce or schedule member catch-up; correctness
alone is not sufficient.

## Copied evidence SHA-256

```text
f553ee64dbc10fd787b192d6568c8af59e4e16d62246fc0bd706ecc799c89c9c  comparison.json
b7a67c085396456ef1761a25497deff985c2055ed3dc1852951f041f19962a05  control-01.json
87ead9506df0abeab77797a5bad33f2df5accdbebc8b68c166db801375a70489  control-02.json
b25f04cc64d4b50cae31ebefa6e50d61bf1305a1a84dfe2aa287bd1bcdde794d  control-03.json
1a45dbf71f585aa758b2e812b5441c891640452c2392e34204561cbb8c5c1ebf  hybrid-01.json
0a667edf07c1fcdfef548e09657bb9bdaa951aff1a039404b3e2a52a3be71c  hybrid-02.json
af095b099f0bbc181ab231d77b1947dc37f5cedc39053a78c437dbd9f0fd60e6  hybrid-03.json
```
