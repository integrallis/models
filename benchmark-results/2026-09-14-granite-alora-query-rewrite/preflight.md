# Granite 3.2 aLoRA query-rewrite qualification preflight

- Experiment: verify an upstream IBM Granite 3.2 8B query-rewrite aLoRA as the first candidate
  composed model with an exact, physically shared base KV prefix.
- Provider / plan: Hetzner Cloud, `cpx62`, `fsn1`, 16 shared x86-64 vCPU, 32 GiB RAM, 640 GB disk.
- Price / ceiling: USD 0.2452 per hour; one host, maximum four hours (USD 0.99) including setup.
- Creation deadline: 2026-09-14T22:30:00Z. Deletion deadline: 2026-09-15T02:30:00Z, or immediately on
  a failing correctness gate.
- Provisioned resource: Hetzner server `165969464` (`modeljars-granite-alora-20260914`), provider
  firewall `11624612`, IPv4 `5.75.251.129`. SSH is restricted to the operator workstation's observed
  `/32`; the server and firewall are both temporary and must be deleted together after evidence copy.
- Correctness gates: exact artifact and upstream-adapter identities; tokenization and first-marker
  boundary match IBM's aLoRA implementation; Java shared and recomputed branches produce identical
  token IDs; Java output matches an independent reference run; the documented query-rewrite cases
  meet the frozen task criteria.
- Performance gates: shared prefix references identical immutable storage, reports less unique KV
  state than recomputation, and measures the 256 / 1024 / 4096-token crossover without an external
  runtime in the product process.
- Evidence destination: this directory. The remote checkout and all model bytes are disposable after
  report hashes are copied and verified locally.

## Execution record

- The original Hetzner `cpx62` host and its dedicated firewall were deleted after the first 256-token
  control established physical sharing. Its retained log is `cpx62-256-baseline.log` (SHA-256
  `2e6836ee0e172560a0b8c8c9171c782ee90548df2916032f5763022ee7b06d19`). It measured recomputed
  handoff-to-first-token at 211,093.9 ms versus 119,100.7 ms shared, with unique inference state
  falling from 251,688,960 to 131,088,000 bytes.
- The four-tier run moved to AWS `c7i.4xlarge` (`i-0a51cf753c1122389`), Intel Xeon Platinum 8488C,
  16 vCPU / 30 GiB RAM, with the supported 512-bit Vector API width explicitly recorded. The
  instance and its SSH-only security group `sg-03876de3ffeb2bcd6` were deleted after evidence copy.
- `c7i4-512-prefix-sharing.log` (SHA-256
  `25df970c3f8233d5e557b59820d3c9fb09a5859739c0133f1ab9243c4b5ada61`) records physical sharing
  and token-exact output at every required tier: 256: 260,560.6 ms recomputed / 148,578.6 ms shared;
  1024: 1,073,533.2 / 553,470.3 ms; 4096: 6,597,127.7 / 3,367,132.8 ms. Unique inference state at
  4096 fell from 4,027,023,360 to 1,389,532,800 bytes.
- The first run completed all inference but failed only while serializing the `Optional` crossover
  field in its JSON evidence. Commit `84fad94` adds the JDK8 Jackson module and a regression test;
  the corrected benchmark must be re-run to produce its machine-readable report.
- The Java-only query-rewrite smoke from the IBM model-card conversation passed at temperature zero:
  a physically shared 42-token prefix produced `{"rewritten_question":"Who is the CEO of
  Microsoft?"}`. This confirms marker placement and adapter behavior, but is not the independent
  multi-case semantic/reference gate required for qualification.
