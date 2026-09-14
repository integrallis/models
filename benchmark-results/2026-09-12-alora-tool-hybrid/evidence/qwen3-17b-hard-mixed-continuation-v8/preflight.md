# Qwen3 1.7B hard-mixed continuation preflight

Status: **frozen before training; candidate rejected after the fixed smoke screen**.

This bounded experiment continues the strongest v4 adapter on a split-preserving mixture of real
positive tool calls and hard, independently pinned BFCL live requests for which none of the offered
tools apply. The prior easy-negative continuation reached zero false calls but destroyed valid tool
calling; this run retains 80% positive examples specifically to prevent that failure. Qualification
data and thresholds are unchanged and were not used to choose the mixture.

- Base: `Qwen/Qwen3-1.7B`
- Base revision: `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`
- Initial v4 adapter SHA-256:
  `dcaeab65a51ebf9d98c97f7203e5890cb3e24ae592ba1f2358cbc1d8e46895bd`
- Initial v4 training-manifest SHA-256:
  `5d73180fd1bbc13ef7664133cb2c3e5762025c408e35f1aac0ee9fbcb62d63bc`
- Positive parent manifest SHA-256:
  `7b2ef717159ea9900a87ae7f81ecbef0ac1e571d0f35c7aa966d294ced8bdcc6`
- Hard-negative parent manifest SHA-256:
  `b0a7eac21a634d6fbd071d7eb7ebdf51f2da8bc7e15e2a9c19c851a46f39f291`
- Merge script SHA-256:
  `1e095f6f3644eea3b2e8e9f6febbe03217bffdc3ff97bc0e2eafa039b3a698eb`
- Continuation trainer SHA-256:
  `bcf5942e8689aa92c02998ce858b56580c728c1fce7e23451578b5c75e05ad8c`
- Independent evaluator SHA-256:
  `74b41820632773f915f5e0f1c643bf51e139fcae989c3a3817040fa96b82f645`
- Merged manifest SHA-256:
  `6abf29b6040a2831f52ad0b77a7276a2213e5c6bc775c4c68b40e3a3dc9b788c`
- Train: 2,400 calls plus 600 hard no-calls; SHA-256
  `c46b606c74a3d1bda38c6f50a55854bf9112f0ae9d92993eb32e577c211f3002`
- Validation: 400 calls plus 100 hard no-calls; SHA-256
  `7006d89f1be432424889f8fbc846ff348cf2770a56f409c9e38361ae4857cfb0`
- Adapter contract: rank 16, alpha 32, zero dropout, all seven attention/MLP projection families,
  with the exact v4 adapter and manifest verified before loading
- Continuation: one epoch, learning rate `2e-5`, batch size 1, gradient accumulation 16, maximum
  length 1,024, seed `20260916`, BF16, SDPA, gradient checkpointing
- Remote output: `/opt/modeljars-alora-v3/qwen17-hard-mixed-v8`

The fixed diagnostic gates remain 100% syntax, 100% schema, at least 85% exact tool calls, no worse
than the unadapted base, and at most 5% false calls. A diagnostic failure rejects this run without a
full qualification, Java packaging, catalog entry, or release.

## Result

Training completed with adapter SHA-256
`5a7541ef7074a76b3b8a8018172729cc7eb10518192218b04873dea723e83d94`. On the frozen 25-case-per-
category screen, the adapter reached 100% syntax, 100% schema, and 86% exact tool calls, equal to
the unadapted base's 86%. It made false tool calls on 20% of irrelevance cases, above both the
base's 12% and the fixed 5% ceiling. The candidate is therefore rejected; no full qualification,
Java behavior gate, packaging, catalog entry, or release was attempted.

The copied evidence hashes are:

- training manifest: `608f558b97c50171d17dc7259319528eed2b7a9edd5315db868a22f03550454e`
- smoke report: `d27a101bfd3ed5f9f826dd8e0da6a5a0d15e01776c1f78fe1e7105ea66437dbe`
- smoke records: `7e28c65fa7a2bf9305430e17f66df04cfe9c86db7f0787a076a7792136155c12`
