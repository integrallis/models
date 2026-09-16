# Base/specialist dual-decision diagnostic preflight

Status: **frozen before host creation or dual-branch output**.

This is a bounded diagnostic over the 75 already-exposed V15 development inputs. It tests whether
the exact base model's call/no-call margin complements the activated specialist margin when both
branches retain the same physical KV prefix. It cannot qualify or publish a hybrid model.

## Immutable inputs and procedure

- Models revision: `487986fcaafd4301124ded60d279fafd42834e76`
- Base GGUF SHA-256:
  `061b54daade076b5d3362dac252678d17da8c68f07560be70818cace6590cb1a`
- Activated adapter SHA-256:
  `f1a4a1bc15e3942ed5a3fbe2c4750f39a8b24804e20566242ec740c5ce851c21`
- Exposed records SHA-256:
  `944d137ba1e325e0f3daa922a73108de0da5ccf0292bcf4dbe4a565aa11fcb4c`
- Call/no-call token IDs: `4913` / `19536`
- V15's existing 30-case calibration and 45-case screen partition is unchanged.
- For each input, Java scores both the exact base and activated branches at `<tool_call>\n` after
  evaluating the common prefix once and verifying physical storage identity.
- Candidate score is `specialistMargin + w * baseMargin`, where
  `w in [-2, -1, -0.5, 0, 0.5, 1, 2]`.
- Weight and threshold selection uses only the 30 calibration rows, maximizing balanced accuracy,
  then correct no-calls, correct calls, smaller absolute weight, positive weight, and higher
  threshold. The 45 screen rows cannot influence selection.

The diagnostic advances only if all 75 prefixes are physically shared. A useful result must exceed
V15's combined 45/50 calls while preserving at least 24/25 no-calls; otherwise this decision family
stops.

## Runtime and infrastructure boundary

- Provider / region / plan: Hetzner Cloud `ash`, `cpx51`
- Live price: USD 0.4479/hour
- Expected duration: under one hour
- Hard deletion deadline: `2026-09-14T12:41:13Z`
- Maximum compute ceiling: two hours / USD 0.90
- Opening inventory: zero Hetzner servers and zero firewalls
- Image: Ubuntu 24.04 x86 `161547269`
- SSH key: `vectors-bench-v2` ID `114663461`, fingerprint
  `60:4e:57:38:96:48:c5:45:73:10:9d:37:f7:4f:74:cf`

The scorer constructs `PureJavaBackend` directly. No external inference server may be installed or
used. The repository's owned Rust crate may compile as an existing Gradle dependency, but no native
backend or Rust kernel participates in either margin.
