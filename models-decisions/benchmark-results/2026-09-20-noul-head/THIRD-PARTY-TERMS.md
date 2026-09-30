# Third-party terms review — TypeSafe Jev

**Reviewed 2026-09-20, before any API call was made.** Amendment 2 required this before the first
call; this is that record. **Not legal advice.** The reading below is an engineer's, and the two
open questions at the end need a qualified lawyer before anything is published.

## What the documents say

Source: `typesafe.ai/legal/mca` (Master Customer Agreement), reached from `docs.typesafe.ai/legal.md`.

| Activity | Finding |
| --- | --- |
| Benchmarking or evaluating the service | **No clause either way.** Not restricted. |
| Publishing performance results | **No clause either way.** Not restricted. |
| Comparative analysis against competitors | **No clause either way.** Not restricted. |
| **Output used to train, distil or imitate** | **Prohibited — MCA 2.3(b)** |
| Confidentiality | 14.1 designates Documentation and non-public information about the Services as Confidential. Performance figures are not named. |

MCA 2.3(b), verbatim in the part that binds us: use of the Services or any Output to *"perform model
distillation, train a model to imitate the output of the Services, or develop ... a similar or
competing product"*.

Separately, the Terms of Use prohibit automated scraping of the **Site** and reverse engineering the
proprietary software. Neither touches ordinary API use.

## Why the pre-registered design already complies

Amendment 2 scoped the arm to **evaluation only** on 2026-09-20, before these terms were read, for
licence-hygiene reasons: our teacher stack is deliberately Apache-2.0 and MIT so no trained weight
carries a licence question. That boundary is exactly what 2.3(b) requires.

- The head is fitted on **hidden states from our own Granite 4.1 3B (Apache-2.0)** and on **labels
  from published corpora** pinned by digest. No third-party output reaches it.
- Jev output is used to compute accuracy, calibration and latency **for Jev**, joined to corpus
  labels by item id. It is never a label, never a soft target, never a feature.

## The boundary is structural, not advisory

Three tests enforce it (`ExternalArmIsolationTest`):

1. The compiled bytecode of every class that fits a head or a temperature is scanned, and the test
   fails if any of them so much as names the `external` package.
2. `ExternalVerdict` holds an id, a probability and a service name. It has **no hidden state, no
   logits, no feature vector** — there is nothing on it a head could be fitted to.
3. `LogisticHeadTrainer.fit` has exactly one overload, taking `float[][]` hidden states and `int[]`
   corpus labels. There is no signature that accepts probabilities, so a third-party verdict cannot
   be passed as a soft label, which is the shape distillation would take.

The provenance trail is itself the defence. The pre-registration was committed **before access was
granted**, the corpora are pinned by digest, the teacher is our own Apache-2.0 base, and every
commit is dated. That the head was trained without Jev is demonstrable from the repository rather
than merely claimed.

## Two questions for a lawyer, not for me

1. **"develop ... a similar or competing product."** We are building a System One decision tier,
   which is plainly similar in category. My reading is that 2.3(b) attaches to *use of their
   Output*, not to independent development, and our head never touches their output. That reading
   should be confirmed by counsel before any public comparison is made, not taken from an engineer.
2. **14.1 confidentiality.** Comparative results are unrestricted by the MCA, but measured latency
   and cost figures are not published by TypeSafe. Whether measuring them ourselves makes them
   "non-public information about the Services" should be checked before publication.

**Until both are answered, run the arm and keep the numbers internal.** Nothing here should be
published on an engineer's reading of a contract.
