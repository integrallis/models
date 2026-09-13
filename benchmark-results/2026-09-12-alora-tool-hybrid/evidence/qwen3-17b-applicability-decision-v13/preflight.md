# Qwen3 1.7B applicability-decision V13 preflight

Status: **frozen before GPU provisioning or candidate output**.

V13 is a paired causal continuation from the exact retained V9 rank-32 adapter. Both arms use the
same weights, data, order, optimizer, scheduler, seed, and endpoint. The control uses the unchanged
completion language-model loss. The treatment adds only a class-balanced call/no-call loss at the
first causal decision token. Only the treatment may advance to generation qualification.

## Exact ancestry and inputs

- Base: `Qwen/Qwen3-1.7B` revision
  `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`
- V9 adapter SHA-256:
  `f1a4a1bc15e3942ed5a3fbe2c4750f39a8b24804e20566242ec740c5ce851c21`
- V9 training manifest SHA-256:
  `d0620df4860c879f6d3f6e5573168bb08afc0b48d954040825e1317972f7de47`
- Frozen V8 merged manifest SHA-256:
  `6abf29b6040a2831f52ad0b77a7276a2213e5c6bc775c4c68b40e3a3dc9b788c`
- Raw train split SHA-256:
  `c46b606c74a3d1bda38c6f50a55854bf9112f0ae9d92993eb32e577c211f3002`
- Raw validation split SHA-256:
  `7006d89f1be432424889f8fbc846ff348cf2770a56f409c9e38361ae4857cfb0`
- After exact 1,024-token filtering, train has 2,840 rows: 2,356 call and 484 no-call;
  source-line identity SHA-256
  `ceab03a876d41ada5e7a3f99cad4357e0b70a478b6f8c20f82da7aa0a1626941`.
- Validation has 471 rows: 390 call and 81 no-call; source-line identity SHA-256
  `1ce289514ff5ddbe72a34e240196925b8d341058e66278d627b477f5fc622f51`.

The source files were downloaded again from their pinned revisions and matched their historical
hashes. The current formatter correctly rejects ambiguous duplicate callables and therefore no
longer reproduces V8. `prepare_v9_compatibility.py` restores only the historical duplicate-callable
normalization for reconstruction. The resulting 12,000/1,000 parent split bytes match V9 exactly;
the current formatter remains fail-closed for every new corpus.

## Objective

The exact tokenizer resolves the shared output prefix as `[151657, 198]`, the contextual call token
as `4913`, and the contextual no-call token as `19536`. Token `19536` includes the following newline;
V13 therefore discovers these IDs from full canonical contexts rather than assuming standalone
`{"` and `[]` tokenization.

For the logit immediately before that decision token:

```text
L = L_SFT + lambda * w_y * CE([z_call, z_none], y)
```

- Control `lambda`: `0.0`; `combined_loss` returns the original SFT loss object unchanged.
- Treatment `lambda`: `0.1`.
- Call weight: `2840 / (2 * 2356) = 0.6027164685908319`.
- No-call weight: `2840 / (2 * 484) = 2.9338842975206614`.
- Weights multiply the per-example loss manually; batch-size-one weighted-mean cancellation is not
  permitted.

## Frozen training contract

- Rank/alpha/dropout: 32/64/0 over all seven attention and MLP projection families.
- One epoch; learning rate `2e-5`; cosine schedule; 3% warm-up.
- Batch size 1; gradient accumulation 16; maximum length 1,024.
- Seed `20260918`; BF16; SDPA; TF32 disabled; gradient checkpointing enabled.
- Original canonical Qwen tool prompt; no V12 policy, masking, DPO, decoder threshold, classifier,
  or runtime rule.
- The 32-step screen stops through a trainer callback while the scheduler retains the full one-epoch
  horizon. `max_steps=32` is prohibited because it would change the cosine schedule.
- Passing screen arms restart from the same V9 bytes and run to the fixed one-epoch endpoint; no
  intermediate checkpoint or seed may be selected.

## Frozen paired screen

After exactly 32 optimizer steps, both arms evaluate all 471 validation rows with teacher forcing.
The treatment must satisfy every condition:

- at least nine additional correct no-call decisions among 81;
- no more than seven fewer correct call decisions among 390;
- higher balanced applicability accuracy;
- ordinary validation language-model loss no more than 10% above control;
- finite losses and gradients;
- exact initial-logit equality between arms;
- exact base versus disabled-adapter logits before and after training; and
- exact trained versus saved/reloaded adapter logits.

Failure stops V13. A pass continues both arms to the fixed endpoint and repeats the same directional
checks. Only then may the treatment see the already exposed 25 simple, 25 multiple, and 25
irrelevance generation screen: 75/75 syntax, 75/75 schema, at least 43/50 positive exact, no worse
than a freshly rerun same-prompt base, and at most 1/25 false calls. The sealed 300-case window stays
closed until that complete pass. All JVM, Spring AI, LangChain4j, long-context, physical-sharing,
crossover, NMT, memory, performance, packaging, clean-host, and published-artifact gates remain
unchanged.

## Local preflight result

At `2026-09-13T19:56:07Z`, the complete experiment suite passed 106 tests. The CPU preflight passed
all 3,311 usable examples, found zero overlap with 735 static BFCL queries, proved that the auxiliary
gradient touched only token logits 4913 and 19536 at the correct causal position, verified the
gradient signs and zero-coefficient identity, and reproduced a serialized tensor fixture exactly.

- Trainer SHA-256: `9aa594e6542e725d7f307297d0a4742dd4d2947998d046cbef95d0fdb2b3682b`
- Trainer tests SHA-256: `d98eabab7adf6e7892a0fb23765d8f5426614c9ddbd51c7811d23012651f4b4c`
- Paired gate SHA-256: `b58401caab3f858dd5a4a1118db208e2e31e19d3bf6a1ef8f9b89e64b90915ef`
- CPU preflight SHA-256: `0b4eac37d8998f5d36aee2c9476f1eadf6249b85414f32d9307887c866dbf4a2`
- Preflight report SHA-256:
  `f4f2ee135cff4e16f6278747c955ea7953e59925708caac872f3346726d8603e`
- Committed normalized preflight evidence SHA-256:
  `17cf4c46525705cd6ef147b5f6275079ec825fed8711041bf66b46fc9bd5ea18`

GPU budget is capped at two hours for the paired screen and fixed endpoints, with an absolute
three-hour deletion ceiling. At the prior A40-8Q rate this is approximately USD 0.58 expected and
USD 0.86 maximum. Generation or JVM performance work is prohibited on the training host.
