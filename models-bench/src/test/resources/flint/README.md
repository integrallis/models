# Captured follow-up regression

`truncated-follow-up.txt` is the exact `followUp.output` from the completed remote Flint report
in `benchmark-results/2026-10-29-flint/qualification-attempt/results/tool-qualification.json`.
The runtime source was `7541b1a4676ffadcfd728aec39753258341f27f4`; the model and host are pinned
in the adjacent campaign inputs. Generation used the suite's 96-token follow-up limit.

- Original report SHA-256: `7a202734168486662cee79b66d6b88aaec443836c59251c967cd81c3eae78815`
- Captured output SHA-256: `58deaf42884611ba1ccaaf3f1936fdc6d0f020e8a076c425b71b561ca1739ef6`

The v1 scorer found the weather facts inside `<think>` and incorrectly passed the follow-up.
The visible answer ends at “shows some interesting” and contains neither required fact.
The v2 regression test rejects that response, closed or unclosed reasoning-only responses,
and nested reasoning, while accepting a complete visible answer. The original evidence is
preserved unchanged. No inference or model download is performed by this regression test.
