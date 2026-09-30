# Gold answers used by the logit-fusion G4 extractor tests

These files hold only item ids and gold answers (ARC also the option labels), derived by
`benchmark-results/2026-09-17-logit-fusion-study/prepare_fusion_data.py` from pinned public datasets.
They are test fixtures for answer extraction and scoring; no questions or solutions are included.

| File | Source | Revision | License |
|------|--------|----------|---------|
| gsm8k-test.jsonl, gsm8k-dev.jsonl | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) (config `main`; dev = 500 train rows, seed 20260917) | 740312add88f781978c0658806c59bc2815b9866 | MIT |
| arc-challenge-test.jsonl | [allenai/ai2_arc](https://huggingface.co/datasets/allenai/ai2_arc) (ARC-Challenge test) | 210d026faf9955653af8916fad021475a3f00453 | CC-BY-SA-4.0 |
| math500-test.jsonl | [HuggingFaceH4/MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500), a subset of Hendrycks et al. MATH | 6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be | MIT |

ARC-Challenge content is licensed CC-BY-SA-4.0 by the Allen Institute for AI; the ARC file here is
distributed under the same license.
