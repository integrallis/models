# MiniLM semantic applicability diagnostic

Status: **rejected as a hybrid decision component**.

This bounded diagnostic tested whether the qualified 20 MiB all-MiniLM-L6-v2 Q4_K_M encoder could
filter the unchanged V15 activated-adapter margin. Everything ran in-process through the Java 25
backend. It used only the 75 already-exposed V15 development inputs and therefore cannot qualify a
model.

The deterministic calibration rule selected:

```text
call iff maximum query/tool cosine > 0.3635590742647936
        and adapter margin > 6.2768946
```

| Partition | Correct calls | Correct no-calls | Balanced accuracy |
| --- | ---: | ---: | ---: |
| Calibration | 16/20 | 10/10 | 0.90000 |
| Screen | 24/30 | 15/15 | 0.90000 |
| Combined | 40/50 | 25/25 | 0.90000 |

The semantic gate removed every false call, but it also rejected ten valid calls. That is worse
than V15's 45/50 call result and well below the fixed 47/50 development floor. Adding a separate
embedding model would increase composition complexity without solving applicability, so this path
stops here.

- Embedding GGUF SHA-256:
  `2ec4cee28a27a9c973d5f5230930d6ef6e52694bd2bc71be26a9bef5b1d755e6`
- V15 records SHA-256:
  `944d137ba1e325e0f3daa922a73108de0da5ccf0292bcf4dbe4a565aa11fcb4c`
- V15 decision profile SHA-256:
  `799b1645ec510b6765ad50057ef379bd549e9dabc127516289189484e159643a`
- Diagnostic report SHA-256:
  `f30a91d55f95162925291e27244135b1d0e5519aa0dad84974b886df968a146a`

The first invocation was stopped after 38 rows because the historical source contains paired base
and adapter outputs for each input. The final loader deduplicates only byte-equivalent
messages/tools/prompts and rejects conflicting duplicate IDs; that aborted attempt produced no
retained report.
