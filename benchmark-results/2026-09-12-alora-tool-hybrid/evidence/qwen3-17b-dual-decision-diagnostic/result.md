# V19 result: base/specialist dual decision rejected

V19 tested whether the exact base model's call/no-call margin adds useful information to the
activated specialist's margin when both decisions retain the same physical KV prefix. It does not.
The calibration-only selector chose:

```text
score = specialist margin + 1.0 * base margin
call when score > 50.940744400024414
```

That rule reached 40/50 correct calls and 25/25 correct no-calls across the already-exposed V15
development cases. V15's specialist-only rule reached 45/50 calls and 24/25 no-calls. V19 therefore
reduces call recall and misses its frozen requirement to exceed V15 while retaining at least 24/25
no-calls. This decision family is rejected. No fresh qualification data, generation gate,
packaging, catalog entry, or release was opened.

| Metric | Calibration | Held-out screen | All exposed cases | Advance requirement |
| --- | ---: | ---: | ---: | ---: |
| Correct calls | 16/20 | 24/30 | 40/50 | more than 45/50 |
| Correct no-calls | 10/10 | 15/15 | 25/25 | at least 24/25 |
| Balanced accuracy | 0.9000 | 0.9000 | 0.9000 | not a substitute for count floors |
| Physical shared-prefix identity | 30/30 | 45/45 | 75/75 | 75/75 |

The candidate weight set and threshold-selection order were frozen before host creation. Weight and
threshold selection used only the 30 calibration observations; the 45-case screen could not affect
them. An independent `jq` recount from the copied observations reproduced every reported count.

The scorer constructed `PureJavaBackend` directly. The repository-owned Rust kernel project was
compiled by the existing Gradle dependency graph, but no native backend or Rust kernel participated
in either margin. No external inference engine or server was installed or used.

## Bound evidence

| Item | Value |
| --- | --- |
| Models scorer revision | `487986fcaafd4301124ded60d279fafd42834e76` |
| Source records SHA-256 | `944d137ba1e325e0f3daa922a73108de0da5ccf0292bcf4dbe4a565aa11fcb4c` |
| Base GGUF SHA-256 | `061b54daade076b5d3362dac252678d17da8c68f07560be70818cace6590cb1a` |
| Activated adapter SHA-256 | `f1a4a1bc15e3942ed5a3fbe2c4750f39a8b24804e20566242ec740c5ce851c21` |
| `dual-decision.json` SHA-256 | `9c59acb4d8e19e5d4414f63183cbac4c72d66b2f4b3bd18376ca110d99c92339` |
| `dual-decision.log` SHA-256 | `dc7e562833e9ca2064eaf133b479dfe3e21527bfcc935cb7133eb56bc432684a` |

The report records Java 25.0.4.1 on 16 vCPU AMD EPYC Rome with 32 GB RAM. The 75 scored turns took
2,457,121 ms in aggregate; the complete Gradle diagnostic took 41 minutes 9 seconds.
