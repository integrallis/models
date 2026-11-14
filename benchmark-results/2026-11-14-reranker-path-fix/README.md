# Reranker path fixed: packed batching and no WordPiece trie put 0.5.21 at or ahead of 0.5.20

**Measured here.** Nothing in this directory is projected. Every number below is read from `runs/`
or `runs-notrie/` by `derive.py`; the tables are its output pasted verbatim.

    python3 derive.py runs
    python3 derive.py runs-notrie

## What was wrong

`benchmark-results/2026-11-14-reranker-path-bisect/` found two regressions in `v0.5.20..main` on the
pure-Java reranker path, confirmed at zero warm-up in its decisive run: `scoreAll` **1.31x slower**
(#440 padded every document to the longest and ran projections and attention over the padding, so
the batch cost more than the loop it replaced) and cold load **1.39x slower** (#435/#437 built a
WordPiece trie at construction with a `String` per character). A third apparent step - `score()`
1.27x - was the experiment's own warm-up no longer exercising that path and is withdrawn there.

## The fix, in two parts

1. **Packed batch.** `BertForwardPass.encodeBatchCls` lays the documents end to end with no
   padding: projections run over exactly the real tokens, where batching pays (each weight block
   dequantized once for every token of every document), and attention runs per document over that
   document's own tokens with the same operations as the single-sequence `attend`. No padded
   position exists, so no key is masked and the softmax row that threw in #455 cannot be built.
   `GgufRerankingModel.scoreBatch` passes ragged rows straight through.
   `BertForwardPassRaggedBatchTest` asserts the packed result equals encoding each document alone
   **bit for bit**.
2. **No trie.** The WordPiece trie's hot-path benefit was asserted by a synthetic micro-benchmark
   that prints a number and asserts nothing; on this model the pair p50 did not move with it. Its
   load cost was measured. Keying it by code point (arm `P-fix`) recovered half; removing it (arm
   `Q-notrie`) recovered all of it. The greedy longest-match of v0.5.20 returns, over an `int[]`
   builder instead of boxed ids.

## Session 1 - `runs/`: released, the candidate, and the fix with the code-point trie

```
host: AMD EPYC 7R13 Processor / 2 / 8
V-0520    83641c1886ae669def8eed5b62901c957cad8e3c  released v0.5.20
A-main    9db11325d940e8c62aa85f0792c1f9ec3dc8086b  main 9db11325 + #456 (the 0.5.21 candidate before this branch)
P-fix     8448b7d0b81ee736af34b5a9093548be5f0bbe7b  packed batch + code-point trie
runs per arm: {'V-0520': 5, 'A-main': 5, 'P-fix': 5}; warmups=5 pairIterations=60 batchIterations=20; artifact sha256=0752a92bc33289f3…

| metric | V-0520 | A-main | P-fix |
| --- | --- | --- | --- |
| coldLoadMillis | 158.29 [156.13..160.74] | 220.59 [219.02..243.32] (1.394x) | 189.68 [178.16..192.52] (1.198x) |
| pairP50Millis | 62.52 [61.59..63.26] | 80.53 [79.39..80.81] (1.288x) | 62.61 [61.52..62.94] (1.001x) |
| pairP95Millis | 98.94 [98.75..100.54] | 101.17 [99.68..101.60] (1.022x) | 100.67 [98.94..101.29] (1.017x) |
| batchP50Millis | 443.15 [442.33..451.52] | 603.69 [593.53..611.56] (1.362x) | 443.97 [437.65..446.02] (1.002x) |
| batchP95Millis | 639.98 [639.59..646.58] | 606.74 [594.06..611.90] (0.948x) | 444.35 [438.44..446.29] (0.694x) |
| batchP50DocumentsPerSecond | 13.54 [13.29..13.56] | 9.94 [9.81..10.11] (0.734x) | 13.52 [13.45..13.71] (0.998x) |
| maxRssMiB | 611.13 [594.75..714.97] | 696.05 [664.06..712.99] (1.139x) | 623.91 [593.85..744.54] (1.021x) |

scoreChecksum per arm (distinct, 9dp): {'V-0520': [-161.44592762], 'A-main': [-161.44592762], 'P-fix': [-161.44592762]}
every arm reproduces released v0.5.20 scores to 9dp: True
```

## Session 2 - `runs-notrie/`: released, the fix with the trie, the fix without it

```
host: AMD EPYC 7R13 Processor / 2 / 8
V-0520    83641c1886ae669def8eed5b62901c957cad8e3c  released v0.5.20
P-fix     8448b7d0b81ee736af34b5a9093548be5f0bbe7b  packed batch + code-point trie
Q-notrie  9db11325+q.patch                          packed batch, WordPiece trie removed
runs per arm: {'V-0520': 5, 'P-fix': 5, 'Q-notrie': 5}; warmups=5 pairIterations=60 batchIterations=20; artifact sha256=0752a92bc33289f3…

| metric | V-0520 | P-fix | Q-notrie |
| --- | --- | --- | --- |
| coldLoadMillis | 158.19 [154.80..164.13] | 181.16 [175.44..193.91] (1.145x) | 158.32 [156.10..165.77] (1.001x) |
| pairP50Millis | 62.41 [62.05..65.90] | 62.73 [61.58..62.97] (1.005x) | 61.70 [61.47..63.05] (0.989x) |
| pairP95Millis | 99.62 [99.01..100.69] | 99.70 [99.01..101.21] (1.001x) | 99.14 [98.91..100.67] (0.995x) |
| batchP50Millis | 447.55 [444.38..450.10] | 437.29 [435.22..445.29] (0.977x) | 437.25 [436.89..441.54] (0.977x) |
| batchP95Millis | 642.86 [640.52..646.98] | 437.43 [435.62..445.50] (0.680x) | 437.73 [437.41..443.35] (0.681x) |
| batchP50DocumentsPerSecond | 13.41 [13.33..13.50] | 13.72 [13.47..13.79] (1.023x) | 13.72 [13.59..13.73] (1.024x) |
| maxRssMiB | 608.72 [605.48..613.44] | 611.74 [608.86..632.74] (1.005x) | 616.59 [614.98..625.07] (1.013x) |

scoreChecksum per arm (distinct, 9dp): {'V-0520': [-161.44592762], 'P-fix': [-161.44592762], 'Q-notrie': [-161.44592762]}
every arm reproduces released v0.5.20 scores to 9dp: True
```

## Result, against released v0.5.20 (session 2, `Q-notrie`, what this branch ships)

| metric | v0.5.20 | this branch | ratio |
| --- | --- | --- | --- |
| cold load | 158.19 ms | 158.32 ms | **1.001** - the regression is gone |
| `score()` p50 | 62.41 ms | 61.70 ms | 0.989 |
| `scoreAll` p50 | 447.55 ms | 437.25 ms | **0.977** |
| `scoreAll` p95 | 642.86 ms | 437.73 ms | **0.681** |
| scores | identical to nine decimal places on every run of every arm | | |

`msMarcoMiniLmRerankerIntegrationTest` - pinned tokens, determinism, ONNX scale and ordering, batch
equals sequential - passes all four on `P-fix` (`qualification/`) and on `Q-notrie`
(`qualification-notrie/`).

No metric regresses against the released version. The batched path is now at least as fast as the
loop on the median and a third faster on the tail - the loop's p95 was one slow document; the packed
batch's p95 is its own median. The cold-load number equals v0.5.20's because the code that made it
slower is gone, not because its cost moved somewhere else.

## What this does not establish

- **Any other model or host.** Same caveats as the bisect: EPYC 7R13, AVX2, this quantization.
  A model whose documents are long and equal in length would see less from packing than this
  probe's six short ragged ones; the direction holds, the magnitude is this model's.
- **The batched path on the native kernel route.** `GgufRerankingModel` installs no kernel; the
  `bert`/`nomic-bert` families under `PureJavaBackend` were not measured here.
- **Anything about #438, #447 or #448.** They are in 0.5.21 and were not measured by this or any
  committed run; the CHANGELOG describes them as changes, not as performance.
