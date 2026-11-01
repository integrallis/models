# Validating the round-trip-WER judge on Soprano, 2026-11-01

`MODEL-QUALIFICATION-PLAN-REVIEW.md` said to validate the TTS recipe against a model we already
ship before spending on a new one. Reproducing the shipped gate was the first half
(`../2026-11-01-soprano-tts-gate-reproduction/`) and it found that the gate already existed. This
is the second half: the one piece of that gate Paradee will need and Soprano does not — a
**round-trip-WER judge**, because `RandomNormalLike` in Paradee's graph makes the shipped gate's
waveform-cosine block undefined for it.

**This is not a qualification.** Ten sentences on one machine score the *pipeline*, not the model.
No threshold is set here and no model passes or fails.

## What was run

Corpus synthesized through the **released** route, not the worktree — `ModelJars.openSpeech` on
`org.modeljars:modeljars:0.1.71` with the catalog's Soprano Q8_0 coordinate, which is the route a
user gets (`SopranoCorpus.java`, run with `jbang`). One warm-up synthesis, not recorded. Ten
sentences in `sentences.txt`, ten wavs and `wav/manifest.tsv`.

```
jbang SopranoCorpus.java sentences.txt wav
python3 judge.py          # writes judge-result.json
```

Judge, pinned because the judge is part of the measurement (`versions.txt`):

| | |
| --- | --- |
| implementation | `faster-whisper==1.2.1` on `ctranslate2==4.8.4`, model `base`, int8, CPU, beam 5 |
| normalizer | `whisper-normalizer==1.0.0` `EnglishTextNormalizer` |
| scorer | `jiwer==4.0.0` |
| resampling | ours, stated: 63-tap Hamming-windowed sinc then 2:1 decimation, 32 kHz to 16 kHz |

Two deliberate differences from the Paradee card's judge, recorded rather than elided. Its judge is
"Whisper (base)"; this is a **CTranslate2 conversion** of that model, not the same binary. And the
resampling is ours: faster-whisper's own file decoding goes through PyAV, PyAV 19 dropped the
`metadata_errors` argument faster-whisper 1.2.1 passes it, and `av<15` will not build here without
ffmpeg headers — so the array is decoded and resampled locally, which makes the filter a stated
choice instead of a library default.

## Result: the pipeline works, and its failure modes are in the scoring, not the model

Corpus WER **0.0526** over ten sentences, six transcribed exactly, worst sentence 0.250.
Per-sentence RTF 0.293–0.429 on the released jar. Table generated from `judge-result.json`:

| # | WER | what the judge heard | what it was given |
| --- | ---: | --- | --- |
| 0 | 0.167 | `the gbm can speak for itself` | `the jvm can speak for itself` |
| 1 | 0.000 | `a small model that runs anywhere is worth more than a large one that does not` | `a small model that runs anywhere is worth more than a large one that does not` |
| 2 | 0.250 | `she sells seashells by the sea shore` | `she sells sea shells by the sea shore` |
| 3 | 0.111 | `a quick brown fox jumps over the lazy dog` | `the quick brown fox jumps over the lazy dog` |
| 4 | 0.000 | `portability is a feature not an afterthought` | `portability is a feature not an afterthought` |
| 5 | 0.000 | `12 catalog entries were qualified under the same prompt template` | `12 catalog entries were qualified under the same prompt template` |
| 6 | 0.000 | `measurement without provenance is just a confident opinion` | `measurement without provenance is just a confident opinion` |
| 7 | 0.000 | `the oracle cannot judge a model that draws its own noise` | `the oracle cannot judge a model that draws its own noise` |
| 8 | 0.000 | `reproducibility is the only claim a user can check` | `reproducibility is the only claim a user can check` |
| 9 | 0.091 | `publish the knowles because a failed hypothesis is still a result` | `publish the nulls because a failed hypothesis is still a result` |

Three of the four non-zero rows are **artifacts of scoring or of corpus design, not of synthesis**:

- **Row 2, 0.250 — compounding.** The judge heard "seashells", the reference said "sea shells".
  The audio is right and the normalizer does not join or split compounds. One sentence contributed
  a quarter of its own words as errors for a spelling convention, and on a ten-sentence corpus that
  is a large share of the corpus total.
- **Row 0, 0.167 — an initialism.** "JVM" came back "gbm". Letter-by-letter acronyms are hostile
  to this metric and say nothing about the model.
- **Row 3, 0.111 — a function-word substitution.** "The" heard as "a", the classic Whisper
  article slip.
- **Row 9, 0.091 — a genuine rare word.** "nulls" came back "knowles". This is the only row of
  the four that is plausibly about the audio.

The normalizer earns its place on row 5: "Twelve" against the judge's "12" scores 0.000 because it
maps number words, which is exactly the kind of free WER a hand-rolled normalizer would have
charged.

## What this settles for the gate design

It confirms the decision recorded as Recommendation 3 in
`modeljars/docs/CANDIDATE-PARADEE-8M-20261101.md`, and it quantifies the reason: **the floor has to
be relative to a comparator arm run on the same sentences with the same judge, never to a published
number.** A metric where a compounding convention costs 0.250 on a correct sentence cannot be
compared across corpora, normalizers or judge builds. Paradee's card reports 5.9% and 5.8%; this
pipeline reports 5.26% for Soprano on ten different sentences, and the two numbers are not
comparable in either direction.

Two corpus rules follow, and they are cheap to apply before any run rather than after:

1. **No initialisms or acronyms** in the sentence set unless the point is to measure them.
2. **Avoid words whose compounding is unsettled** ("sea shells", "web site"), or accept that the
   normalizer will charge for them and say so.

## What it does not settle

The shipped gate's oracle block is still the right gate for a deterministic vocoder, and this is
still only proposed as the substitute for the one case that block cannot express. Nothing here
measures Paradee, which has no port. And ten sentences is a pipeline check: the corpus for a
verdict is frozen before the run, on two datasets, per the working agreement.
