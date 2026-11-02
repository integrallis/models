# Qualifying Paradee-8M v1.2: protocol, frozen before data

**Status 2026-11-01, evening: protocol frozen; the WER passes of both arms are running; no verdict
exists.** Every threshold and every corpus was fixed in this file before the first synthesis so
that none could be chosen after seeing numbers. The candidate record that decided to run this is
`modeljars/docs/CANDIDATE-PARADEE-8M-20261101.md`; the artifact triage is
`modeljars/docs/model-evidence/paradee-8m/TRIAGE-v1.2.md`.

## State of the work (updated 2026-11-02, after the first candidate WER)

- **The first candidate WER on our phonemes failed the WikiText cell for the predicted reason.**
  `af_heart`, pass 1: 7.56 % against a floor of 6.18 % (comparator 5.68 %). On the publisher's
  phonemes the same pass scored 5.60 %, so the executor is not the cause; the 4.1 % of WikiText
  words the lexicon path drops are. Harvard `af_heart` passed at 4.09 % (floor 4.59 %), Harvard
  `bf_isabella` at 3.32 % (floor 3.88 %). Those passes are kept under
  `evidence/devbox/candidate-no-fallback/` as the record of the configuration without a fallback.
- **The fix is the fallback the port did not have.** The publisher's pipeline hands unresolved words
  to eSpeak NG (GPL-3.0, not bundlable). `models-audio` now bundles two small letter-to-phoneme
  models of our own, `g2p-us.onnx` and `g2p-gb.onnx`, trained by `g2p/train_g2p.py` on misaki
  0.9.4's gold and silver dictionaries (Apache-2.0; DEFAULT pronunciations; 174,945 US and 188,542
  GB words of letters and apostrophes, 2 % held out): a character embedding, each letter repeated
  three times, a two-layer bidirectional LSTM, a linear layer over the 46 (45) phoneme symbols and
  the CTC blank. CTC means the exported graph has no decoder loop: `LetterToPhonemeFallback` runs
  it once per word on `models-backend-onnx` and collapses the argmax path. It sits exactly in
  misaki's fallback slot (single words and whole unresolved groups), the words it guessed are still
  reported, and nothing else in the phonemizer changed. Three numbers describe it and all three
  are derived by script. Held-out dictionary words, one word per run as deployed
  (`g2p/out/g2p-*.json`, 2 epochs, 16 minutes each on the loaded dev box): **US 46.0 % exact,
  10.9 % phoneme error rate; GB 50.2 %, 10.5 %** (the same checkpoints score 55.4 % and 61.4 % in
  padded batches, which is how they were trained; the gap is the backward LSTM direction seeing
  padding, a known weakness of this quick recipe, fixable with packed sequences and more epochs).
  Agreement with misaki's eSpeak fallback on the corpora's own 160 (US) and 159 (GB) unresolved
  words, almost all proper nouns (`g2p/out/espeak-agreement.json`): **24.4 % / 27.0 % identical
  strings, 25.4 % / 26.0 % phoneme error rate against eSpeak's spelling**, where eSpeak is itself a
  rule-based guess for such words. The Java executor reproduces the exporting PyTorch model on 200
  held-out words per accent within two near-tie flips (`LetterToPhonemeFallbackTest`), which needed
  one more operator, `Tile`, in `models-backend-onnx`. The out-of-vocabulary rate is unchanged by
  construction (4.13 % / 4.11 % on WikiText, 0.06 % on Harvard); those words are now pronounced
  rather than dropped, and still listed. **The candidate passes on our phonemes are rerun
  in full with the fallback; the comparator and publisher-phoneme passes stand.** The judge, the
  corpora and the floors are unchanged; this is a change to the candidate, made after one failing
  pass and recorded as such.

## State of the work (updated 2026-11-01, evening)

- **Kernels on the Vector API** landed after the parity closure; parity was not re-run after the
  last kernel change on the dev box and is scheduled to run again on whichever machine measures
  throughput (`paradee-chain.sh` stage 5 in the box log).
- **Phonemizer, measured** (`evidence/phonemizer/*.json`, written by
  `ParadeePhonemizerReportIntegrationTest`): on `harvard-200` the port leaves **1 of 1,765 words**
  unresolved in both accents (`rosebush`); on `wikitext103-heldout-200` it leaves **161 of 3,896
  (4.13 %)** in US spelling and 160 (4.11 %) in British, touching 90 and 91 of the 200 sentences,
  almost all proper nouns, which is the eSpeak fallback the port does not carry. Two of the
  unresolved tokens are punctuation, checked against misaki's own lexicon path (`fallback=None`)
  on the same sentences: for `Xenoceratops - (Alberta, Canada)` misaki tags `-` as punctuation
  with empty phonemes and the port counts it as unresolved, and the two phoneme strings are
  identical, so that is an accounting difference only; for `Play Zone!'s PC game …` misaki keeps
  the `!` inside the word (`zˈOn!s`) where the port drops it and spells the clitic as a letter
  (`zˈOn ˈɛs`), a real tokenizer discrepancy on one sentence in 400. **Exact match against the
  publisher's phonemes**, from `derive.py` over the first `af_heart` passes on
  `wikitext103-heldout-200`: **44.5 % of the 200 sentences** were phoneme-for-phoneme identical
  on the first candidate pass; the 90 sentences with an unresolved word cannot match, and among
  the other 110 the rate was **80.9 %**. Read side by side, the mismatches were a stressed
  `that` where misaki's lexicon has the unstressed complementizer as its default (the port's
  tagger called every `that` a determiner), the initialism `US` tagged as the pronoun instead of
  a proper noun (misaki spells it, `jˌuˈɛs`), the verb `progresses` read as a noun, and the
  direction of a quotation mark. **Two of these are tagger defects and were corrected before any
  WER had been read** (`that` is now a determiner only when a noun phrase follows it; an all-caps
  word in mixed-case text is a proper noun, as spaCy tags it); with the correction the rate is
  **49.5 % overall and 90.0 % among resolved sentences** (`bf_isabella`: 49.0 % and 89.9 %). The
  one candidate pass on our phonemes that had run with the old tagger was discarded and rerun, so
  every candidate pass uses the corrected port. The verb/noun case and the quotation direction
  stand as known discrepancies.
- **Where the arms run, and why that deviates from "this machine, nothing else building".** The dev
  box is shared by several concurrent agent sessions running Gradle builds; its load average was
  16 at 21:16Z and between 50 and 450 (on 12 logical CPUs) through the first passes. A dedicated
  Hetzner box was provisioned for both arms and torn down eight minutes later because the session
  was not permitted to copy the worktree to it (`evidence/bench-box-2026-11-01-vps-log.md`). So:
  the **intelligibility gate**, which does not depend on load, is measured here under
  `evidence/devbox/` by `comparator/run_comparator_passes.sh` and `comparator/run_candidate_passes.sh`
  running side by side (the first passes came from the interleaved `run_devbox_arms.sh`), each
  writing the load averages before every pass (`load.txt`); the **throughput gate** is not measured
  by those passes and stays open until both arms run on a quiet machine. The per-pass real-time multiples under `evidence/devbox/` are reported with their
  load values and are not the gate number.
- First passes present (`evidence/devbox/*/wikitext103-heldout-200/af_heart/t1/pass1/pass.json`):
  comparator 1,362.8 s of audio at 8.74x realtime (load 14–21), candidate on our phonemes 1,284.8 s
  at 1.69x (load 50 rising to 400), candidate on the publisher's phonemes **1,362.8 s, the
  comparator's sample count exactly**, at 2.07x. That pass has no `junit.xml` because a concurrent
  Gradle run in the same module clobbered the test-results directory after the pass had written
  its `pass.json`; the pass is complete and the runners are now sequenced.
- Judge: running over completed passes as they appear (`comparator/run_judge_loop.sh`); the
  Soprano judge-sanity pass (`SopranoJudgeSanityIntegrationTest`, the shipped Q8_0 artifact,
  `evidence/judge-sanity-soprano/`) is scored by the same loop. **A second round-trip judge
  landed on `main` the same evening** (PR #328,
  `benchmark-results/2026-11-01-tts-roundtrip-judge-validation/`): faster-whisper's CTranslate2
  `base` at int8 with beam 5, `whisper-normalizer`'s English normalizer, a 63-tap sinc resampler,
  ten sentences through the released `ModelJars.openSpeech` route, corpus WER 0.0526. The two
  do not compete: that one validated that a round-trip pipeline works at all; this protocol pins
  **the publisher's own judge** (`openai-whisper` 20250625 `base`, `fp16=False`, greedy,
  `torchaudio` resampling, the regex normalizer of `onnx_eval.py`) because its purpose is to
  re-derive the card's figure, and the comparator's first pass did (5.68 % against the card's
  5.8 %). Whisper's `EnglishTextNormalizer` WER is reported beside every number for comparison
  with #328's convention. The first comparator result also answers #328's open question about
  judge throughput: the publisher's protocol costs about ten minutes per 200 sentences on this
  loaded machine.

## State of the work (updated 2026-11-01, end of the first session)

Built and passing their unit suites on branch `feat/paradee-8m-tts`:

- `models-backend-onnx`: a protobuf wire reader, tensor type with float/double/int64/byte stores,
  an eager executor with constant folding and an injectable noise source, and the 56 operators the
  Paradee graphs use. The structural integration test matches the triage exactly (2,117 / 53 and
  2,535 / 56 nodes and operator types; 8,581,449 and 7,867,227 initializer elements). Both real
  graphs run end to end to finite waveforms.
- `models-audio/phonemizer`: the Java port of misaki's English lexicon path (gold and silver
  dictionaries, stems, numbers, stress, compound handling), a rule-based tokenizer and tagger in
  place of spaCy, no eSpeak fallback; unknown words are reported.
- `models-audio/ParadeeTextToSpeechModel`: the publisher's pipeline over the two, all four voices,
  sentence-level streaming, seedable noise, and `synthesizePhonemes` for feeding the publisher's
  own phonemes.

**Unoptimized executor speed on the dev box, one sentence, not a measurement of the gate:** about
0.6x realtime (5.45 s of audio in 8.4 to 9.3 s per voice, first run after one warm-up). The
publisher states about 62x for onnxruntime on an M4 Pro. Per-operator time on the smoke sentence
puts 4.2 s of 5.4 s in `Conv`, then `ConvTranspose`, `MatMul`, broadcast `Mul`/`Less`, `LSTM`,
`Sin`. No kernel has been optimized yet; the gate ratio will be measured when both arms exist.

**Seeded parity: closed, 2026-11-01.** `evidence/parity/seeded-parity-2026-11-01.json`, derived by
`comparator/derive_parity.py` from the JUnit XML beside it: 40 of 40 cases pass (fp32 and int8
graphs, four voices, five sentences each) with PCM cosine 1.000000 on every case, worst
signal-to-difference 63.37 dB against the 20 dB floor, worst absolute sample difference 0.00155,
and identical sample counts throughout. The executor reproduces onnxruntime 1.23.2 to single
precision when fed the noise it drew.

**Comparator environment: installed.** After an Intel-Mac re-pin (onnxruntime 1.23.2, torch 2.2.2,
numba 0.62.1) and the eSpeak data-path workaround in `pins.prepare_espeak()`, the publisher's
package phonemizes and synthesizes on this host. Steps 1 and 2 of the order of work are running or
next; no WER or throughput number exists yet.

## Hypothesis

A pure-Java executor for the publisher's int8 ONNX graph, fed by a Java port of misaki's English
lexicon path, synthesizes speech that Whisper base transcribes as accurately as the publisher's own
onnxruntime pipeline does on the publisher's own held-out sentences, at no less than 0.80 of its
real-time multiple on the same CPU thread.

## The artifact

`sahilmahendrakar/Paradee-8M-v1.2` at `ed2934d4df8a4fd891e29584c5e1104649409765`:

| file | sha256 |
| --- | --- |
| `onnx/paradee_int8.onnx` (the catalog artifact) | `a6989796692ac173597e15d405c971989202d5c73cfad65f22e90bd7b6c97da1` |
| `onnx/paradee.onnx` (fp32, parity reference for the executor) | `8bac7b0bebef403cb22b54ead6bee423fa34a04fd91f73326b7f65cd9d0d25b9` |
| `config.json` | `002535cf8dd08a434cdddb13372da8a7b4844152bdf9b5b4e7342e12ec30bcfb` |

Four voices, in the config's order: `af_heart` (0), `bf_isabella` (1), `am_puck` (2), `bm_fable` (3).

## Corpora (pinned in `corpus/`, with source digests)

| set | sentences | words | why |
| --- | ---: | ---: | --- |
| `wikitext103-heldout-200` | 200 | 3,463 | **the publisher's own evaluation set.** `training/README.md` @ `aae3cc46`: "`data/wikitext_sents.json` holds the 12,000 WikiText-103 sentences that were used. The first 200 are held out for evaluation." This is what the card's WER was scored on |
| `harvard-200` | 200 | 1,561 | second dataset, phonetically balanced, public domain; the two-dataset rule |
| `card-samples-5` | 5 | — | the five sentences the card publishes audio for; the seeded-parity set |

## Arms

**Comparator (the publisher's runtime):** the `paradee` package at `aae3cc46`, onnxruntime
1.23.2, misaki 0.9.4 with the eSpeak fallback, exactly as `paradee/tts.py` wires them; pinned in
`comparator/uv.lock`. onnxruntime is 1.23.2 rather than the current 1.31.1 because the dev box is
an Intel Mac and 1.23.2 is the newest release that ships a wheel for it; the publisher's package
requires only `>=1.17` and pins nothing itself. The judge runs on torch 2.2.2 for the same reason.
Both deviations are stated here before any arm has run. A third, mechanical one: the loader
wheel's libespeak-ng on this host ignores explicit data paths and misreads `ESPEAK_DATA_PATH`, so
`pins.prepare_espeak()` points it at a directory of symlinks to the wheel's own data files; the
phoneme data is byte-identical to what the publisher's pipeline reads elsewhere. Sentence split, phonemization, pad tokens, truncation and `voice` index are
the package's own code, called through its own methods, never re-implemented here.

**Candidate (ours):** `models-backend-onnx` executing the same int8 graph, `models-audio`'s
Paradee model feeding it from a Java misaki port. Run in two configurations: on the publisher's
phonemes (isolates the executor) and on our phonemes (what a user gets). Both are published.

Both arms run on this machine (`evidence/machine.txt`), CPU only, with the same thread count, on
byte-identical sentence lists, in sessions where nothing else is building. *Deviation, recorded
before any verdict:* the WER passes ran on this machine under other sessions' build load, with
the load averages written beside every pass, because WER does not depend on load; the throughput
comparison is deferred to a quiet machine (see "State of the work").

## Gates, and where each number comes from

Replication: **three passes per arm and configuration; the minimum across the candidate's passes
must clear each floor.** Measured on the shipped TTS gate on 2026-11-01, two passes of one artifact
differed by 2.68x in p95 real-time factor on this machine, so one pass is not a verdict.

| gate | rule | status of the number |
| --- | --- | --- |
| intelligibility | corpus-level WER per voice and per corpus, judge = Whisper `base` via `openai-whisper` 20250625, `fp16=False`, `language="en"`, audio resampled 24 kHz to 16 kHz with `torchaudio.functional.resample`, normalizer `" ".join(re.sub(r"[^a-z0-9 ]+", " ", s.lower()).split())`, `jiwer.wer` over all sentences, the publisher's protocol verbatim from `training/scripts/onnx_eval.py`. **Floor: on every candidate pass, WER ≤ the comparator's worst pass + 0.5 points absolute**, per voice and corpus. Whisper's own `EnglishTextNormalizer` WER is reported beside it, never gated | the 0.5-point margin is a **policy choice made here** (about 17 words of the 3,463); the comparator's number is re-derived, not read from the card |
| throughput | real-time multiple = Σ audio seconds / Σ `session.run` seconds over the corpus after one untimed warm-up sentence, the publisher's convention; **ratio candidate / comparator ≥ 0.80 at 1 thread**, reported also at 6 threads (the physical core count). The shipped gate's p95 per-sentence real-time factor and time-to-first-audio are recorded for the entry | 0.80 inherited; the 1-thread figure is the one the card states |
| port correctness (beside the verdict) | **seeded parity** on `card-samples-5` × 4 voices: the oracle run exposes the `RandomNormalLike` output as an extra graph output, the executor consumes that tensor instead of drawing its own, then PCM cosine ≥ 0.995 and signal-to-difference ≥ 20 dB per the shipped `speech-oracle-streaming-latency-v1`, on the fp32 graph and on the int8 graph | thresholds inherited; proves the executor, not the model |
| phonemizer | per-sentence exact-match rate of our phoneme string against the comparator's, per accent, on both corpora; out-of-vocabulary word rate reported | reported, not gated; the user-facing WER on our phonemes is the gate |
| completion | every sentence yields finite PCM of non-zero length; any non-finite sample fails the pass | rule |

## Kill criteria

- If the comparator's own three passes spread by more than 1.0 WER point on either corpus, the
  judge cannot support a 0.5-point floor; stop, publish the spread, redesign the floor before any
  candidate run.
- If seeded parity cannot reach 0.995 on the fp32 graph after per-operator comparison against
  onnxruntime's intermediates, the executor is wrong; stop and publish which operator diverged.
- If the comparator's re-derived WER on `wikitext103-heldout-200` with `af_heart` is more than 2.0
  points from the card's 5.8%, the protocol does not reproduce the publisher's and the anchor is
  void; publish and stop until the difference is explained.

## Order of work

1. `comparator/`: `uv sync`, `oracle_dump.py` (parity tensors), `comparator_arm.py` three passes
   per voice and corpus at 1 and 6 threads, `judge.py`, `derive.py` → `evidence/comparator-report.json`.
2. Judge sanity on a known-good model: the same `judge.py` over Soprano output produced by the
   shipped runtime, so the judge pipeline is seen scoring a model whose waveform parity is proven.
3. `models-backend-onnx`: ONNX reader, eager executor, per-operator parity, seeded whole-graph parity.
4. Java misaki port, exact-match rate against the comparator's phonemes.
5. Candidate arm through `TextToSpeechModel`, three passes, judge, verdict per voice.

## What is not claimed

UTMOS and any naturalness judgement are not reproduced. Round-trip WER is intelligibility. The
card's figures (5.8% / 4.40 / 62x) are the publisher's until step 1 re-derives the first and third
on this machine.
