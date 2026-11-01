# Qualifying Paradee-8M v1.2: protocol, frozen before data

**Status 2026-11-01: protocol frozen. No arm has run. Nothing below is a result.** Every threshold
and every corpus is fixed in this file before the first synthesis so that none can be chosen after
seeing numbers. The candidate record that decided to run this is
`modeljars/docs/CANDIDATE-PARADEE-8M-20261101.md`; the artifact triage is
`modeljars/docs/model-evidence/paradee-8m/TRIAGE-v1.2.md`.

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

**Blocked on the comparator environment:** `uv sync --frozen` in `comparator/` pulls the
publisher's package from GitHub and was refused by the session's tool policy as external code; the
owner has to run it. Everything in steps 1 and 2 of the order of work, and the seeded parity
tensors, wait on it. Nothing numeric from those steps exists yet.

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
1.31.1, misaki 0.9.4 with the eSpeak fallback, exactly as `paradee/tts.py` wires them; pinned in
`comparator/uv.lock`. Sentence split, phonemization, pad tokens, truncation and `voice` index are
the package's own code, called through its own methods, never re-implemented here.

**Candidate (ours):** `models-backend-onnx` executing the same int8 graph, `models-audio`'s
Paradee model feeding it from a Java misaki port. Run in two configurations: on the publisher's
phonemes (isolates the executor) and on our phonemes (what a user gets). Both are published.

Both arms run on this machine (`evidence/machine.txt`), CPU only, with the same thread count, on
byte-identical sentence lists, in sessions where nothing else is building.

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
