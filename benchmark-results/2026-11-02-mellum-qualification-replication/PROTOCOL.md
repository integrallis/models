# mellum qualification: replication protocol

**Frozen before any data was collected.** Written because the qualification entry landed in
ModelJars/modeljars#258 rests on a **single** paired run, and
`benchmark-results/2026-11-01-mellum-replicated-verdict` had already established, for this exact
model, that one pass is not a pass: a run reading `QUALIFIED` at 0.8112 was overturned by three
replications averaging 0.7630, because the comparator's spread on a shared box reaches +/-8%. That
record states the rule plainly -- "a pass obtained from the favourable end of a comparator's noise
is not a pass" -- and I landed the entry without satisfying it. This replicates it.

## Question

Does mellum's `decodeThroughputRatio`, with the Q4_K per-block unpack fix (models `dcd8b172`),
clear the 0.80 floor **on a replicated mean**, rather than on one draw?

## Design

Three complete paired runs on one box, strictly sequential, one session per pair, candidate then
comparator each time. Identical to the protocol the 2026-11-01 replication used: workload `coding`,
template `chatml-no-think`, top-k 1, `--warmups 1 --iterations 3`, `--threads 16`, context 2048,
max-tokens 256, `COMPARATOR=ollama`, `--artifact` passed so `artifactSha256` is non-null and the
comparison is admissible. Verdicts come from `RagQualificationCli` under
`RagProductionQualificationPolicy`, never from dividing reports by hand.

The native library is built from the merged source on the box itself; its sha256 is recorded and
compared against `0c6127f17821773f...bdd474`, the library both the qualification and the correctness
smoke were measured with.

## Decision rule, fixed in advance

- **mean `decodeThroughputRatio` over the three paired runs >= 0.80** -> the entry in
  `catalog/qualifications.json` is earned, and this record is committed beside it as its
  replication.
- **mean < 0.80** -> the entry is **not** earned and must be withdrawn from the catalog, whatever
  any individual run says. A revert PR follows, and this record is published as the null.
- Per-run ratios and the spread of both arms are reported whatever the outcome. If individual runs
  straddle the floor, the mean decides and the straddle is reported as the headline, because that is
  the condition under which the earlier false pass occurred.

No run is discarded. If a run fails for an environmental reason the failure is recorded and the run
is repeated, with both kept.

## What this cannot settle

The 2026-11-01 replications ran on an AMD EPYC 7713 without AVX-512 and averaged our decode at
27.92 against a 36.61 comparator. Whatever box this lands on, its absolute numbers are not
comparable to those; only the ratio and its spread are, and the comparator here has its own
AVX-512 tier where our kernels have none. The host is recorded with the result.
