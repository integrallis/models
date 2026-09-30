# Logit-fusion study — results log (append-only)

One row per executed phase: every gate, pilot arm, full arm, tuning run and summary. The protocol is
[PREREGISTRATION.md](PREREGISTRATION.md); how to reproduce any row is in [REPRODUCE.md](REPRODUCE.md).

## Rules

1. **Append only.** Never edit or delete a past row, including rows for runs that failed, were
   aborted by the watchdog, or later turned out to be wrong. A correction is a new row whose outcome
   names the row it supersedes.
2. **Every row carries a provenance label**, and the three are never mixed in one row:
   - `measured` — produced by our own harness on the named host at the named commit, with its
     evidence file present and listed in that host's `SHA256SUMS`;
   - `read` — taken from a paper, model card, leaderboard or other document (cite it in the outcome);
     a `read` row never enters a decision rule;
   - `believed` — an estimate or recollection awaiting measurement (for example the cost envelope in
     the protocol). A `believed` row is replaced by a later `measured` row, never edited into one.
3. **Evidence must exist.** `evidence` is the path inside the host evidence directory (or the
   committed copy under `host-evidence/`), and the file's sha256 must appear in that run's
   `SHA256SUMS`.
4. **Commit is the Models commit the host ran** (`environment.modelsCommit` in the report), not the
   commit that added the row.
5. **Pilot substitutions are stated.** Pilot rows for tuned arms ran with uniform weights, tie-break
   member B, and uncalibrated confidence; the `outcome` must say so (host-run records it as
   `pilotSubstitution`).
6. A gate failure is recorded as `fail` and stops the stage. It is never a reason to adjust a result.
7. A dataset that cannot exercise a feature (ORACLE headroom under 3 points) is recorded as
   `no headroom`, not `no effect`.

## Regenerating from host evidence

`host-run.sh` appends one JSON line per phase (and one per gate) to `$RUN_ROOT/evidence/results-log.jsonl`:

```json
{"date":"…","modelsCommit":"…","host":"…","phase":"pilot","arm":"F-tuned","dataset":"gsm8k",
 "evidencePath":"…/pilot-gsm8k-F-tuned.json","outcome":"done","provenance":"measured",
 "pilotSubstitution":"tuned weights -> uniform (S2 precedes S3 tuning)"}
```

Render it with the Java CLI (no Python needed):

```bash
models-bench/build/install/models-bench/bin/models-bench logit-fusion results-log \
  --in "$RUN_ROOT/evidence/results-log.jsonl" --report "$RUN_ROOT/evidence/RESULTS-LOG.generated.md"
# or, on the host:  MODELS_COMMIT=<sha> bash host-run.sh results-log
```

Copy the generated rows below the last existing row. Do not re-render over rows that are already
here.

## Log

| date | commit | host | phase | arm | dataset | evidence | outcome | provenance |
|------|--------|------|-------|-----|---------|----------|---------|------------|
| 2026-09-17T15:01:22Z | `f2bf5183df` | f2 | bootstrap | — | — | — | done | measured |
| 2026-09-17T15:01:23Z | `f2bf5183df` | f1 | bootstrap | — | — | — | done | measured |
| 2026-09-17T15:01:25Z | `f2bf5183df` | f2 | data | — | — | `f2/data-manifest.produced.json` | done | measured |
| 2026-09-17T15:01:26Z | `f2bf5183df` | f1 | data | — | — | `f1/data-manifest.produced.json` | done | measured |
| 2026-09-17T15:03:19Z | `f2bf5183df` | f3 | bootstrap | — | — | — | done | measured |
| 2026-09-17T15:03:20Z | `f2bf5183df` | f3 | data | — | — | `f3/data-manifest.produced.json` | done | measured |
| 2026-09-17T15:03:24Z | `f2bf5183df` | f1 | models | — | q8 | `f1/models-q8.jsonl` | done | measured |
| 2026-09-17T15:04:15Z | `f2bf5183df` | f2 | models | — | q8 | `f2/models-q8.jsonl` | done | measured |
| 2026-09-17T15:04:15Z | `f2bf5183df` | f2 | pilot | — | gsm8k | — | failed rc=1 — arm selector aborted before the arm ran (see note below) | measured |
| 2026-09-17T15:04:15Z | `f2bf5183df` | f2 | pilot | — | gsm8k | — | failed rc=1 — arm selector aborted before the arm ran (see note below) | measured |
| 2026-09-17T15:04:15Z | `f2bf5183df` | f2 | pilot | — | gsm8k | — | failed rc=1 — arm selector aborted before the arm ran (see note below) | measured |
| 2026-09-17T15:04:15Z | `f2bf5183df` | f2 | pilot | — | gsm8k | — | failed rc=7 — arm selector aborted before the arm ran (see note below) | measured |
| 2026-09-17T15:04:15Z | `f2bf5183df` | f2 | pilot | — | gsm8k | — | failed rc=7 — arm selector aborted before the arm ran (see note below) | measured |
| 2026-09-17T15:04:16Z | `f2bf5183df` | f2 | pilot | — | gsm8k | — | failed rc=7 — arm selector aborted before the arm ran (see note below) | measured |
| 2026-09-17T15:04:16Z | `f2bf5183df` | f2 | pilot | — | gsm8k | — | failed rc=7 — arm selector aborted before the arm ran (see note below) | measured |
| 2026-09-17T15:04:16Z | `f2bf5183df` | f2 | pilot | — | gsm8k | — | failed rc=1 — arm selector aborted before the arm ran (see note below) | measured |
| 2026-09-17T15:04:16Z | `f2bf5183df` | f2 | pilot | — | gsm8k | — | failed rc=1 — arm selector aborted before the arm ran (see note below) | measured |
| 2026-09-17T15:07:21Z | `f2bf5183df` | f1 | models | — | big-q8 | `f1/models-big-q8.jsonl` | done | measured |
| 2026-09-17T15:07:31Z | `f2bf5183df` | f1 | gate-g0 | — | — | `f1/gate-g0.json` | pass | measured |
| 2026-09-17T15:09:56Z | `f2bf5183df` | f3 | models | — | q8 | `f3/models-q8.jsonl` | done | measured |
| 2026-09-17T15:09:56Z | `f2bf5183df` | f3 | pilot | — | gsm8k | — | failed rc=1 — arm selector aborted before the arm ran (see note below) | measured |
| 2026-09-17T15:09:56Z | `f2bf5183df` | f3 | pilot | — | gsm8k | — | failed rc=1 — arm selector aborted before the arm ran (see note below) | measured |
| 2026-09-17T16:20:49Z | `f2bf5183df` | f3 | pilot | F-mix | gsm8k | `f3/pilot-gsm8k-F-mix.json` | done — accuracy 0.86 (43/50) | measured |
| 2026-09-17T16:55:26Z | `f2bf5183df` | f3 | pilot | AB | gsm8k | `f3/pilot-gsm8k-AB.json` | done — accuracy 0.76 (38/50) | measured |
| 2026-09-17T17:19:51Z | `f2bf5183df` | f1 | gate-g1 | — | — | `f1/gate-g1.json` | pass | measured |
| 2026-09-17T17:27:04Z | `f2bf5183df` | f1 | gate-g2 | — | gsm8k | `f1/gate-g2.json` | pass | measured |
| 2026-09-17T17:27:05Z | `f2bf5183df` | f1 | gate-g4-gsm8k | — | — | `f1/gate-g4-gsm8k.json` | pass | measured |
| 2026-09-17T17:27:06Z | `f2bf5183df` | f1 | gate-g4-gsm8k-dev | — | — | `f1/gate-g4-gsm8k-dev.json` | pass | measured |
| 2026-09-17T17:27:07Z | `f2bf5183df` | f1 | gate-g4-arc | — | — | `f1/gate-g4-arc.json` | pass | measured |
| 2026-09-17T17:27:08Z | `f2bf5183df` | f1 | gate-g4-math500 | — | — | `f1/gate-g4-math500.json` | pass | measured |
| 2026-09-17T17:27:09Z | `f2bf5183df` | f1 | gate-g6 | — | — | `f1/gate-g6.json` | pass | measured |
| 2026-09-17T17:27:09Z | `f2bf5183df` | f1 | gates | — | — | — | done | measured |
| 2026-09-17T17:27:09Z | `f2bf5183df` | f1 | pilot | — | gsm8k | — | failed rc=1 — arm selector aborted before the arm ran (see note below) | measured |
| 2026-09-17T17:27:09Z | `f2bf5183df` | f1 | pilot | — | gsm8k | — | failed rc=1 — arm selector aborted before the arm ran (see note below) | measured |
| 2026-09-17T17:27:09Z | `f2bf5183df` | f1 | pilot | — | gsm8k | — | failed rc=1 — arm selector aborted before the arm ran (see note below) | measured |
| 2026-09-17T17:27:09Z | `f2bf5183df` | f1 | pilot | — | gsm8k | — | failed rc=7 — arm selector aborted before the arm ran (see note below) | measured |
| 2026-09-17T17:27:09Z | `f2bf5183df` | f1 | pilot | — | gsm8k | — | failed rc=7 — arm selector aborted before the arm ran (see note below) | measured |
| 2026-09-17T17:27:10Z | `f2bf5183df` | f1 | pilot | — | gsm8k | — | failed rc=1 — arm selector aborted before the arm ran (see note below) | measured |
| 2026-09-17T17:27:10Z | `f2bf5183df` | f1 | pilot | — | gsm8k | — | failed rc=1 — arm selector aborted before the arm ran (see note below) | measured |
| 2026-09-17T17:27:10Z | `f2bf5183df` | f1 | pilot | — | gsm8k | — | failed rc=1 — arm selector aborted before the arm ran (see note below) | measured |
| 2026-09-17T17:48:38Z | `f2bf5183df` | f3 | pilot | AC | gsm8k | `f3/pilot-gsm8k-AC.json` | done — accuracy 0.96 (48/50) | measured |
| 2026-09-17T18:03:51Z | `f2bf5183df` | f1 | pilot | — | gsm8k | — | failed rc=1 — arm selector aborted before the arm ran (see note below) | measured |
| 2026-09-17T18:04:00Z | `f2bf5183df` | f1 | pilot | — | gsm8k | — | failed rc=1 — arm selector aborted before the arm ran (see note below) | measured |
| 2026-09-17T18:04:38Z | `f2bf5183df` | f1 | pilot | — | gsm8k | — | failed rc=1 — arm selector aborted before the arm ran (see note below) | measured |
| 2026-09-17T18:05:01Z | `f2bf5183df` | f1 | pilot | — | gsm8k | — | failed rc=1 — arm selector aborted before the arm ran (see note below) | measured |
| 2026-09-17T18:05:40Z | `f2bf5183df` | f1 | pilot | — | gsm8k | — | failed rc=1 — arm selector aborted before the arm ran (see note below) | measured |
| 2026-09-17T18:07:18Z | `b6350805a3` | f2 | bootstrap | — | — | — | done | measured |
| 2026-09-17T18:07:20Z | `b6350805a3` | f1 | bootstrap | — | — | — | done | measured |
| 2026-09-17T18:27:28Z | `b6350805a3` | f1 | pilot | A | gsm8k | `f1/pilot-gsm8k-A.json` | done — accuracy 0.66 (33/50) | measured |
| 2026-09-17T18:43:43Z | `f2bf5183df` | f3 | pilot | BC | gsm8k | `f3/pilot-gsm8k-BC.json` | failed rc=2 — accuracy 0.86 (43/50) | measured |
| 2026-09-17T18:44:15Z | `b6350805a3` | f3 | bootstrap | — | — | — | done | measured |
| 2026-09-17T18:47:58Z | `b6350805a3` | f1 | pilot | B | gsm8k | `f1/pilot-gsm8k-B.json` | done — accuracy 0.78 (39/50) | measured |
| 2026-09-17T19:27:53Z | `b6350805a3` | f1 | pilot | C | gsm8k | `f1/pilot-gsm8k-C.json` | done — accuracy 0.98 (49/50) | measured |
| 2026-09-17T19:27:55Z | `b6350805a3` | f1 | pilot | VOTE | gsm8k | `f1/pilot-gsm8k-VOTE.json` | done — accuracy 0.82 (41/50) | measured |
| 2026-09-17T19:34:06Z | `b6350805a3` | f1 | pilot | RERANK | gsm8k | `f1/pilot-gsm8k-RERANK.json` | done — accuracy 0.88 (44/50) | measured |
| 2026-09-17T19:55:10Z | `b6350805a3` | f3 | pilot | F-uniform | gsm8k | `f3/pilot-gsm8k-F-uniform.json` | done — accuracy 0.84 (42/50) | measured |
| 2026-09-17T21:06:42Z | `b6350805a3` | f3 | pilot | F-article | gsm8k | `f3/pilot-gsm8k-F-article.json` | done — accuracy 0.90 (45/50) | measured |
| 2026-09-17T21:07:05Z | `b6350805a3` | f1 | pilot | SC-k | gsm8k | `f1/pilot-gsm8k-SC-k.json` | done — accuracy 0.98 (49/50) | measured |
| 2026-09-17T21:20:27Z | `b6350805a3` | f2 | pilot | A-think | gsm8k | `f2/pilot-gsm8k-A-think.json` | done — accuracy 0.76 (38/50) | measured |
| 2026-09-17T22:19:20Z | `b6350805a3` | f1 | pilot | BIG | gsm8k | `f1/pilot-gsm8k-BIG.json` | done — accuracy 0.96 (48/50) | measured |
| 2026-09-17T23:13:19Z | `b6350805a3` | f1 | pilot | BIG-Q8 | gsm8k | `f1/pilot-gsm8k-BIG-Q8.json` | done — accuracy 0.98 (49/50) | measured |
| 2026-09-18T00:14:24Z | `b6350805a3` | f2 | pilot | B-think | gsm8k | `f2/pilot-gsm8k-B-think.json` | done — accuracy 0.82 (41/50) | measured |
| 2026-09-18T04:48:13Z | `b6350805a3` | f2 | pilot | C-think | gsm8k | `f2/pilot-gsm8k-C-think.json` | done — accuracy 0.96 (48/50) | measured |
| 2026-09-18T04:48:14Z | `b6350805a3` | f2 | pilot | VOTE-think | gsm8k | `f2/pilot-gsm8k-VOTE-think.json` | done — accuracy 0.86 (43/50) | measured |
| 2026-09-18T04:48:16Z | `b6350805a3` | f2 | pilot | VOTE-consist | gsm8k | `f2/pilot-gsm8k-VOTE-consist.json` | done — accuracy 0.86 (43/50) | measured |
| 2026-09-18T04:48:17Z | `b6350805a3` | f2 | pilot | VOTE-conf | gsm8k | `f2/pilot-gsm8k-VOTE-conf.json` | done — accuracy 0.92 (46/50) | measured |
| 2026-09-18T04:55:11Z | `b6350805a3` | f2 | pilot | RERANK-think | gsm8k | `f2/pilot-gsm8k-RERANK-think.json` | done — accuracy 0.90 (45/50) | measured |
| 2026-09-18T15:39:42Z | `b6350805a3` | f2 | pilot | SC-k-think | gsm8k | `f2/pilot-gsm8k-SC-k-think.json` | done — accuracy 1.00 (50/50) | measured |

## Rows appended 2026-09-26 from retained host evidence

These 68 rows were rendered from the three hosts' `results-log.jsonl` files, which the run wrote but
which were never copied into this table while the study was live. Provenance is `measured` exactly as
each host recorded it; nothing here was re-run, and no row was edited.

Rule 3 was applied rather than assumed: every referenced evidence file present locally was hashed and
checked against its host's `SHA256SUMS`. **37 files verified, 0 mismatches.** The remaining 31 rows
reference a directory or a path with no per-file checksum entry and are marked accordingly in the
evidence column.

### Why so many pilot rows read `failed rc=1` with no arm name

The 25 `rc=1` and 6 `rc=7` pilot rows clustered at 15:04 and 17:27 are one harness defect, not 31
failed experiments. `host-run.sh` read arm fields with `jq -e`, which exits non-zero when the value
it reads is `false`; under `set -e` every arm declaring `requiresFrozen=false` therefore aborted
before it ran. Only the tuned arms survived that first pass. The bug is fixed (`jq -r` plus an
explicit absence check) and `host-run-test.sh` covers it. The rows are kept because a harness defect
that silently drops most of a study is the most important thing in this log.

### Arms with no evidence file

| arm | why |
|---|---|
| `BIG-think` | host f2 was deleted mid-run by a stale watchdog whose deadline had not been cancelled when a second watchdog extended it; roughly 40 of 50 items were complete. `pilot-gsm8k-BIG-think.log` survives, the report does not. |
| `F-tuned`, `F-tuned-Q4` | require the S3 weight-tuning stage, which never ran. Pilot rows for tuned arms therefore ran with uniform weights (recorded as `pilotSubstitution`). |
| `BIG-Q4` | not reached before the study stopped. |

### What the 22 completed arms measure

| arm | kind | accuracy | core s/item |
|---|---|---|---|
| `SC-k-think` | sc | 1.00 | 10883 |
| `BIG-Q8` | member | 0.98 | 1032 |
| `C` | member | 0.98 | 764 |
| `SC-k` | sc | 0.98 | 1562 |
| `BIG` | member | 0.96 | 1371 |
| `C-think` | member | 0.96 | 5254 |
| `AC` | fuse | 0.96 | 1017 |
| `VOTE-conf` | vote-conf | 0.92 | 0 |
| `RERANK-think` | rerank | 0.90 | 129 |
| `F-article` | fuse | 0.90 | 1363 |
| `RERANK` | rerank | 0.88 | 116 |
| `VOTE-consist` | vote-consist | 0.86 | 0 |
| `VOTE-think` | vote | 0.86 | 0 |
| `BC` | fuse | 0.86 | 1053 |
| `F-mix` | fuse | 0.86 | 1329 |
| `F-uniform` | fuse | 0.84 | 1351 |
| `VOTE` | vote | 0.82 | 0 |
| `B-think` | member | 0.82 | 3337 |
| `B` | member | 0.78 | 392 |
| `A-think` | member | 0.76 | 3706 |
| `AB` | fuse | 0.76 | 661 |
| `A` | member | 0.66 | 386 |

**Every fusion arm scored at or below the best single member.** The best member, `C`, scored 0.98;
the best fusion arm, `AC`, scored 0.96, and the three-way arms scored 0.84 to 0.90. On this pilot the
pre-registered rationale for logit fusion does not hold.

Two secondary observations, both worth following up independently of fusion:

* **Confidence weighting helps voting.** `VOTE-conf` 0.92 against plain `VOTE` 0.82 on the same
  members. `VOTE-consist` scored 0.86. This is aggregation over completed generations, so its
  `core s/item` of 0 excludes the generation it consumes — quoting that as the arm's cost would be
  wrong.
* **Rerank is by far the cheapest non-trivial arm**, at 116–129 core s/item against 660–1363 for the
  fusion arms, for 0.88–0.90.

`SC-k-think` reached 1.00 at 10,883 core s/item — roughly 8× the cost of the best single member for
two points. It is recorded, not recommended.

**Scope.** One dataset (GSM8K), 50 items per arm, one 3-host generation, uniform-weight substitution
on tuned arms, and four arms missing. Under the working agreement one dataset cannot settle a general
claim: this is sufficient to stop spending on fusion, and not sufficient to publish "fusion does not
work" as a general result.
