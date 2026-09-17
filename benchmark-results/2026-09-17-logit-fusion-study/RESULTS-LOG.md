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
