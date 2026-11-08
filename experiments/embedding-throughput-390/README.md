# Embedding throughput diagnosis (#390)

This is a dedicated Linux-host experiment, not a catalog qualification shortcut.
See [the control record](CONTROL-2026-11-08.md) for hardware, costs, deadlines,
correctness requirements, timing protocol and the correction distinguishing
Models' existing oracle threshold from memory's proposed stricter one.

The harness resolves published Models **0.5.16** and Vectors **0.1.33**, with a
locked dependency graph. It injects the same bundled native matrix kernel as
ModelJars commit `10663435221396445e6c423020162cbd8f3971a7`. The released
ModelJars **0.1.76** predates that loader and pins older dependencies; this harness
does not prove an already-released ModelJars consumer has the fix.

Build the harness without downloading or running any model locally:

```sh
./gradlew -p experiments/embedding-throughput-390/harness installDist
./gradlew :backend-java:jar
```

On the dedicated host, use `/opt/embedding390` with:

- `baseline/lib`: the harness installation's unmodified published dependency JARs.
- `candidate/lib`: the same JARs, replacing **only** `backend-java-0.5.16.jar`
  with the candidate build (routing-only commit `e9c1760e` for the first experiment).
- `references`: `probes.txt`, `reference-embeddinggemma2-q8_0.json`, and
  `reference-embeddinggemma-300m-q8_0.json` from
  `models-bench/src/main/resources/embedding-equivalence`.
- This directory's `setup.sh`, `run.py` and `models.json`.

`bash setup.sh` records the host, installs and records the JDK and profiler,
verifies immutable model downloads, and runs the experiment serially. The
operator must archive output even when the script fails, verify copied evidence,
and delete the exact temporary server and firewall. See the workspace
`VPS-BENCHMARK-OPERATIONS.md`; provider credentials never go onto the worker.

`EmbeddingProfile` refuses inference outside Linux. Each report includes model
and classpath hashes, JVM arguments, tokenizer IDs, raw warmup/measured samples,
vector bits and execution-plan environment. Hash verification reads the model
before load: reported load time is **not cold-storage loading**. The timing text
is synthetic repeated text chosen by actual tokenizer length; retrieval quality
requires the independent memory datasets.

Trace and CPU-profile runs are separate from latency comparisons. Trace observes
successful delegated matrix calls. Profiles start after warmup and retain Java
and native frames in JFR. Frozen oracle checks verify reference/model/probe hashes
and report both the existing **0.999** catalog floor and memory's proposed
**0.99999** floor. Exact candidate/control vector hashes are an additional
regression gate. No threshold or catalog entry is changed by this experiment.

For the Qwen follow-up, wait until the first worker has exited successfully, then
prepare `qwen-candidate/lib` from the same published graph with candidate
`16a4410f`, add `reference-qwen3-embedding-0.6b-q8_0.json`, and run
`python3 run-qwen.py`. It uses LAST_TOKEN pooling and the same timed protocol.

The per-file commands and raw reports are the evidence; a completed worker alone
does not establish a performance gain or memory admission. A final report must
also compare candidate gains with the bracketing baseline drift and disclose
remaining accuracy, latency, release and retrieval-quality gaps.
