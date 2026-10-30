# Flint scheduling diagnosis — complete

Runtime payload `9c453e9de85f77b398e9ad57eba13e305db96f72`; worker `15df96f1`.
Sequential same-host arms: default 16-worker pool, public pool8, pool8/poll1ms, pool16/poll1ms,
instrumented kernel timing at decode8, then unchanged Ollama16 control. All arms use the same
single prompt, 512-token limit, one warmup and three measurements. This is not qualification.

All 12 native pilot answers and the three instrumented answers are byte-identical; all pilot
answers pass the visible fact/citation screen. The instrumented boundary includes FFM, copies,
scheduling, reflection and timer overhead: about 72–74% native-call wall time, 26–28% other Java
work. Q8 projections account for more native time than the grouped Q4_K expert projections.

The public 8-worker pool is the selected supported setting for the following full campaign.
The 1ms polling option gives a small gain at 8 workers and a loss at 16; it is not a supported
ModelJars recommendation and is not selected. The next runtime also removes repeated F32 router
copies, with its own fresh parity, default, performance, and accuracy checks.

The 90-minute bounded instance and root volume were deleted; see cleanup.json. Source and
manifest hashes, reports, raw outputs and logs are retained. Metrics are in assessment.json.
