# Q5 current implementation screen

Source baseline is the corrected native candidate 12796ebf. No routing is changed.
For Q5_1 compare current serial Java, a Java prototype using the existing reusable
GGUF stage executor with per-thread decoded-row reuse, and the current Rust route.
For Q5_0 compare two actual independent native calls with the actual public dual
native call that production eligibility currently excludes. Keep identical inputs,
outputs, quantized blocks and full float bit parity checks before timing.

Use real MiniLM projection shapes 384x384 and 1536x384, batches 1/8/64/256,
and matched worker settings 1/4/16. Record six counterbalanced 500ms cohorts
following 250ms warmups at each shape. This is a screening measurement, not a
whole-model qualification or a basis to guess a new routing threshold. Actual
embedding quality and model-level public-dataset timings are required for adoption.

Pinned JDK/Rust and immutable payload, dedicated x86/ARM hosts, 30-minute host
limit and $1.25 combined budget, immediate diagnostics and owned termination.
