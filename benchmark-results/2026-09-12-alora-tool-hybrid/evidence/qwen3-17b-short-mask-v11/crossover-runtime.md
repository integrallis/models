# Runtime crossover policy

Physical prefix sharing avoids the second base-prefix evaluation, but freezing and forking the
cache also has a fixed cost. The production runtime therefore cannot assume that sharing is faster
for every prompt length.

The test-first runtime change on this branch makes the measured crossover an explicit constructor
input. Automatic turns recompute independent base and activated branches below that token count and
physically share the same immutable KV blocks at or above it. Qualification retains explicit
`SHARED` and `RECOMPUTED` modes so both arms can be compared against the same loaded weights.

For a stateful conversation below the crossover, the runtime does not discard or rebuild the
existing base lineage. It prefills only the missing suffix into that lineage and independently
evaluates the activated branch. If a later turn reaches the measured crossover, the retained base
lineage is frozen in place and both branches fork from it.

Synthetic tests prove the selection boundary, physical storage identity, independent control arm,
retained-session behavior, and promotion on a later longer turn. The production crossover value is
still unset: it will be the smallest fixed benchmark tier where the shared median beats independent
recomputation, provided the complete 4,096-token, memory, and token-exactness gates also pass.

ModelJars carries that value inside the component-qualification resource. Its evidence gate requires
the value to match the immutable, SHA-256-bound Models report before the activated runtime can open.
