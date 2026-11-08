# Pinned graph arithmetic, 2026-11-08

The completed c7 campaign remains a failed attempt: Q4_K_S missed one answer; Q4_K_M exceeded the first performance bucket. New synthetic evidence is diagnostic, not catalog admission.

The remaining numerical differences were caused by graph dispatch and operation order:

* Q4_K tensors divisible into eight output rows use the pinned reference CPU_REPACK buffer. Its first complete groups of four activation rows use 64-element GEMM folds; tails use a 256-element GEMV fold. Ordinary vector-dot fixtures do not exercise this contract.
* Q5_K ordinary vector dot retains eight floating lanes across the row and a separate minimum correction. The previous implementation folded each block before accumulation.
* For eight or more activation rows, the reference dispatches Q5_K and Q6_K to its tiled graph kernels. These round the weight-scaled integer block first, apply the Q5 minimum correction, then accumulate against the activation scale. Smaller batches retain ordinary vector-dot arithmetic.
* Interleaved rotary outputs use `fma(x0, cos, -(x1*sin))` and `fma(x0, sin, x1*cos)`. Native float power/sine/cosine factors are required to reproduce the Linux reference factors. Live first-layer query/key probes matched all 4096 values after both corrections.
* Pruning a final-layer projection to one row must preserve its original batch-dependent arithmetic. Unsafe query/output-projection pruning is now withheld; final feed-forward pruning remains available.

The production implementation adds capability-gated formats 11/12/13. Existing ordinary and Flint ordered Q4 contracts remain distinct. An existing duplicate capability bit was corrected: mixed Q8 activation grouping moves to 34, packed Q4 remains 31, and reference K-quant arithmetic uses 35. Unsupported native libraries do not advertise the new image contract.

Independent test generators and binary fixtures are retained beside this document and in native testdata. Q5/Q6 graph fixtures passed all 60 combinations of format, dimensions, batch boundary and worker count. Q4 fixtures cover worker partitions and batch tails; original and ordered paths retain their tests. Java injected-factor rotation is bit-exact against the independent graph fixture. Linux Rust tests include exact float-factor checks.

The external globally-replaced Q4 spike and ad hoc Java rotary classes are diagnostic only; neither is installed in the public runtime. Public-path regressions and a fresh complete frozen campaign are required before qualification or publication.
