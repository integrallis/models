# Independent Q5_K row-accumulation fixtures

Generated on an AVX2/FMA/F16C Linux m6a.4xlarge using llama.cpp revision
`2cbca389f515df455412dafa07e2b1359cc72a6d`. The generator is
`experiments/vision-catalog-qualification/diagnostics/arithmetic-fixtures/pr380-q5-oracle.py`.
It calls the reference's `ggml_quantize_chunk(Q5_K)`, `quantize_row_q8_K`, and
`ggml_vec_dot_q5_K_q8_K` entry points; Models does not calculate expected values.
All input is synthetic deterministic LCG data, with nine weight rows and seventeen
activation rows. Columns are 768 and 4096; the generator also checks 256 and 11008.
Files use little-endian floats and native GGUF Q5_K block packing. See sha256.json.

The test exercises the shipping format selector and activation quantizer for
batch sizes 1, 3, 8 and 17. It requires exact float bits at every output, covering
cancellation and the single-row/batched boundary. The former block-wise reduction
fails these fixtures; the full-row eight-lane accumulation passes.

The external oracle is MIT-licensed llama.cpp. No reference library is linked
into the runtime or needed to run these tests. No model weights are included.
