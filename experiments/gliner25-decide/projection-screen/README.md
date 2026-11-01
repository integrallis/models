# Synthetic dense projection screen

This is an isolated component experiment, not a production backend. It uses seeded random F32
matrices at the encoder's three projection shapes and sequence lengths 32, 128 and 512. It
loads no model weights. Results are under `benchmark-results/2026-11-01-gliner25-decide/projection-screen/`.

The current row-by-row Java projection is compared with the existing Vectors 0.1.32 batched
matrix operation and, separately, Rust `matrixmultiply=0.3.11` behind a minimal FFM call.
Both arms use eight workers. Native inputs, weights and outputs are allocated before timing;
this excludes per-inference copies that an actual integration might need. The standalone C
ABI deliberately assumes valid benchmark-owned pointers and dimensions; it is not a shippable
shim and must never be exposed to untrusted callers.

Measured on macOS x86_64, Java 25.0.3: the existing Vectors batch operation gained 1.04–2.53x.
An initial Rust implementation partitioned tokens, repeated weight packing across workers,
and regressed the short shapes to 0.41–0.56x. Its failing results are retained; its dispatch
is not retained. Partitioning output rows improved all nine screened shapes by 1.70–6.18x,
with maximum absolute difference below 0.00001. These are eight paired samples after three
warmups, not the frozen final qualification procedure or a Linux whole-model speed claim.

Build the crate with `cargo build --release --locked`. Run each Java source with Vectors 0.1.32
on the classpath, `--add-modules jdk.incubator.vector`, `-Dvectors.gguf.threads=8`, and
`--enable-native-access=ALL-UNNAMED`. Pass the built native library's absolute path as the single
argument to `GlinerSgemmScreen.java`. The screen is intended for a remote worker for subsequent
reproductions. Model inference remains remote-only.

Next: account for native buffer ownership and copying, exercise the target Linux CPU, and
measure encoder-wide parity, accuracy and latency before adopting a production kernel.
