# QuickGELU arithmetic diagnostic

Pinned PyTorch 2.9.0+cpu, 4097 float32 inputs in [-12,12], eight threads. Matching the upstream reciprocal-then-multiply graph reduces RMS error from 6.42e-8 to 1.44e-9. Input and expected-output fixtures are committed under backend-native test resources. Updated Linux native tests pass (63 Rust, 8 Java) and Clippy passes.

The seven retained disagreements are diagnostic cases, not admission. `production-quickgelu-worker.log` uses double CLIP moments and F32 attention; `quickgelu-diagnostic.log` additionally uses experimental Welford normalization and F16-rounded attention. Neither experimental precision change is retained in the production branch. The production diagnostic still disagrees on Q4_K_M textvqa-38337 (Movies versus the current AUTO reference Electronics). Full corpus collection is required, and no catalog claim is made here.

`quick-build-receipt.json` binds the full-repeat JARs and Linux native library to production revision 3f31646c743647449310de2c5ff57fbd55d9e957. The fixture-test-only commit follows that immutable runtime source.
