# Independent vision arithmetic fixtures

`pr380-arithmetic-fixtures.py OUTPUT_DIRECTORY` generates deterministic affine-projection and
ERF GELU expectations with PyTorch 2.9.0+cpu, MKL 2024.2 and oneDNN 3.7.1. It does not call Models.
The complete generation environment is retained with the remote experiment. The Java native
regressions regenerate the LCG inputs, exercise owned immutable weights/bias, and check the output
sentinel and rejected shapes. GELU's 8,193 elements exercise both vector blocks and a ragged tail.

`pr380-rope-fixture.cpp OUTPUT_DIRECTORY` links only the independent pinned ggml test oracle,
2cbca389f515df455412dafa07e2b1359cc72a6d. Compile with its `ggml/include` headers and
`libggml-cpu`/`libggml-base`; it produces normal and NeoX layouts at five positions and two heads.
No external oracle is linked into the Models runtime.

Fixture locations and their hash manifests:

- `backend-native/src/test/resources/vision-arithmetic/`: independent affine and GELU outputs.
- `backend-java/src/test/resources/vision-rotary/`: normal and NeoX RoPE outputs.
- `backend-java/src/test/resources/vision-layernorm/`: independent PyTorch Welford normalization.
- `backend-native/src/test/resources/vision-attention/`: independent retained-prefix chunk outputs.
- `backend-native/src/main/rust/model-kernels/testdata/vision-attention/`: online, tiled, split-KV,
  and pruned-query attention outputs, including original logical batch sizes.

These fixtures catch arithmetic and lifecycle regressions. They do not replace the full frozen
80-case task corpus or the same-host two-arm performance gates.
