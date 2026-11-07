# LLaVA vision integration and qualification

PR 380's initial CLIP encoder was disconnected from model loading and generation. The completed
path loads a matched text GGUF and mmproj, preprocesses an image, runs its trained CLIP encoder and
MLP projector, consumes the projected patches in the existing Llama KV sequence, and continues text
generation. The first concrete candidate is LLaVA 1.5 7B Q4_K_M with its F16 projector. The pinned
files and SHA-256 values are in `experiments/vision-qualification/artifacts.json`.

The [final paired remote run](../experiments/vision-qualification/results/2026-11-07-performance/README.md)
passes all three encoder probes, exact caption-token agreement, and the frozen latency gates against
both F32 and stock F16 reference caches. It closes the integration performance gaps exposed by the
[original run](../experiments/vision-qualification/results/2026-11-07/README.md). Published-task accuracy
and complete workload qualification remain before catalog admission.

## Public API

```java
try (ImageInferenceBackend backend = PureJavaBackend.load(textGguf, mmprojGguf)) {
    backend.reset();
    backend.prefill(prefixTokens, 0);
    int position = prefixTokens.length;
    backend.prefillImage(Files.readAllBytes(imagePath), position);
    position += backend.imageTokenCount();
    float[] logits = backend.prefill(suffixTokens, position);
    position += suffixTokens.length;
    // Sample logits with the usual sampler, then backend.forward(token, position++).
}
```

`LlavaBackend.load(textGguf, mmprojGguf)` exposes the same implementation directly. Callers own the
prompt template: use the model's LLaVA/Vicuna prefix, insert the image where its `<image>` marker
belongs, and encode the suffix without a second BOS. `VisionQualificationCli` contains a complete
greedy example. No fake image token is consumed. One image occupies 576 ordinary sequence positions
for the pinned model. Configure a context large enough for the patches, prompt and generated answer.
Dimension checks detect incompatible files, but cannot establish that two arbitrary same-width
checkpoints were trained together; use the pinned matched pair.

`ImageInferenceBackend` has the same single-threaded, reset/close lifecycle as `InferenceBackend`.
Returned image logits are stable caller-owned arrays. Invalid image/position/context inputs are
rejected before consuming decoder state. Close releases both mapped models. The Llama embedding
input uses the existing transformer graph and KV cache, without a duplicate decoder implementation. Image rows use the existing batched prefill engine,
including chunk boundaries and the final single-row tail.

## Supported profile

The loader deliberately accepts the converted, fixed-resolution **23-block CLIP + two-layer MLP**
LLaVA 1.5 profile. It reads signed or unsigned GGUF dimensions, the 4-D patch convolution,
`v.class_embd`, separate Q/K/V tensors, optional patch bias, and legacy reversed FFN names. The
converter already removed CLIP's final block: execute all 23 stored blocks, drop CLS, and do not
apply a post-layer norm. CLIP uses QuickGELU when declared; the MLP uses exact GELU. Projector output
dimension comes from `mm.2.weight`, not the misleading original CLIP `projection_dim` metadata.
Preprocessing is RGB, mean-colored square padding, antialiased bicubic resize, CLIP mean/std and CHW.

The optimized path reuses Models' owned Rust dense GEMM and quantized GGUF kernels. Dense CLIP
attention and causal prefill use the existing native pool; single-token decode uses the quantized
attention kernel. Q4 row packing preserves the original accumulation tree, and Q6 uses eight-query
tiles. The provider selects a 128-row image-prefill batch when the loaded library advertises the
packing capability. `vectors.gguf.threads` bounds both native and Java work; idle Java workers park
during native work. Diagnostics report the selected implementation and batch size.

Original checkpoint weights are preserved. The experimental load-time Q8 conversion and its unused
weight copies were removed after a real-model ablation failed both speed and numerical gates. The
portable Java implementation remains available when native kernels are absent.

**Qwen3-VL, Qwen3.5 vision, SigLIP, any-resolution tiling, video, OCR specialists, and cross-modal
retrieval are not supported by this loader.** In particular, Qwen3-VL still requires its vision
encoder, spatial merger, deepstack features, multimodal position IDs, and decoder architecture.
Its `clip` container architecture alone is not compatibility. Unsupported projector profiles fail
at loading rather than producing plausible-looking incorrect embeddings.

## Reproduce integration evidence on remote Linux

Build locally (no real-model inference or weight download on the development Mac):

```sh
./gradlew :models-bench:installDist :backend-native:jar
```

Copy that distribution, `backend-native/build/libs/backend-native-*.jar`,
`backend-native/src/main/rust/model-kernels/` (excluding build outputs), and
`experiments/vision-qualification/` to an ephemeral worker. Place the native Java bridge JAR in the
distribution's `lib/` directory and compile the owned Rust library on the worker with
`cargo build --release --locked`. Do not copy a macOS native library to Linux. Install Java
25+, CPU PyTorch 2.9.0, Transformers 4.57.1, gguf 0.17.1, NumPy 2.3.4 and Pillow 12.0.0. The committed
`run.sh` downloads and verifies the pinned artifacts, creates two deterministic non-square probes,
downloads the pinned photographic probe, runs both arms sequentially, and compares the results.
Pass the immutable candidate commit and distribution path explicitly. The runner also builds a
pinned external llama.cpp decoder comparator (requires Git, CMake and a C++ compiler), unless
`VISION_DECODER_REFERENCE_DIRECTORY` points to an existing clean build from
`build-decoder-reference.sh`. That comparator is never linked or packaged with Models. The worker must have at least
16 GiB available memory, sufficient context capacity, and about 8 GiB of disk for artifacts/tools.

```sh
VISION_NATIVE_LIBRARY=/path/to/model-kernels/target/release/libjmodels_kernels.so \
VISION_THREADS=8 VISION_WARMUPS=3 VISION_REPETITIONS=7 \
bash experiments/vision-qualification/run.sh /path/to/models-bench COMMIT_SHA /path/to/evidence
```

The oracle uses **Transformers' CLIPVisionModel**, loaded strictly from the identical converted
GGUF weights, and PyTorch's MLP operations. It independently preprocesses with Pillow. It does not
import Models code or require users to supply reference vectors. Comparing to the same converted
weights separates runtime error from checkpoint quantization. A separate publisher-checkpoint/task
campaign is still needed to assess the checkpoint itself.

The gate fails on missing, nonfinite, wrong-size or mismatched-host/artifact/image/thread results.
Per probe it records normalized pixels, all projected patch vectors, runtime versions, warmups and
individual timings. Numerical limits: pixel max absolute error <= 0.02, embedding cosine >= 0.999,
and embedding normalized RMSE <= 0.01. These are integration gates, not an invented catalog accuracy
policy. The photograph also exercises full two-file image-to-answer generation; its text output and
first logits are retained for inspection. The reference decoder consumes independently generated
PyTorch patch embeddings, the exact same prefix/suffix tokens, the same text GGUF and context, and
32 greedy output tokens. Both F32 KV (matched precision) and stock F16 KV are checked. Every generated
token must match in every repetition. Encoder, complete image/text prefill, decode, and total median
latency ratios must each be at most 1.10. The reference full prefill composes separately measured
PyTorch encoding/preprocessing and llama.cpp prefill; it is not a single-process multimodal timing.

Reports retain individual timings, source/artifact/payload hashes, settings, peak RSS and load times.
Load times use a warm filesystem and are not cold-load measurements. `INTEGRATION_PERFORMANCE_PASS`
is written only after all three probes and both decoder comparisons pass. The result still does not
establish accuracy on published VQA/OCR workloads.

## Proceeding to catalog qualification

1. Retain the three-probe numerical evidence and paired decoder performance/fidelity results from this PR.
2. Freeze representative published VQA and OCR datasets, prompts, scoring, output limits and an
   admission policy; run both runtimes on the same examples. Shape tests and a plausible caption do
   not establish task accuracy. Do not apply text-only RAG scores to a vision capability.
3. Extend the paired performance protocol to those full workloads and qualification hardware,
   including TTFT, image encoding, prefill, decode, memory and cold-load measurements. The current
   bounded-caption campaign does not substitute for complete task throughput.
4. Validate Java and any proposed Rust execution policies against the same vectors before admitting
   either. Converting weights to Q8 is a separate numerical change, not a free speed comparison.
5. Add the two-artifact catalog record and public image demo only after these gates pass. Existing
   text qualifications and catalog entries are unchanged.

The old synthetic ViT-S versus estimated ViT-L FLOP-normalized comparison has been removed. There
is no defensible llama.cpp parity or speed claim from it. llama.cpp versions that subtract another
block from an already-truncated projector also need investigation before serving as an oracle.
See [upstream issue 25817](https://github.com/ggml-org/llama.cpp/issues/25817).
