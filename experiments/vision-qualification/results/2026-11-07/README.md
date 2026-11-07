# LLaVA 1.5 integration evidence — 2026-11-07

All three numerical probes passed. This is integration evidence for proceeding to qualification;
**it is not catalog admission, a task-accuracy score, or performance qualification**.

Candidate commit: `4a933e8ca50186520b862ab0636ca9184eae3c81`. The later evidence/CI documentation
commit does not change runtime code. Payload JAR hashes, both artifact hashes, image hashes, complete
individual timings, hardware, runtime versions, and raw-vector hashes are retained in this directory.
The raw archive is retained at `java-ai/audit-artifacts/pr380-vision/pr380-results.tar.gz` in the shared
workspace; its checksum is in `archive-sha256.txt`. No model weights are committed.

Worker: EC2 m6a.4xlarge, 16 vCPU/64 GiB, Ubuntu x86_64, AVX2; both arms configured for eight threads.
Java 25.0.5, Torch 2.9.0+cpu, Transformers 4.57.1, Pillow 12.0.0. Runs were sequential on the same
worker, using the identical pinned F16 projector, one warmup and three measured encodes per image.
Timings below cover CLIP plus its MLP projector and exclude loading, preprocessing and text decoding.
No cache-hit or FLOP-normalized estimate is used.

| Probe | Pixel max error | Embedding cosine | Embedding normalized RMSE | Java median ms | PyTorch median ms | Java/oracle latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cats | 0 | 0.999999993511 | 0.01140% | 6760.1 | 853.9 | 7.92× |
| wide-gradient | 0 | 0.999999997344 | 0.00729% | 6371.4 | 844.0 | 7.55× |
| tall-bands | 0 | 0.999999999735 | 0.00230% | 6499.6 | 897.9 | 7.24× |

The photograph is the pinned Hugging Face COCO sample, and the two generated probes exercise
landscape/portrait padding, resizing and RGB/CHW order. The oracle loads exactly the same converted
GGUF weights into Transformers' independent CLIPVisionModel and PyTorch's MLP; it does not execute
Models code. The conversion already selected 23 CLIP blocks. This establishes fidelity to that
artifact, not fidelity of the converter to the original publisher checkpoint.

## Full two-file smoke

`cats.json` also records text Q4_K_M + F16 mmproj inference through `LlavaBackend`:

- Two-file load: 1,255 ms (warm filesystem; not a cold-load benchmark).
- Image + text prefill: 174,097 ms, including the image encoder/projector.
- Decode loop: 13,668 ms; 32 generated tokens, stopped at the deliberately bounded token limit.
- Output begins: “The image features a couch with two cats lying on it. One cat is located on the left
  side of the couch, while the other cat is”. The photographed subjects are two cats on a couch.
- First full vocabulary logits and all projected image rows are retained in the raw archive.

This smoke verifies that the image positions reach the shipping decoder and that decoding continues.
It is neither a published VQA/OCR evaluation nor a paired decoder-accuracy benchmark. The 174-second
prefill and 7.2–7.9× encoder latency ratios are gaps to address before performance admission. Profile
exact projection GEMMs, per-query vision attention, and Q4_K decoder prefill separately; validate any
Java/Rust improvements against these same vectors before running the frozen task campaign.

## Engineering checks

- `models-api:check` and `backend-java:check`: pass, including formatting, SpotBugs, coverage and
  dependency policy. 1,335 test cases passed and 11 skipped; cached real-model fixtures were excluded
  on the development Mac. Real inference ran on this remote worker.
- API/backend Javadoc generation: pass.
- Four evidence-comparison tests: pass, including missing, truncated, nonfinite, mismatched-source
  and incomplete-timing rejection. These are now included in the regular CI workflow.
- Numerical tests cover the residual branch, legacy FFN shape mapping, decoder embedding-input
  parity including batched chunk/tail boundaries and subsequent decoding, repeated image calls,
  rejected profiles, close handling, and an independently computed Pillow pixel fixture.

The worker `i-05c3bd4526cd0c2d2` was terminated after evidence retrieval; its ephemeral SSH key was
removed. The control record is [CONTROL.md](../../CONTROL.md). The next steps are the published-task
accuracy campaign, matched end-to-end performance gates, and two-artifact catalog/demo packaging
specified in [the qualification guide](../../../../docs/VISION-QUALIFICATION.md).
