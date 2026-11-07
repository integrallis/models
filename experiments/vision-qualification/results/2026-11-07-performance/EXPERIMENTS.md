# Rejected approaches and retained contracts

These exploratory runs selected the final candidate. They are not the final paired qualification distributions. The archive retains the experimental source, payloads, vectors, timings and profiling records. Temporary instrumentation and losing implementations are absent from production.

The old load-time Q8 conversion measured 2653.938 ms per encode on the photographic probe, with cosine 0.98668604 and normalized RMSE 0.162823. It failed the unchanged cosine/RMSE gates and was slower than the final unconverted path. Its conversion method, alternative encoder branch, unused weight copies, and synthetic-only benchmark were removed. `q8-payload-sha256.txt` identifies the exact experimental payload; `lib-lanes4-final.so` is the saved library used in that run.

The first packed implementation forced ordered block reduction in the image decoder. Its median prefill was 11472.420 ms, but it changed the caption from ‘located’ to ‘positioned’. That policy was rejected. The retained packing preserves each original partial sum and the final reduction tree; its preliminary median was 12621.447 ms and all caption tokens matched the original. The production defaults and final paired evidence are recorded separately in `README.md`.

Both existing ordinary and explicitly ordered Q4 contracts remain supported. Packed-vs-original comparisons were bit-identical on actual GGUF matrices and synthetic ragged ranges, including untouched output sentinels. Rust tests cover both modes, rows, batch tails, signed scales, and dispatch fallbacks. The ordered mode remains for existing callers that already require it; image inference does not select it.

A final query-tile sweep compared real Q4 gate weights, eight threads, two warmups and three measured calls. Shape is 128 queries × 4096 columns × 11008 output rows. Smaller tiles were rejected:

| Query tile | Median ms |
| --- | ---: |
| 1 | 16.088327 |
| 2 | 13.493386 |
| 4 | 13.006109 |

Earlier dense dequantization, row-pair packing, larger tiles and alternate loop layouts were also profiled and rejected; their raw microbenchmarks and sources remain in the archive. Full-run measurements, not microbenchmarks alone, determine acceptance.
