# Published chat repairs: targeted source validation

Measured 2026-10-31 on two m6a.4xlarge Linux CPU workers, each running its candidate
and reference serially on the same host. Raw prompts, token IDs,
answers, sampling controls, host and classpath hashes are retained alongside this file.
Product candidates are source-built Models b4c724a1 and ModelJars 385300e, combined
with the released unchanged dependencies listed in classpath.json. This is not a
released-runtime qualification or a general accuracy benchmark.

| Exact artifact | Exact requested answer / 8 | Same input-token count / 8 | Identical raw output / 8 |
| --- | ---: | ---: | ---: |
| Qwen3.5 0.8B Q5_K_M | 8 | 8 | 8 |
| Llama 3.2 1B Q5_K_M | 5 | 8 | 6 |
| Llama 3.2 1B Q6_K | 5 | 8 | 8 |
| Qwen2.5 Math 1.5B Q4_K_M | 0 | 8 | 6 |
| EuroLLM 1.7B Q4_K_M | 0 | 8 | 4 |
| SmolLM2 1.7B Q4_K_M | 4 | 8 | 5 |
| DeepSeek R1 Distill Qwen 1.5B Q4_K_M | 1 | 8 | 1 |
| Mathstral 7B Q4_K_M | 5 | 8 | 8 |
| Mistral 7B Instruct v0.3 Q4_K_M | 6 | 8 | 8 |
| EdgeRock 4B Q4_K_M | 6 | not measured | not measured |

The Llama repair restores the implicit BOS for llama-bpe aliases. Both variants
abstain on the two unsupported-context questions after previously inventing a
museum price and library opening time. They still fail exact formatting: they
explain abstentions instead of returning UNKNOWN, and return $12 instead of 12.
Q5 and the independent runtime differ only in abstention wording here.

The ModelJars default-chat change selects Qwen3.5's non-thinking envelope, avoiding
exhausting the output budget in thoughts. Qwen Math loses the historical RAG answer
prefill but still explains its answers and incorrectly calls Japan's capital
“otsu”; the independent runtime does the same. Do not claim its general-knowledge
or concise-format accuracy is repaired. These eight questions are diagnostics;
existing specialist RAG qualifications are not replaced by them.

The publisher Mistral/Mathstral envelopes give identical raw responses to the
reference on all eight cases. All answers are substantively correct on these
questions; the exact-format misses add explanatory wording. This does not
establish a general accuracy advantage of Mathstral over Mistral.

EuroLLM gives 32 for 17 + 25 and 23 for 23 + 19 in both runtimes; its translation
qualification does not promise general chat or arithmetic accuracy. SmolLM2 gives
32 for 17 + 25 and 56 for 81 - 36 in both. DeepSeek's publisher envelope restores
its intended formatting but does not repair the wrong Japan-capital answer or
the missing-context case that exhausts 512 tokens without a final answer. The
other raw-output differences remain recorded; matching token counts alone
cannot establish numerical parity.

EdgeRock passes the requested default-demo Paris question and six exact answers.
Two other questions exhaust the 512-token budget without a final answer. This
run deliberately contains no independent EdgeRock comparator, as recorded in
its reference-scope file, and does not replace its historical publisher-runtime
RAG qualification or its performance-reduced status.

Reference: Ollama 0.40.13, raw identical prompts and SHA-verified identical GGUFs,
CPU only, eight threads, fresh candidate context per case. Matching token counts
is a boundary check, not proof of token-by-token identity. Historical manual-BOS
experiments are not parity evidence because Ollama added a second BOS.

Campaign: quality-fixed-20261031T174724Z. Both shards completed successfully;
their host and completion receipts are in shard-0 and shard-1. Nexus and
MobileMoE diagnostics are separate experiments and are not counted in this table.
