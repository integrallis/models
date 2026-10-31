# Published chat repairs: targeted source validation

Measured 2026-10-31 on one m6a.4xlarge Linux CPU worker. Raw prompts, token IDs,
answers, sampling controls, host and classpath hashes are retained alongside this file.
Product candidates are source-built Models b4c724a1 and ModelJars fe61f10, combined
with the released unchanged dependencies listed in classpath.json. This is not a
released-runtime qualification or a general accuracy benchmark.

| Exact artifact | Exact requested answer / 8 | Same input-token count / 8 | Identical raw output / 8 |
| --- | ---: | ---: | ---: |
| Qwen3.5 0.8B Q5_K_M | 8 | 8 | 8 |
| Llama 3.2 1B Q5_K_M | 5 | 8 | 6 |
| Llama 3.2 1B Q6_K | 5 | 8 | 8 |
| Qwen2.5 Math 1.5B Q4_K_M | 0 | 8 | 6 |

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

Reference: Ollama 0.40.13, raw identical prompts and SHA-verified identical GGUFs,
CPU only, eight threads, fresh candidate context per case. Matching token counts
is a boundary check, not proof of token-by-token identity. Historical manual-BOS
experiments are not parity evidence because Ollama added a second BOS.

Campaign: quality-fixed-20261031T174724Z. Additional published-model results are
still being collected separately; their absence here is not a passing result.
