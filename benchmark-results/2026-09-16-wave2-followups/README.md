# Wave-2 follow-ups: end-of-turn tokens, `</s>` in stop sets, two cliffs, GGUF alignment

Date: 2026-09-16. Branch `feat/wave2-followups`. This note covers items 1 to 4. Item 5 (LangChain4j
cancellation) is covered by its tests and the PR body. Item 6 (the flaky test) has its own analysis in
[`../2026-09-16-flaky-injected-attention/README.md`](../2026-09-16-flaky-injected-attention/README.md).

Labels: **measured** means run here. **Read** means read in a file or source nobody here wrote, with the
place it was read. **Believed** means neither.

## Host and data

| | |
|---|---|
| Host | MacBook Pro, Intel Core i7-9750H (6C/12T, AVX2), macOS (Darwin 25.6.0, x86_64) |
| JVM | Temurin 25.0.3+9 |
| GGUF files | 74 `.gguf` files found under `~/.jvllm/models`, `~/.cache` and `~/Code`. 36 are model files. 38 are llama.cpp vocabulary-only fixtures: 19 in `~/Code/llama.cpp/models` (llama.cpp commit `a58222229`) and 19 duplicates vendored by llama-cpp-python 0.3.34. Deduplicated, that leaves **55 vocabularies** |
| Harness | `harness/*.java`, compiled against the `backend-java` test classpath. `EogAll` prints each stop set, `EogProv` prints provenance and the template resolution, `TypeProbe` prints token types of entries matched by text, `ParseAll` parses every file, `EotProto` is the resolver prototype, `Render` renders each native template, `CliffLoad` loads a model and prints diagnostics |

The chat templates are not copied here, because they are the model publishers' text.
`chat-template-hashes.txt` lists the SHA-256 prefix and size of each template that was read.

## Item 2: token id 128247 in the Qwen stop sets

**What it is (measured).** In all 16 Qwen2-vocabulary GGUFs on the host (Qwen2.5-Coder, Qwen2.5-Math,
Qwen3, DeepSeek-R1-Distill-Qwen, HuatuoGPT-o1, Fin-R1, Hammer 2.1) id 128247 is `</s>`, and its
`tokenizer.ggml.token_type` is 1 (NORMAL). Its neighbours are also ordinary text: 128243 `<unk`,
128244 `<unk>`, 128245 `<s>`, 128246 `</s`. In the Hugging Face `tokenizer.json` of Hammer 2.1 0.5B, a
Qwen2 vocabulary, `</s>` is vocabulary id 128247. It is not an added token. The only way to reach it is
the merge `["</s", ">"]`. `eos_token` there is `<|im_end|>`.

**The rule that added it (read).** It came from `GgufTokenizer.END_OF_GENERATION_TOKEN_TEXTS`, which
marks every vocabulary entry whose text is in the list. The list is a copy of llama.cpp's
(`src/llama-vocab.cpp`, lines 2774-2797 at `a58222229`). llama.cpp added `</s>` to it with the comment
`// paddleocr`. llama.cpp applies the same heuristic, and it overrides a NORMAL type to CONTROL with the
warning "this is probably a bug in the model".

**Is it correct? No.** Upstream `generation_config.json` (read 2026-09-16) gives:

- Qwen2.5-Coder-0.5B-Instruct and Qwen3-0.6B: `[151645, 151643]`
- Gemma 3 1B IT (unsloth mirror, because the Google repository is gated): `[1, 106]`
- Phi-3-mini-4k-instruct: `[32000, 32001, 32007]`

The same heuristic also put Gemma 3's id 212 (`</s>`) in its stop set. That id is typed USER_DEFINED
and sits inside a block of HTML tags: 203 `<s>`, 209 `</b>`, 212 `</s>`, 215 `</code>`, 227 `</a>`
(measured). Being in the stop set also meant the tokenizer dropped these ids from decoded text and
parsed them as atomic control tokens inside trusted prompt segments.

**How much it matters (measured, and believed where marked).** Plain-text encoding of
`Use <s>old price</s> for strikethrough.` never produces 128247. The Qwen2 pre-tokenizer splits the
text into `</`, `s`, `>`. So the model is unlikely ever to have seen that id in training (believed), and
a real generation would rarely reach it. The fix removes a wrong member from the stop set. It does not
change how a common output ends.

**Fix.**

- A text match no longer counts when the file types the token NORMAL.
- `</s>` additionally has to be typed CONTROL.
- Ids declared in metadata are unaffected.
- `tokenizer.json` vocabularies have no token types. There, when the tokenizer has added tokens, only
  added tokens can match by text.

**Effect on real vocabularies (measured, `item2-eog-before.txt` → `item2-eog-after.txt`).** Across the
55 vocabularies, only these ids left a stop set:

- 128247, from all 16 Qwen2-vocabulary sets
- 212, from Gemma 3
- 2, from Phi-3's set. The "before" state is inferred, because the old parser could not open the
  vocabulary-only files (see item 4). Phi-3's `</s>` is USER_DEFINED and is not in upstream's list.

These stay:

- Gemma 4's `<|tool_response>` (USER_DEFINED)
- EuroLLM's `</s>` (CONTROL)
- Aquila's `</s>` id 100007 (CONTROL). Upstream declares EOS 2, so it is likely extra as well, but no
  type evidence separates it. Left alone.

**Pinned oracles (measured).** `:backend-java:integrationTest` at commit `60c3dde8` (this item applied):
142 tests, 0 failed, 30 skipped (fixtures not configured on this host). No pinned greedy oracle changed.

Red: `item2-red.txt` (3 of 4 new tests failing). Green: `item2-green.txt`. Real-fixture test:
`EndOfGenerationFixtureIntegrationTest`.

## Item 1: end-of-turn tokens from the chat template

**Starting point (measured).** Before this change, Models' stop sets already contained the end-of-turn
token for both cases the ModelJars profile work named, through the vocabulary-text rule and not through
the declared EOS: Gemma 3 1B `{1, 106 <end_of_turn>}` and MiniCPM5 1B `{1, 130073 <|im_end|>}`. The
ModelJars finding holds for the declared metadata, but it did not show up as a runtime stop-set gap in
Models on any local file.

**GGUF `tokenizer.chat_template`.** The template is not evaluated. The resolver:

1. Finds every occurrence of a CONTROL-typed vocabulary entry, and of the `eos_token` variable, in the
   template text.
2. Takes the last occurrence before the final `add_generation_prompt` as the candidate.
3. Accepts the candidate only if (a) it is not the token that opens the generation prompt and (b)
   somewhere in the template it is the first marker after a `content` reference.

Otherwise the result is `unresolved:<reason>`, and the set is left as the other rules built it.
USER_DEFINED entries (`<think>`, `<tool_call>`) are never candidates. The Harmony/Solar and Gemma 4
exclusions run after this rule, so it cannot undo them.

The first prototype (`EotProto`) also tried voting on "first marker after content" alone. That was
noisy: `<|im_start|>` won the vote on Qwen3, and `<start_of_image>` won it on Gemma 3. The combined rule
is what shipped.

**Resolution on the 55 vocabularies (measured, `item1-provenance.txt`):**

| outcome | files |
|---|---|
| resolved | 26: Qwen2.5/Qwen3/Qwen3.5 family → `<|im_end|>`; DeepSeek-R1-Distill-Qwen → `<｜end▁of▁sentence｜>`; Gemma 3 → `<end_of_turn>`; Gemma 4 vocab → `<turn|>`; Granite 4.1 → `<|end_of_text|>`; SmolLM2, SmolLM3, MiniCPM5 → `<|im_end|>`; TinyLlama → `</s>` via `eos_token`; Command-R vocab → `<|END_OF_TURN_TOKEN|>` |
| unresolved: no `add_generation_prompt` block | 5 (Hammer 2.1 ×3, EuroLLM, Phi-3) |
| no template | 24 |

Whether each resolved marker is the right one was checked by reading its template. That check is
**read**, not measured by generation. No file resolved to a wrong marker.

**No stop set changed (measured).** `item1-eog-after.txt` is identical to `item2-eog-after.txt`: every
resolved marker was already present through metadata or the vocabulary-text rule. What this item adds on
current data is provenance plus coverage for a future template whose marker is not in the fixed list.
Whether such a GGUF exists is **no data**, because none is on this host.

**Provenance (measured).** `GgufTokenizer.endOfGenerationSources()` lists the rules behind each id.
`PureJavaBackend` diagnostics (and therefore `RustFfmBackend` diagnostics) carry three keys:
`end-of-generation-token-ids`, `end-of-generation.<id>` and `end-of-generation.chat-template`.
Qwen3 0.6B, from `item3-real-process.txt`:

```
end-of-generation-token-ids=151643,151645
end-of-generation.151643=vocabulary-text
end-of-generation.151645=tokenizer.ggml.eos_token_id,vocabulary-text,chat-template-end-of-turn
end-of-generation.chat-template=resolved:151645
```

**Native `ChatTemplate` (models-runtime).**

- `endOfTurnMarker()` names each family's terminator. A parameterised test pins it to what the renderer
  writes after an assistant message, with GPT-OSS as the documented exception (`<|return|>`, not
  `<|end|>`).
- `endOfTurnTokenId(Tokenizer)` and `endOfTurnStopsGeneration(Tokenizer)` resolve and check it against a
  loaded tokenizer.
- The fixture integration test checks eight pinned GGUF and native template pairings, and each marker is
  in the stop set (measured): Qwen2.5-Coder/CHATML, Qwen3/CHATML_NO_THINK, SmolLM2/CHATML,
  TinyLlama/ZEPHYR, Gemma 3/GEMMA, MiniCPM5/MINICPM5_NO_THINK, DeepSeek-Coder/DEEPSEEK, EuroLLM/CHATML.

**Not done: injecting a native template's marker into the stop set at runtime.** The backend never learns
which `ChatTemplate` a caller chose. Templates are attached in the framework adapters, and
`GenerationLoop` stops only on `backend.tokenizer().isEndOfGeneration`. Wiring this would need a new
channel, for example stop token ids on `SamplingOptions` or a tokenizer view passed to the loop. That is
an API change across models-api, models-runtime and every adapter, so it was left out of this PR.

The one native family whose marker is outside the vocabulary-text list is MOBILE_MOE (`<|eot|>`). It
loads through the `tokenizer.json` path with ids declared from `generation_config.json`. No MobileMoE
checkpoint is on this host, so whether `<|eot|>` is in its runtime stop set is **no data**.

**Cost.** Tokenizer construction was timed best-of-8 on Gemma 3 1B and Qwen3.5 4B, with and without the
resolver: 202-333 ms against 256-393 ms. Those runs were uncontrolled (an integration suite was running
on the same host at the time), and the two ranges overlap, so no cost claim is made.

Red: `item1-red.txt` (6 of 8 template tests failing against stubbed accessors; diagnostics test failing).
The native-template accessors were red only at compile level. Green: `item1-green.txt`.

## Item 3: two unnamed slow paths

| reason | emitted at | provoked by | fast-path control |
|---|---|---|---|
| `gguf-parallel-disabled` | `ExecutionPlanner.persistentExecutor`, when `ggufParallel` is false and `processors > 1` | unit: fingerprint with parallel false, 8 processors, 3 plans → 1 event. **Real process (measured):** Qwen3 0.6B with `-Dvectors.gguf.parallel=false` → `performance-cliff.gguf-parallel-disabled=gguf-parallel=false, executor=persistent, processors=12` | same fingerprint with 1 processor → 0 events. Real process without the flag → key absent |
| `native-grouped-attention-span-limit` | `LlamaForwardPass.attendNatively`, the `spanCount() > 2` return | unit: Granite with a kernel that supports grouped attention; prefix frozen from an already-forked session (3 spans), 2 decode steps → 1 event, detail `spans=3`, and the kernel is called 0 times for those steps | 2 spans (one freeze, one fork) → 0 events, and the kernel is called once per step |

The span-limit cliff was not provoked in a real process. That would need Granite 4.1 on the native
kernel with nested prefix forks, which was not run. The cost of either path was **not measured**.

Red: `item3-red.txt`. Green: `item3-green.txt`.

## Item 4: GGUF alignment

**Read.** llama.cpp's reader (`ggml/src/gguf.cpp` at `a58222229`, lines 755-786) is stricter than the
rule added here. It requires each tensor offset to equal the running sum of the previous tensors' sizes
padded to the alignment, and it seeks to the aligned data start only when `n_tensors > 0`.

**Change.** Three commits:

1. The synthetic writers (`SyntheticGgufBuilder`, `NanoGgufModel`) now pad tensors to alignment. Two
   overlap tests declare `general.alignment` 4 so they keep testing the size check.
2. `GgufParser` rejects any tensor offset that is not a multiple of `general.alignment` (default 32),
   naming the tensor, its offset and the alignment.
3. A file with no tensors no longer fails because its aligned data start lies past the end of the file.

**Real files (measured, `item4-scan-before.txt` / `item4-scan-final.txt`).**

- Before: 36 accepted, 38 rejected. Every rejection was a llama.cpp vocabulary-only file, failing with
  `aligned tensor data offset N exceeds file size M`.
- With the alignment check alone: the same 36 and 38, with an identical verdict for every file.
- After the vocabulary-only fix: **74 accepted, 0 rejected.**
- All 74 declare or default to alignment 32. A non-default alignment is exercised only by synthetic
  tests, so on real files that case is no data.

Red: `item4-red.txt`, `item4b-red.txt`. Green: `item4-green.txt`, `item4b-green.txt`.
