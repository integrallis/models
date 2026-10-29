# Flint GGUF execution

Flint is an experimental Models architecture. This is runtime work toward
ModelJars admission, not a published or performance-qualified ModelJar.

The decoder reads `general.architecture=flint` and executes the leading dense layers, sigmoid
top-k routed experts plus the shared expert, per-head Q/K RMS normalization, learned attention
sinks, and independent per-layer sliding-window and RoPE patterns. All large tensors remain
memory-mapped. Full-attention layers have linear KV storage; sliding layers use bounded ring
buffers. Sessions are independent. Rewinding replays the prefix only when the required sliding
window has already been overwritten. Prefill runs sequentially and skips intermediate vocabulary
projections.

`ChatTemplate.FLINT` renders the publisher's text envelope. Retain `<think>...</think>` in
assistant history. Its native XML argument protocol is not yet implemented by the tool parser;
the template explicitly refuses tool requests rather than labelling them as Qwen tool calls.
The tokenizer implements Flint's sequential digit/CJK/word splitting and recognizes its EOS
and user, developer, and observation transitions as generation terminators.

## Checkpoint

The integration fixture is the following exact artifact:

- Repository: `ggml-org/Flint-GGUF`
- Revision: `468307a1b38259e9e4c9d4fe53d4e3b76386572c`
- File: `Flint-Q4_K_M.gguf`
- Size: `23387911136` bytes
- SHA-256: `c3c4d93f9db92627a2fc47ea8fe8c64a094cabd3549db544bcdd2c00e9a9a347`
- Source: [Recursive Flint](https://huggingface.co/RecursiveSI/flint), Apache-2.0

Its 49 layers comprise 2 leading dense layers and 47 MoE layers; each routed layer selects 6
of 256 experts. Query width is 4096, distinct from the 2048 model width. There are 35 sliding
layers with a 513-token window and 14 full-attention layers. Only sliding layers apply RoPE.
The fused shared tensors have width 1024; the GGUF shared-count value of 2 describes two
512-wide expert units, not two separately routed shared branches.

Set an explicit context budget before loading on a laptop, for example
`-Dmodels.purejava.maxContextLength=4096`. The model advertises 262144 tokens, and allocating
the full context requires much more memory than the weights alone. A 32 GB Intel laptop is
a constrained correctness-test host for this 23.4 GB file, not the performance target.

Use the local GGUF directly from a build of this branch with `backend-java` and `models-runtime`
on the classpath. Launch Java 25+ with `--add-modules=jdk.incubator.vector` and the context property
above, then render the prompt explicitly:

```java
import com.integrallis.models.api.SamplingOptions;
import com.integrallis.models.backend.purejava.PureJavaBackend;
import com.integrallis.models.runtime.InferencePipeline;
import com.integrallis.models.runtime.chat.ChatMessage;
import com.integrallis.models.runtime.chat.ChatTemplate;
import java.nio.file.Path;
import java.util.List;

var prompt = ChatTemplate.FLINT.render(List.of(ChatMessage.user("What is 2 + 2?")));
var options = SamplingOptions.builder().temperature(0).maxTokens(128).build();
try (var pipeline = new InferencePipeline(PureJavaBackend.load(Path.of("Flint-Q4_K_M.gguf")))) {
    System.out.println(pipeline.generate(prompt, options));
}
```

## Validation

Fast tests, requiring no checkpoint download:

```sh
./gradlew :backend-java:test --tests '*Flint*' \
  :models-runtime:test --tests '*Flint*' --tests '*ChatTemplateEndOfTurnTest'
```

The F32 fixture's full logit vectors are independently generated with llama.cpp commit
`0893ec98f5eafad36a43a72cd1c3723d72b0d03a`, CPU inference, F32 KV, and flash attention disabled.
The reference helper disables extra repacked weight buffers to keep the real checkpoint mapped.
The test covers leading dense and routed layers, selection bias, shared experts, sinks, Q/K
normalization, selective RoPE, and positions past the sliding window. The absolute tolerance is
`2e-5`; the oracle values are not captured from the Java implementation.

Reproduce the reference in a separate llama.cpp checkout pinned to that commit:

```sh
cmake -S /path/to/llama.cpp -B /path/to/llama.cpp/build \
  -DGGML_METAL=OFF -DGGML_BLAS=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build /path/to/llama.cpp/build -j 4 --target llama
c++ -std=c++17 -O2 scripts/flint/reference-logits.cpp \
  -I/path/to/llama.cpp/include -I/path/to/llama.cpp/ggml/include \
  -L/path/to/llama.cpp/build/bin -Wl,-rpath,/path/to/llama.cpp/build/bin \
  -lllama -lggml -lggml-base -o /tmp/flint-reference-logits
./gradlew :backend-java:flintReferenceFixture
/tmp/flint-reference-logits backend-java/build/flint-reference/flint-f32.gguf 3,4,5,6,7,8
```

The real-checkpoint gate downloads and verifies the pinned file before running, and does not
silently skip if the artifact is absent:

```sh
./gradlew :backend-java:flintIntegrationTest
```

Use `-Dmodels.fixtures.directory=/path/to/models` to reuse an existing verified download.
The test uses a 128-token context and a 2 GB heap; the mapped weights are additional memory.
It verifies the tokenized prompts and eight greedy continuation tokens for each of a completion
and a chat prompt. The independent expectations are in
`backend-java/src/test/resources/flint/reference.properties`. Reproduce them with:

```sh
/tmp/flint-reference-logits /path/to/Flint-Q4_K_M.gguf 'text:The quick brown fox' 7
/tmp/flint-reference-logits /path/to/Flint-Q4_K_M.gguf \
  'text:<|user|>What is 2 + 2?<|assistant|>' 7
```

For each run, take the `top` values starting at the last prompt row. These short continuations
exercise execution; they do not assess answer quality or completed reasoning.

### Development-laptop observation

On 2026-10-29 the pinned Q4_K_M test passed on a six-core 2.6 GHz Intel Core i7 MacBook Pro
with 32 GB RAM, Java 25.0.3, 256-bit Vector API operations, and the default 12-worker vector
executor. Both prompts matched the reference tokenization and all 16 continuation tokens matched.

| Prompt | Prompt tokens | Prefill | Subsequent tokens | Decode time | Decode rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Completion | 4 | 11.098 s | 7 | 10.024 s | 0.70 tokens/s |
| Chat | 10 | 12.419 s | 7 | 9.192 s | 0.76 tokens/s |

This was one short, uninstrumented test invocation, with a 128-token context and 2 GB heap.
The file had already been read for checksum verification and reference generation. There was
no warmup protocol, isolated-host control, or sustained throughput measurement. The observation
shows that this laptop can execute the checkpoint correctly on these cases; interactive latency
is high. It is not a performance qualification or evidence of frontier-LLM answer quality.

## Publication work remaining

Controlled default-correctness and RAG performance qualification, longer-context validation,
tool protocol integration and its separate qualification, and acceleration measurements remain
separate gates. No claim about all Flint quantizations, frontier-model parity, or catalog
admission follows from the implementation or short numerical comparisons.
