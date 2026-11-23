/*
 * Copyright 2025-2026 Integrallis Software, LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.integrallis.models.backend.purejava;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.integrallis.models.api.SamplingOptions;
import com.integrallis.models.runtime.InferencePipeline;
import com.integrallis.models.runtime.PromptPrefixCacheMetrics;
import com.integrallis.models.runtime.PromptPrefixCacheOptions;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * The prompt-prefix cache (#474) on a real model: a three-turn conversation and a second
 * conversation sharing its system prompt, with the cache off and on. Outputs must be identical; the
 * per-turn prefill times and reuse counts are printed as this machine's smoke, not as a
 * measurement.
 */
@Tag("model-fixture")
class PromptPrefixCacheFixtureIntegrationTest {

  private static final String SYSTEM =
      "<|im_start|>system\nYou are a careful assistant for a software team. Answer briefly, in"
          + " plain prose, and never invent file names. The repository is a Java library for"
          + " running language models on the JVM; its modules are models-api, models-runtime,"
          + " backend-java, backend-native and the Spring AI and LangChain4j adapters."
          + " Tools: read_file(path), list_dir(path), run_tests(module).<|im_end|>\n";

  private static Path fixture() {
    String dir =
        System.getProperty(
            "models.fixtures.directory",
            Path.of(System.getProperty("user.home"), ".jvllm", "models").toString());
    return Path.of(
        dir, System.getProperty("models.fixtures.prefixCacheModel", "Qwen3-1.7B-Q8_0.gguf"));
  }

  private static String turn(String history, String user) {
    return history + "<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n";
  }

  private static SamplingOptions greedy() {
    return SamplingOptions.builder().temperature(0.0f).maxTokens(16).build();
  }

  @Test
  void aConversationAndASiblingReuseTheirSharedPrefixWithIdenticalAnswers() {
    Path model = fixture();
    assumeTrue(Files.isRegularFile(model), "no model fixture at " + model);
    List<String> prompts = new ArrayList<>();
    String history = SYSTEM;
    String p1 = turn(history, "Which module holds the Spring AI adapter?");
    prompts.add(p1);
    // the second turn carries the first answer back, as an agent client would
    String p2 = turn(p1 + "models-spring-ai holds it.<|im_end|>\n", "And the LangChain4j one?");
    prompts.add(p2);
    String p3 =
        turn(p2 + "models-langchain4j holds it.<|im_end|>\n", "Which tool lists a directory?");
    prompts.add(p3);
    // a second conversation with the same system prompt and tools
    prompts.add(turn(SYSTEM, "Name one backend module."));

    List<String> cold = new ArrayList<>();
    List<Long> coldPrefillSideMillis = new ArrayList<>();
    try (InferencePipeline reference = new InferencePipeline(PureJavaBackend.load(model))) {
      for (String prompt : prompts) {
        // a fresh context each time: the cold arm must not benefit from the loop's own rewind reuse
        reference.resetContext();
        long started = System.nanoTime();
        cold.add(reference.generate(prompt, greedy()));
        long wall = (System.nanoTime() - started) / 1_000_000L;
        coldPrefillSideMillis.add(wall - reference.lastGenerationMetrics().decode().toMillis());
      }
    }
    List<String> warm = new ArrayList<>();
    List<Long> warmPrefillSideMillis = new ArrayList<>();
    List<Long> reusedPerTurn = new ArrayList<>();
    List<Long> evaluatedPerTurn = new ArrayList<>();
    List<Integer> promptTokens = new ArrayList<>();
    PromptPrefixCacheMetrics cacheMetrics;
    try (InferencePipeline pipeline =
        new InferencePipeline(
            PureJavaBackend.load(model), PromptPrefixCacheOptions.withMaximumBytes(2L << 30))) {
      long reusedBefore = 0;
      long evaluatedBefore = 0;
      for (String prompt : prompts) {
        promptTokens.add(pipeline.tokenize(prompt).length);
        long started = System.nanoTime();
        warm.add(pipeline.generate(prompt, greedy()));
        long wall = (System.nanoTime() - started) / 1_000_000L;
        // the call's wall less its decode: the lookup, the evaluation of what was new, the publish
        // and the fork's one-token step, which the fork's own prefill metric leaves out
        warmPrefillSideMillis.add(wall - pipeline.lastGenerationMetrics().decode().toMillis());
        PromptPrefixCacheMetrics now = pipeline.promptPrefixCacheMetrics().orElseThrow();
        reusedPerTurn.add(now.reusedTokens() - reusedBefore);
        evaluatedPerTurn.add(now.evaluatedTokens() - evaluatedBefore);
        reusedBefore = now.reusedTokens();
        evaluatedBefore = now.evaluatedTokens();
      }
      cacheMetrics = pipeline.promptPrefixCacheMetrics().orElseThrow();
    }

    System.out.println(
        "prompt-prefix cache on "
            + model.getFileName()
            + " (this machine's smoke, not a measurement)");
    System.out.println(
        "| turn | prompt tokens | reused | evaluated | cold prefill-side ms | cached prefill-side ms |");
    System.out.println("|---|---|---|---|---|---|");
    for (int index = 0; index < prompts.size(); index++) {
      System.out.printf(
          Locale.ROOT,
          "| %d | %d | %d | %d | %d | %d |%n",
          index + 1,
          promptTokens.get(index),
          reusedPerTurn.get(index),
          evaluatedPerTurn.get(index),
          coldPrefillSideMillis.get(index),
          warmPrefillSideMillis.get(index));
    }
    System.out.println("cache: " + cacheMetrics);

    assertThat(warm)
        .as("the cache changes when tokens are evaluated, never what they yield")
        .isEqualTo(cold);
    assertThat(cacheMetrics.hits()).isEqualTo(3);
    // turns 2 and 3 extend the previous prompt: everything but the generation cue is reused
    assertThat(reusedPerTurn.get(0)).isZero();
    assertThat(reusedPerTurn.get(1)).isEqualTo((long) promptTokens.get(1) - 1);
    assertThat(reusedPerTurn.get(2)).isEqualTo((long) promptTokens.get(2) - 1);
    // the sibling conversation shares the system prompt and the user header, nothing after
    assertThat(reusedPerTurn.get(3)).isPositive().isLessThan((long) promptTokens.get(3) - 1);
    // the third turn replays two turns of history; its prefill-side time must be a fraction of cold
    assertThat(warmPrefillSideMillis.get(2))
        .as("turn 3 prefill-side time, cached vs cold")
        .isLessThan(Math.max(1, coldPrefillSideMillis.get(2) / 2));
  }
}
