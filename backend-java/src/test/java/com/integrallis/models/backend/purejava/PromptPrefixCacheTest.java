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

import com.integrallis.models.api.SamplingOptions;
import com.integrallis.models.api.Tokenizer;
import com.integrallis.models.runtime.InferencePipeline;
import com.integrallis.models.runtime.PromptPrefixCacheMetrics;
import com.integrallis.models.runtime.PromptPrefixCacheOptions;
import com.integrallis.models.test.NanoGgufModel;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;

/**
 * The prompt-prefix cache (#474) on a real pure-Java backend over the nano llama model: what it
 * reuses, what it refuses, and that its outputs are the outputs of the same prompts evaluated from
 * scratch.
 */
@Tag("unit")
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class PromptPrefixCacheTest {

  private Path model;

  /** Vocabulary pieces that encode to one token each when concatenated ("t10" .. "t31"). */
  private static final int FIRST_PIECE = 10;

  private static final int PIECES = 22;

  /** Tokens the tokenizer adds before the pieces (a BOS, if it adds one). */
  private int overhead;

  @BeforeAll
  void model() throws IOException {
    Path dir = Files.createTempDirectory("prompt-prefix-cache-");
    model = NanoGgufModel.write(dir.resolve("nano.gguf"));
    try (PureJavaBackend backend = PureJavaBackend.load(model)) {
      Tokenizer tokenizer = backend.tokenizer();
      overhead = tokenizer.encode(text(0, 1)).length - 1;
      assertThat(overhead).isBetween(0, 1);
      for (int count : new int[] {3, 7, 20}) {
        int[] encoded = tokenizer.encode(text(0, count));
        assertThat(encoded.length)
            .as("%d pieces encode to %d tokens", count, count)
            .isEqualTo(count + overhead);
      }
    }
  }

  @AfterAll
  void cleanup() throws IOException {
    Files.deleteIfExists(model);
    Files.deleteIfExists(model.getParent());
  }

  /** {@code length} pieces starting at piece {@code offset}, cycling through the vocabulary. */
  private static String text(int offset, int length) {
    StringBuilder out = new StringBuilder();
    for (int index = 0; index < length; index++) {
      out.append('t').append(FIRST_PIECE + (offset + index) % PIECES);
    }
    return out.toString();
  }

  private static SamplingOptions greedy(int tokens) {
    return SamplingOptions.builder().temperature(0.0f).maxTokens(tokens).build();
  }

  private InferencePipeline cached(long budget) {
    return new InferencePipeline(
        PureJavaBackend.load(model), PromptPrefixCacheOptions.withMaximumBytes(budget));
  }

  private InferencePipeline plain() {
    return new InferencePipeline(PureJavaBackend.load(model));
  }

  /** A conversation's next turn extends the previous prompt: only the new tokens are evaluated. */
  @Test
  void aFollowUpTurnReusesEverythingButTheNewTokensAndAnswersTheSame() {
    String first = text(0, 12);
    String second = first + text(5, 6);
    List<String> expected = new ArrayList<>();
    try (InferencePipeline reference = plain()) {
      expected.add(reference.generate(first, greedy(3)));
      expected.add(reference.generate(second, greedy(3)));
    }
    try (InferencePipeline pipeline = cached(1L << 30)) {
      int firstTokens = pipeline.tokenize(first).length;
      String a = pipeline.generate(first, greedy(3));
      PromptPrefixCacheMetrics afterFirst = pipeline.promptPrefixCacheMetrics().orElseThrow();
      assertThat(afterFirst.lookups()).isEqualTo(1);
      assertThat(afterFirst.misses()).isEqualTo(1);
      assertThat(afterFirst.entries()).isEqualTo(1);
      assertThat(afterFirst.heldBytes()).isPositive();

      String b = pipeline.generate(second, greedy(3));
      PromptPrefixCacheMetrics afterSecond = pipeline.promptPrefixCacheMetrics().orElseThrow();
      assertThat(afterSecond.hits()).isEqualTo(1);
      // the first prompt less its last token, the generation cue, is what was published
      assertThat(afterSecond.reusedTokens()).isEqualTo(firstTokens - 1);
      assertThat(afterSecond.entries())
          .as("the longer entry replaces the one it extends")
          .isEqualTo(1);
      assertThat(pipeline.lastGenerationMetrics().promptCache().cacheReadInputTokens())
          .isEqualTo(pipeline.tokenize(second).length - 1);

      assertThat(List.of(a, b)).isEqualTo(expected);
    }
  }

  /** Two conversations with one system prompt share it; the second evaluates only its own turn. */
  @Test
  void aSecondConversationSharesTheSystemPromptThroughATruncatedView() {
    String system = text(0, 16);
    String conversationA = system + text(2, 5);
    String conversationB = system + text(9, 5);
    List<String> expected = new ArrayList<>();
    try (InferencePipeline reference = plain()) {
      expected.add(reference.generate(conversationA, greedy(3)));
      expected.add(reference.generate(conversationB, greedy(3)));
    }
    try (InferencePipeline pipeline = cached(1L << 30)) {
      assertThat(pipeline.supportsPortablePromptPrefixes()).as("truncated views").isTrue();
      int[] a = pipeline.tokenize(conversationA);
      int[] b = pipeline.tokenize(conversationB);
      int common = 0;
      while (common < a.length && common < b.length && a[common] == b[common]) common++;
      assertThat(common).isLessThan(a.length - 1);

      String answerA = pipeline.generate(conversationA, greedy(3));
      String answerB = pipeline.generate(conversationB, greedy(3));
      PromptPrefixCacheMetrics metrics = pipeline.promptPrefixCacheMetrics().orElseThrow();
      assertThat(metrics.hits()).isEqualTo(1);
      assertThat(metrics.reusedTokens()).isEqualTo(common);
      assertThat(metrics.entries()).as("both conversations stay cached").isEqualTo(2);
      assertThat(List.of(answerA, answerB)).isEqualTo(expected);
    }
  }

  @Test
  void aPromptSharingNothingMissesAndIsCached() {
    try (InferencePipeline pipeline = cached(1L << 30)) {
      pipeline.generate(text(0, 10), greedy(2));
      pipeline.generate(text(7, 10), greedy(2));
      PromptPrefixCacheMetrics metrics = pipeline.promptPrefixCacheMetrics().orElseThrow();
      assertThat(metrics.hits()).isZero();
      assertThat(metrics.misses()).isEqualTo(2);
      assertThat(metrics.entries()).isEqualTo(2);
    }
  }

  /** A budget smaller than one entry keeps only the newest, and the evicted prompt misses again. */
  @Test
  void theByteBudgetEvictsTheLeastRecentlyUsedEntry() {
    String first = text(0, 10);
    String second = text(7, 10);
    try (InferencePipeline pipeline = cached(1)) {
      pipeline.generate(first, greedy(2));
      pipeline.generate(second, greedy(2));
      PromptPrefixCacheMetrics afterTwo = pipeline.promptPrefixCacheMetrics().orElseThrow();
      assertThat(afterTwo.entries()).isEqualTo(1);
      assertThat(afterTwo.evictions()).isEqualTo(1);
      pipeline.generate(first + text(3, 4), greedy(2));
      PromptPrefixCacheMetrics afterThree = pipeline.promptPrefixCacheMetrics().orElseThrow();
      assertThat(afterThree.hits()).as("the first prompt was evicted").isZero();
      assertThat(afterThree.evictions()).isEqualTo(2);
    }
  }

  @Test
  void withoutTheOptionThePipelineHasNoCache() {
    try (InferencePipeline pipeline = plain()) {
      pipeline.generate(text(0, 8), greedy(2));
      assertThat(pipeline.promptPrefixCacheMetrics()).isEmpty();
    }
  }
}
