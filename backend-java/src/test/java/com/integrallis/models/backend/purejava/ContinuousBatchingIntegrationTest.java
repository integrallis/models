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
import com.integrallis.models.backend.purejava.fixture.ModelFixtureRegistry;
import com.integrallis.models.backend.purejava.fixture.ModelFixtureRequirement;
import com.integrallis.models.runtime.ContinuousBatchingOptions;
import com.integrallis.models.runtime.InferencePipeline;
import com.integrallis.models.runtime.TextGenerationSession;
import com.integrallis.models.runtime.chat.ChatMessage;
import com.integrallis.models.runtime.chat.ChatTemplate;
import java.time.Duration;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/** Real-weight gate for the runtime scheduler over the public pipeline/session API. */
@Tag("integration")
class ContinuousBatchingIntegrationTest {
  private static final ModelFixtureRequirement QWEN3_0_6B_Q4_0 =
      ModelFixtureRequirement.of("hf://ggml-org/Qwen3-0.6B-GGUF")
          .version("[3.0.0,4.0.0)")
          .variant("q4_0")
          .backend("pure-java")
          .capability("text-generation");

  @Test
  void producesTokenExactAnswersWhileSharingPhysicalModelSteps() throws Exception {
    var fixture = ModelFixtureRegistry.fromClasspath().resolve(QWEN3_0_6B_Q4_0).orElseThrow();
    var path = fixture.localPath().orElseThrow();
    var batching =
        ContinuousBatchingOptions.builder()
            .maximumBatchSize(2)
            .batchFormationDelay(Duration.ofMillis(10))
            .build();
    var options = SamplingOptions.builder().temperature(0).maxTokens(8).build();
    var prompt =
        ChatTemplate.CHATML_NO_THINK.render(List.of(ChatMessage.user("Reply with exactly: READY")));

    String expected;
    try (PureJavaBackend baselineBackend = PureJavaBackend.load(path);
        InferencePipeline baselinePipeline = new InferencePipeline(baselineBackend);
        TextGenerationSession baseline = baselinePipeline.openGenerationSession()) {
      expected = baseline.generate(prompt, options);
    }

    try (PureJavaBackend backend = PureJavaBackend.load(path);
        InferencePipeline pipeline = new InferencePipeline(backend, batching);
        TextGenerationSession first = pipeline.openGenerationSession();
        TextGenerationSession second = pipeline.openGenerationSession();
        var callers = Executors.newFixedThreadPool(2)) {
      CountDownLatch ready = new CountDownLatch(2);
      CountDownLatch start = new CountDownLatch(1);
      var firstResult = callers.submit(() -> generate(first, prompt, options, ready, start));
      var secondResult = callers.submit(() -> generate(second, prompt, options, ready, start));

      assertThat(ready.await(5, TimeUnit.SECONDS)).isTrue();
      start.countDown();
      String firstAnswer = firstResult.get(2, TimeUnit.MINUTES);
      String secondAnswer = secondResult.get(2, TimeUnit.MINUTES);

      assertThat(firstAnswer).isNotBlank().isEqualTo(expected).isEqualTo(secondAnswer);
      assertThat(pipeline.continuousBatchingMetrics().orElseThrow().largestBatch()).isEqualTo(2);
      assertThat(pipeline.continuousBatchingMetrics().orElseThrow().meanBatchSize())
          .isGreaterThan(1.0);
      assertThat(first.lastGenerationMetrics().successful()).isTrue();
      assertThat(second.lastGenerationMetrics().successful()).isTrue();
    }
  }

  private static String generate(
      TextGenerationSession session,
      com.integrallis.models.api.ModelPrompt prompt,
      SamplingOptions options,
      CountDownLatch ready,
      CountDownLatch start)
      throws InterruptedException {
    ready.countDown();
    start.await();
    return session.generate(prompt, options);
  }
}
