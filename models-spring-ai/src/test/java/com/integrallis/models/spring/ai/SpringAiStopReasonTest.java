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
package com.integrallis.models.spring.ai;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.models.api.BackendDiagnostics;
import com.integrallis.models.api.GenerationUsage;
import com.integrallis.models.api.SamplingOptions;
import com.integrallis.models.api.StopReason;
import com.integrallis.models.api.TextGenerationModel;
import com.integrallis.models.api.TokenStream;
import java.time.Duration;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.springframework.ai.chat.metadata.ChatGenerationMetadata;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.Prompt;

/** The runtime's typed stop reason reaches Spring AI as generation finish-reason metadata. */
@Tag("unit")
class SpringAiStopReasonTest {

  @ParameterizedTest
  @CsvSource({
    "EOS, STOP",
    "STOP_SEQUENCE, STOP",
    "CONSTRAINT_COMPLETE, STOP",
    "MAX_TOKENS, LENGTH",
    "REPETITION_LOOP, REPETITION_LOOP",
    "CANCELLED, CANCELLED"
  })
  void callCarriesTheFinishReasonAndTheExactStopReason(StopReason stopReason, String expected) {
    ModelsSpringAiChatModel model =
        new ModelsSpringAiChatModel(new ScriptedModel(List.of("an", "swer"), stopReason));

    ChatResponse response = model.call(new Prompt("question"));

    ChatGenerationMetadata metadata = response.getResult().getMetadata();
    assertThat(response.getResult().getOutput().getText()).isEqualTo("answer");
    assertThat(metadata.getFinishReason()).isEqualTo(expected);
    assertThat(metadata.<String>get("stopReason")).isEqualTo(stopReason.name());
  }

  @ParameterizedTest
  @CsvSource({"EOS, STOP", "MAX_TOKENS, LENGTH", "REPETITION_LOOP, REPETITION_LOOP"})
  void streamEndsWithTheFinishReason(StopReason stopReason, String expected) {
    ModelsSpringAiChatModel model =
        new ModelsSpringAiChatModel(new ScriptedModel(List.of("an", "swer"), stopReason));

    List<ChatResponse> responses = model.stream(new Prompt("question")).collectList().block();

    ChatGenerationMetadata last = responses.getLast().getResult().getMetadata();
    assertThat(last.getFinishReason()).isEqualTo(expected);
    assertThat(last.<String>get("stopReason")).isEqualTo(stopReason.name());
  }

  @Test
  void engineWithoutStopReasonsKeepsTheFinishReasonAbsent() {
    ModelsSpringAiChatModel model =
        new ModelsSpringAiChatModel(new ScriptedModel(List.of("answer"), null));

    assertThat(model.call(new Prompt("question")).getResult().getMetadata().getFinishReason())
        .isNull();
  }

  @Test
  void cancellingTheSubscriptionCancelsGeneration() throws InterruptedException {
    ScriptedModel delegate =
        new ScriptedModel(List.of("one", "two", "three", "four"), StopReason.EOS);
    delegate.pauseAfterFirstToken = true;
    ModelsSpringAiChatModel model = new ModelsSpringAiChatModel(delegate);

    ChatResponse first =
        model.stream(new Prompt("question")).take(1).blockFirst(Duration.ofSeconds(5));

    assertThat(first.getResult().getOutput().getText()).isEqualTo("one");
    assertThat(delegate.finished.await(5, TimeUnit.SECONDS)).isTrue();
    assertThat(delegate.reported).isEqualTo(StopReason.CANCELLED);
    assertThat(delegate.emitted).isLessThan(4);
  }

  /** Streams fixed fragments, honours cancellation, and reports a fixed stop reason. */
  private static final class ScriptedModel implements TextGenerationModel {
    private final List<String> tokens;
    private final StopReason stopReason;
    private final CountDownLatch finished = new CountDownLatch(1);
    private volatile boolean pauseAfterFirstToken;
    private volatile int emitted;
    private volatile StopReason reported;

    private ScriptedModel(List<String> tokens, StopReason stopReason) {
      this.tokens = tokens;
      this.stopReason = stopReason;
    }

    @Override
    public String modelName() {
      return "scripted";
    }

    @Override
    public BackendDiagnostics diagnostics() {
      return BackendDiagnostics.unavailable("scripted");
    }

    @Override
    public void generate(String prompt, SamplingOptions options, TokenStream stream) {
      try {
        for (String token : tokens) {
          stream.onToken(token);
          emitted++;
          if (pauseAfterFirstToken && emitted == 1) {
            awaitCancellation(stream);
          }
          if (stream.isCancelled()) {
            reported = StopReason.CANCELLED;
            stream.onComplete(new GenerationUsage(1, emitted), StopReason.CANCELLED);
            return;
          }
        }
        if (stopReason == null) {
          stream.onComplete(new GenerationUsage(1, emitted));
          return;
        }
        reported = stopReason;
        stream.onComplete(new GenerationUsage(1, emitted), stopReason);
      } finally {
        finished.countDown();
      }
    }

    private static void awaitCancellation(TokenStream stream) {
      long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(5);
      while (!stream.isCancelled() && System.nanoTime() < deadline) {
        Thread.onSpinWait();
      }
    }
  }
}
