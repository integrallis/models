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
package com.integrallis.models.runtime;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.models.api.BatchInferenceBackend;
import com.integrallis.models.api.GenerationUsage;
import com.integrallis.models.api.InferenceBackend;
import com.integrallis.models.api.InferenceSession;
import com.integrallis.models.api.LogitBatch;
import com.integrallis.models.api.ModelMetadata;
import com.integrallis.models.api.RepetitionLoopDetection;
import com.integrallis.models.api.SamplingOptions;
import com.integrallis.models.api.StopReason;
import com.integrallis.models.api.TokenStream;
import com.integrallis.models.api.Tokenizer;
import java.time.Duration;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/** Every normal generation end carries a typed stop reason, on every generation path. */
@Tag("unit")
class GenerationStopReasonTest {

  private static final int VOCABULARY = 8;
  private static final int EOS = 1;
  private static final int PROMPT = 2;
  private static final int A = 3;
  private static final int B = 4;
  private static final int C = 5;
  private static final int END_OF_TURN = 6;

  private static final Tokenizer TOKENIZER =
      new Tokenizer() {
        @Override
        public int[] encode(String text) {
          return new int[] {PROMPT};
        }

        @Override
        public String decode(int[] tokens) {
          StringBuilder decoded = new StringBuilder();
          for (int token : tokens) {
            decoded.append(decode(token));
          }
          return decoded.toString();
        }

        @Override
        public String decode(int token) {
          return switch (token) {
            case A -> "a";
            case B -> "b";
            case C -> "c";
            default -> "";
          };
        }

        @Override
        public int vocabSize() {
          return VOCABULARY;
        }

        @Override
        public int bosToken() {
          return 0;
        }

        @Override
        public int eosToken() {
          return EOS;
        }

        @Override
        public boolean isEndOfGeneration(int token) {
          return token == EOS || token == END_OF_TURN;
        }
      };

  private static SamplingOptions.Builder greedy(int maxTokens) {
    return SamplingOptions.builder().temperature(0.0f).maxTokens(maxTokens);
  }

  @Nested
  class SequentialLoop {

    @Test
    void endOfSequenceTokenReportsEos() {
      Outcome outcome = generate(new int[] {A, B, EOS}, greedy(10).build());

      assertThat(outcome.text()).isEqualTo("ab");
      assertThat(outcome.reason()).isEqualTo(StopReason.EOS);
      assertThat(outcome.metricsReason()).contains(StopReason.EOS);
    }

    @Test
    void anyEndOfGenerationTokenReportsEos() {
      Outcome outcome = generate(new int[] {A, END_OF_TURN, B}, greedy(10).build());

      assertThat(outcome.text()).isEqualTo("a");
      assertThat(outcome.reason()).isEqualTo(StopReason.EOS);
    }

    @Test
    void tokenLimitReportsMaxTokens() {
      Outcome outcome = generate(new int[] {A, A, A, A}, greedy(2).build());

      assertThat(outcome.text()).isEqualTo("aa");
      assertThat(outcome.reason()).isEqualTo(StopReason.MAX_TOKENS);
      assertThat(outcome.metricsReason()).contains(StopReason.MAX_TOKENS);
    }

    @Test
    void stopSequenceReportsStopSequence() {
      Outcome outcome = generate(new int[] {A, B, C}, greedy(10).stopSequence("b").build());

      assertThat(outcome.text()).isEqualTo("a");
      assertThat(outcome.reason()).isEqualTo(StopReason.STOP_SEQUENCE);
    }

    @Test
    void stopSequenceOnTheFinalAllowedTokenIsNotReportedAsTruncation() {
      Outcome outcome = generate(new int[] {A, B, C}, greedy(2).stopSequence("b").build());

      assertThat(outcome.reason()).isEqualTo(StopReason.STOP_SEQUENCE);
    }

    @Test
    void completedConstraintReportsConstraintComplete() {
      GenerationLoop loop = new GenerationLoop(new ScriptedBackend(new int[] {C, C, C}));
      Recorder recorder = new Recorder();

      loop.generate("p", greedy(10).build(), recorder, TokenSequenceConstraint.of(A, B));

      assertThat(recorder.text).hasToString("ab");
      assertThat(recorder.reason).isEqualTo(StopReason.CONSTRAINT_COMPLETE);
    }

    @Test
    void cancellationStopsAfterTheCurrentTokenAndReportsCancelled() {
      ScriptedBackend backend = new ScriptedBackend(new int[] {A, B, C, A, B, C});
      GenerationLoop loop = new GenerationLoop(backend);
      Recorder recorder = new Recorder();
      recorder.cancelAfterTokens = 2;

      loop.generate("p", greedy(10).build(), recorder);

      assertThat(recorder.text).hasToString("ab");
      assertThat(recorder.reason).isEqualTo(StopReason.CANCELLED);
      assertThat(recorder.usage.completionTokens()).isEqualTo(2);
      assertThat(backend.forwardCalls).hasValue(1);
      assertThat(loop.lastGenerationMetrics().stopReason()).contains(StopReason.CANCELLED);
    }

    @Test
    void repetitionLoopStopsGenerationAndIncrementsTheCounter() {
      int[] loop = new int[40];
      for (int index = 0; index < loop.length; index++) {
        loop[index] = index % 2 == 0 ? A : B;
      }
      GenerationLoop generationLoop = new GenerationLoop(new ScriptedBackend(loop));
      SamplingOptions options =
          greedy(50).repetitionLoopDetection(new RepetitionLoopDetection(8, 3, 0)).build();
      Recorder first = new Recorder();
      Recorder second = new Recorder();

      assertThat(generationLoop.repetitionLoopStops()).isZero();
      generationLoop.generate("p", options, first);
      generationLoop.generate("p", options, second);

      assertThat(first.text).hasToString("ababab");
      assertThat(first.reason).isEqualTo(StopReason.REPETITION_LOOP);
      assertThat(first.usage.completionTokens()).isEqualTo(6);
      assertThat(second.reason).isEqualTo(StopReason.REPETITION_LOOP);
      assertThat(generationLoop.lastGenerationMetrics().stopReason())
          .contains(StopReason.REPETITION_LOOP);
      assertThat(generationLoop.repetitionLoopStops()).isEqualTo(2);
    }

    @Test
    void repetitionLoopDetectionIsOffByDefault() {
      int[] loop = new int[40];
      for (int index = 0; index < loop.length; index++) {
        loop[index] = index % 2 == 0 ? A : B;
      }
      GenerationLoop generationLoop = new GenerationLoop(new ScriptedBackend(loop));
      Recorder recorder = new Recorder();

      generationLoop.generate("p", greedy(30).build(), recorder);

      assertThat(recorder.text).hasToString("ab".repeat(15));
      assertThat(recorder.reason).isEqualTo(StopReason.MAX_TOKENS);
      assertThat(generationLoop.repetitionLoopStops()).isZero();
    }

    @Test
    void runtimeModelExposesTheRepetitionLoopCounter() {
      RuntimeTextGenerationModel model =
          new RuntimeTextGenerationModel(new ScriptedBackend(new int[] {C, C, C, C, C, C}));

      String output =
          model.generate(
              "p",
              greedy(20).repetitionLoopDetection(new RepetitionLoopDetection(4, 4, 0)).build());

      assertThat(output).isEqualTo("cccc");
      assertThat(model.repetitionLoopStops()).isEqualTo(1);
      assertThat(model.lastGenerationMetrics().stopReason()).contains(StopReason.REPETITION_LOOP);
    }

    @Test
    void failedGenerationHasNoStopReason() {
      GenerationLoop loop = new GenerationLoop(new ScriptedBackend(new int[] {A, B, EOS}));
      Recorder recorder = new Recorder();

      loop.generate(
          "p",
          greedy(10).build(),
          recorder,
          new TokenConstraint() {
            @Override
            public boolean allows(int token) {
              return false;
            }

            @Override
            public void accept(int token) {}
          });

      assertThat(recorder.failure).isNotNull();
      assertThat(recorder.reason).isNull();
      assertThat(loop.lastGenerationMetrics().successful()).isFalse();
      assertThat(loop.lastGenerationMetrics().stopReason()).isEmpty();
    }

    @Test
    void consumersThatOnlyKnowUsageStillReceiveIt() {
      GenerationLoop loop = new GenerationLoop(new ScriptedBackend(new int[] {A, EOS}));
      List<GenerationUsage> usages = new ArrayList<>();

      loop.generate(
          "p",
          greedy(10).build(),
          new TokenStream() {
            @Override
            public void onToken(String token) {}

            @Override
            public void onComplete() {}

            @Override
            public void onComplete(GenerationUsage usage) {
              usages.add(usage);
            }

            @Override
            public void onError(Throwable failure) {
              throw new AssertionError(failure);
            }
          });

      assertThat(usages).containsExactly(new GenerationUsage(1, 1));
    }

    private Outcome generate(int[] script, SamplingOptions options) {
      GenerationLoop loop = new GenerationLoop(new ScriptedBackend(script));
      Recorder recorder = new Recorder();
      loop.generate("p", options, recorder);
      if (recorder.failure != null) {
        throw new AssertionError(recorder.failure);
      }
      return new Outcome(
          recorder.text.toString(), recorder.reason, loop.lastGenerationMetrics().stopReason());
    }
  }

  @Nested
  class ContinuousBatching {

    @Test
    void reportsEachStopReasonThroughTheScheduler() {
      assertThat(generate(new int[] {A, B, EOS}, greedy(10).build(), 0))
          .isEqualTo(new Outcome("ab", StopReason.EOS, Optional.of(StopReason.EOS)));
      assertThat(generate(new int[] {A, A, A}, greedy(2).build(), 0))
          .isEqualTo(new Outcome("aa", StopReason.MAX_TOKENS, Optional.of(StopReason.MAX_TOKENS)));
      assertThat(generate(new int[] {A, B, C}, greedy(10).stopSequence("b").build(), 0))
          .isEqualTo(
              new Outcome("a", StopReason.STOP_SEQUENCE, Optional.of(StopReason.STOP_SEQUENCE)));
      assertThat(generate(new int[] {A, B, C, A, B, C}, greedy(10).build(), 2))
          .isEqualTo(new Outcome("ab", StopReason.CANCELLED, Optional.of(StopReason.CANCELLED)));
    }

    @Test
    void repetitionLoopStopsABatchedSessionAndIsCounted() {
      ScriptedBatchBackend backend = new ScriptedBatchBackend(new int[] {A, B, C, C, C, C, C, C});
      ContinuousBatchingOptions batching =
          ContinuousBatchingOptions.builder()
              .maximumBatchSize(1)
              .batchFormationDelay(Duration.ZERO)
              .build();
      try (InferencePipeline pipeline = new InferencePipeline(backend, batching);
          TextGenerationSession session = pipeline.openGenerationSession()) {
        Recorder recorder = new Recorder();
        session.generate(
            com.integrallis.models.api.ModelPrompt.text("p"),
            greedy(20).repetitionLoopDetection(new RepetitionLoopDetection(4, 3, 0)).build(),
            recorder);

        assertThat(recorder.text).hasToString("abccc");
        assertThat(recorder.reason).isEqualTo(StopReason.REPETITION_LOOP);
        assertThat(session.lastGenerationMetrics().stopReason())
            .contains(StopReason.REPETITION_LOOP);
        assertThat(pipeline.continuousBatchingMetrics().orElseThrow().repetitionLoopStops())
            .isEqualTo(1);
      }
    }

    private Outcome generate(int[] script, SamplingOptions options, int cancelAfterTokens) {
      ScriptedBatchBackend backend = new ScriptedBatchBackend(script);
      ContinuousBatchingOptions batching =
          ContinuousBatchingOptions.builder()
              .maximumBatchSize(1)
              .batchFormationDelay(Duration.ZERO)
              .build();
      try (InferencePipeline pipeline = new InferencePipeline(backend, batching);
          TextGenerationSession session = pipeline.openGenerationSession()) {
        Recorder recorder = new Recorder();
        recorder.cancelAfterTokens = cancelAfterTokens;
        session.generate(com.integrallis.models.api.ModelPrompt.text("p"), options, recorder);
        if (recorder.failure != null) {
          throw new AssertionError(recorder.failure);
        }
        return new Outcome(
            recorder.text.toString(),
            recorder.reason,
            session.lastGenerationMetrics().stopReason());
      }
    }
  }

  record Outcome(String text, StopReason reason, Optional<StopReason> metricsReason) {}

  static final class Recorder implements TokenStream {
    final StringBuilder text = new StringBuilder();
    int tokens;
    int cancelAfterTokens;
    StopReason reason;
    GenerationUsage usage;
    Throwable failure;

    @Override
    public void onToken(String token) {
      text.append(token);
      tokens++;
    }

    @Override
    public void onComplete() {}

    @Override
    public void onComplete(GenerationUsage usage, StopReason stopReason) {
      this.usage = usage;
      this.reason = stopReason;
    }

    @Override
    public boolean isCancelled() {
      return cancelAfterTokens > 0 && tokens >= cancelAfterTokens;
    }

    @Override
    public void onError(Throwable failure) {
      this.failure = failure;
    }
  }

  /** Emits {@code script[i]} as the i-th generated token, then EOS. */
  static final class ScriptedBackend implements InferenceBackend {
    private final int[] script;
    final AtomicInteger forwardCalls = new AtomicInteger();
    private int generated;

    ScriptedBackend(int[] script) {
      this.script = script.clone();
    }

    @Override
    public String name() {
      return "scripted";
    }

    @Override
    public ModelMetadata metadata() {
      return new ModelMetadata("scripted", "Scripted", 4_096, VOCABULARY, 8, 1, 1, 1);
    }

    @Override
    public Tokenizer tokenizer() {
      return TOKENIZER;
    }

    @Override
    public float[] prefill(int[] tokens, int startPosition) {
      generated = 0;
      return logitsFor(0);
    }

    @Override
    public float[] forward(int token, int position) {
      forwardCalls.incrementAndGet();
      return logitsFor(++generated);
    }

    @Override
    public void reset() {
      generated = 0;
    }

    @Override
    public void close() {}

    private float[] logitsFor(int index) {
      return oneHot(index < script.length ? script[index] : EOS);
    }
  }

  /** Session-aware variant of {@link ScriptedBackend} for the continuous-batching scheduler. */
  static final class ScriptedBatchBackend implements BatchInferenceBackend {
    private final int[] script;
    private final Map<InferenceSession, Integer> positions = new HashMap<>();

    ScriptedBatchBackend(int[] script) {
      this.script = script.clone();
    }

    @Override
    public String name() {
      return "scripted-batch";
    }

    @Override
    public ModelMetadata metadata() {
      return new ModelMetadata("scripted", "ScriptedBatch", 4_096, VOCABULARY, 8, 1, 1, 1);
    }

    @Override
    public Tokenizer tokenizer() {
      return TOKENIZER;
    }

    @Override
    public float[] forward(int token, int position) {
      throw new AssertionError("default state must not be used");
    }

    @Override
    public int maxBatchSize() {
      return 1;
    }

    @Override
    public InferenceSession openSession() {
      return new InferenceSession() {
        private boolean closed;

        @Override
        public int checkpoint() {
          return positions.getOrDefault(this, 0);
        }

        @Override
        public boolean isClosed() {
          return closed;
        }

        @Override
        public void close() {
          closed = true;
        }
      };
    }

    @Override
    public synchronized float[] forward(InferenceSession session, int token, int position) {
      positions.put(session, position + 1);
      // The prompt is one token, so position p produces generated index p.
      return oneHot(position < script.length ? script[position] : EOS);
    }

    @Override
    public LogitBatch forwardBatch(InferenceSession[] sessions, int[] tokens) {
      float[] values = new float[sessions.length * VOCABULARY];
      for (int row = 0; row < sessions.length; row++) {
        float[] logits = forward(sessions[row], tokens[row], sessions[row].checkpoint());
        System.arraycopy(logits, 0, values, row * VOCABULARY, VOCABULARY);
      }
      return new LogitBatch(sessions.length, VOCABULARY, values);
    }

    @Override
    public synchronized void rewind(InferenceSession session, int checkpoint) {
      positions.put(session, checkpoint);
    }

    @Override
    public synchronized void reset(InferenceSession session) {
      positions.put(session, 0);
    }

    @Override
    public void close() {}
  }

  private static float[] oneHot(int token) {
    float[] logits = new float[VOCABULARY];
    logits[token] = 100.0f;
    return logits;
  }
}
