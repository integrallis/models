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

import com.integrallis.models.api.InferenceBackend;
import com.integrallis.models.api.LogitBatch;
import com.integrallis.models.api.ModelMetadata;
import com.integrallis.models.api.SamplingOptions;
import com.integrallis.models.api.SpeculativeInferenceBackend;
import com.integrallis.models.api.Tokenizer;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class SpeculativeGenerationLoopTest {

  @Test
  void acceptsTheFullDraftAndUsesTheFinalVerifierRowForEos() {
    int[] promptTokens = {2, 3, 4, 5, 2, 3, 4};
    SequenceBackend backend = new SequenceBackend(promptTokens, new int[] {5, 2, 3, 4, 1});
    GenerationLoop loop = new GenerationLoop(backend, threeTokenDraftOptions());

    String result =
        loop.generate(
            "prompt",
            SamplingOptions.builder()
                .temperature(0.0f)
                .repetitionPenalty(1.0f)
                .maxTokens(8)
                .build());

    assertThat(result).isEqualTo("[5][2][3][4]");
    assertThat(backend.verifiedBatches).containsExactly(new int[] {5, 2, 3, 4});
    assertThat(backend.rewindCheckpoints).isEmpty();
    assertThat(backend.forwardTokens).isEmpty();
    assertThat(loop.lastSpeculativeMetrics().acceptedTokens()).isEqualTo(3);
    assertThat(loop.lastSpeculativeMetrics().acceptanceRate()).isEqualTo(1.0);
    assertThat(loop.lastSpeculativeMetrics().proposedByPosition()).containsExactly(1, 1, 1);
    assertThat(loop.lastSpeculativeMetrics().acceptedByPosition()).containsExactly(1, 1, 1);
    assertThat(loop.lastSpeculativeMetrics().verificationBatchHistogram())
        .containsExactly(0, 0, 0, 0, 1);
  }

  @Test
  void rejectsDraftTailWithoutLeakingTokensOrResamplingTheMismatch() {
    int[] promptTokens = {2, 3, 4, 5, 2, 3, 4};
    SequenceBackend backend = new SequenceBackend(promptTokens, new int[] {5, 2, 9, 7, 1});
    GenerationLoop loop = new GenerationLoop(backend, threeTokenDraftOptions());

    String result =
        loop.generate(
            "prompt",
            SamplingOptions.builder()
                .temperature(0.0f)
                .repetitionPenalty(1.0f)
                .maxTokens(8)
                .build());

    assertThat(result).isEqualTo("[5][2][9][7]");
    assertThat(backend.verifiedBatches).containsExactly(new int[] {5, 2, 3, 4});
    assertThat(backend.rewindCheckpoints).containsExactly(promptTokens.length + 2);
    assertThat(backend.forwardTokens).containsExactly(9, 7);
    assertThat(loop.lastSpeculativeMetrics())
        .extracting(
            SpeculativeGenerationMetrics::draftAttempts,
            SpeculativeGenerationMetrics::proposedTokens,
            SpeculativeGenerationMetrics::acceptedTokens,
            SpeculativeGenerationMetrics::rollbacks,
            SpeculativeGenerationMetrics::ordinaryForwardCalls)
        .containsExactly(1, 3, 1, 1, 2);
    assertThat(loop.lastSpeculativeMetrics().acceptedByPosition()).containsExactly(1, 0, 0);
  }

  @Test
  void seededSamplingMatchesSequentialGenerationAfterDraftRejection() {
    int[] promptTokens = {2, 2, 4, 2, 3, 5, 2};
    UniformBackend speculativeBackend = new UniformBackend(promptTokens);
    UniformBackend sequentialDelegate = new UniformBackend(promptTokens);
    GenerationLoop speculativeLoop =
        new GenerationLoop(
            speculativeBackend,
            SpeculativeGenerationOptions.builder()
                .ngramSize(2)
                .minimumDraftTokens(1)
                .maximumDraftTokens(1)
                .adaptationWindow(100)
                .build());
    GenerationLoop sequentialLoop = new GenerationLoop(sequentialView(sequentialDelegate));
    SamplingOptions sampling =
        SamplingOptions.builder()
            .temperature(1.0f)
            .topK(2)
            .topP(1.0f)
            .seed(123L)
            .repetitionPenalty(1.0f)
            .maxTokens(12)
            .build();

    String speculative = speculativeLoop.generate("prompt", sampling);
    String sequential = sequentialLoop.generate("prompt", sampling);

    assertThat(speculative).isEqualTo(sequential);
    assertThat(speculativeLoop.lastSpeculativeMetrics().draftAttempts()).isPositive();
  }

  @Test
  void doesNotForwardTheFinalTokenWhenNoDraftIsAvailableAtTheLimit() {
    int[] promptTokens = {2, 3};
    SequenceBackend backend = new SequenceBackend(promptTokens, new int[] {5, 5, 5, 1});
    GenerationLoop loop = new GenerationLoop(backend, threeTokenDraftOptions());

    String result =
        loop.generate(
            "prompt",
            SamplingOptions.builder()
                .temperature(0.0f)
                .repetitionPenalty(1.0f)
                .maxTokens(3)
                .build());

    assertThat(result).isEqualTo("[5][5][5]");
    assertThat(backend.verifiedBatches).isEmpty();
    assertThat(backend.forwardTokens).containsExactly(5, 5);
  }

  private static SpeculativeGenerationOptions threeTokenDraftOptions() {
    return SpeculativeGenerationOptions.builder()
        .ngramSize(4)
        .confidenceProbeTokens(3)
        .minimumDraftTokens(3)
        .maximumDraftTokens(3)
        .build();
  }

  private static InferenceBackend sequentialView(UniformBackend delegate) {
    return new InferenceBackend() {
      @Override
      public String name() {
        return delegate.name();
      }

      @Override
      public ModelMetadata metadata() {
        return delegate.metadata();
      }

      @Override
      public Tokenizer tokenizer() {
        return delegate.tokenizer();
      }

      @Override
      public float[] prefill(int[] tokens, int startPosition) {
        return delegate.prefill(tokens, startPosition);
      }

      @Override
      public float[] forward(int token, int position) {
        return delegate.forward(token, position);
      }

      @Override
      public void reset() {
        delegate.reset();
      }

      @Override
      public void close() {}
    };
  }

  private static final class UniformBackend implements SpeculativeInferenceBackend {
    private static final int VOCABULARY_SIZE = 6;

    private final int[] promptTokens;
    private final Tokenizer tokenizer;
    private int nextPosition;

    private UniformBackend(int[] promptTokens) {
      this.promptTokens = promptTokens.clone();
      this.tokenizer =
          new Tokenizer() {
            @Override
            public int[] encode(String text) {
              return UniformBackend.this.promptTokens.clone();
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
              return "[" + token + "]";
            }

            @Override
            public int vocabSize() {
              return VOCABULARY_SIZE;
            }

            @Override
            public int bosToken() {
              return 0;
            }

            @Override
            public int eosToken() {
              return 1;
            }
          };
    }

    @Override
    public String name() {
      return "uniform-stub";
    }

    @Override
    public ModelMetadata metadata() {
      return new ModelMetadata("uniform-stub", "UniformStub", 128, VOCABULARY_SIZE, 16, 1, 1, 1);
    }

    @Override
    public Tokenizer tokenizer() {
      return tokenizer;
    }

    @Override
    public float[] prefill(int[] tokens, int startPosition) {
      nextPosition = startPosition + tokens.length;
      return uniformLogits();
    }

    @Override
    public float[] forward(int token, int position) {
      assertThat(position).isEqualTo(nextPosition);
      nextPosition++;
      return uniformLogits();
    }

    @Override
    public int checkpoint() {
      return nextPosition;
    }

    @Override
    public LogitBatch verify(int[] tokens, int startPosition) {
      assertThat(startPosition).isEqualTo(nextPosition);
      float[] rows = new float[tokens.length * VOCABULARY_SIZE];
      for (int row = 0; row < tokens.length; row++) {
        rows[row * VOCABULARY_SIZE + 2] = 1.0f;
        rows[row * VOCABULARY_SIZE + 3] = 1.0f;
      }
      nextPosition += tokens.length;
      return new LogitBatch(tokens.length, VOCABULARY_SIZE, rows);
    }

    @Override
    public void rewind(int checkpoint) {
      nextPosition = checkpoint;
    }

    @Override
    public void reset() {
      nextPosition = 0;
    }

    @Override
    public void close() {}

    private static float[] uniformLogits() {
      float[] logits = new float[VOCABULARY_SIZE];
      logits[2] = 1.0f;
      logits[3] = 1.0f;
      return logits;
    }
  }

  private static final class SequenceBackend implements SpeculativeInferenceBackend {
    private static final int VOCABULARY_SIZE = 12;

    private final int[] promptTokens;
    private final int[] generatedTokens;
    private final Tokenizer tokenizer;
    private final List<int[]> verifiedBatches = new ArrayList<>();
    private final List<Integer> rewindCheckpoints = new ArrayList<>();
    private final List<Integer> forwardTokens = new ArrayList<>();
    private int nextPosition;

    private SequenceBackend(int[] promptTokens, int[] generatedTokens) {
      this.promptTokens = promptTokens.clone();
      this.generatedTokens = generatedTokens.clone();
      this.tokenizer =
          new Tokenizer() {
            @Override
            public int[] encode(String text) {
              return SequenceBackend.this.promptTokens.clone();
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
              return "[" + token + "]";
            }

            @Override
            public int vocabSize() {
              return VOCABULARY_SIZE;
            }

            @Override
            public int bosToken() {
              return 0;
            }

            @Override
            public int eosToken() {
              return 1;
            }
          };
    }

    @Override
    public String name() {
      return "sequence-stub";
    }

    @Override
    public ModelMetadata metadata() {
      return new ModelMetadata("sequence-stub", "SequenceStub", 128, VOCABULARY_SIZE, 16, 1, 1, 1);
    }

    @Override
    public Tokenizer tokenizer() {
      return tokenizer;
    }

    @Override
    public float[] prefill(int[] tokens, int startPosition) {
      assertThat(tokens).containsExactly(promptTokens);
      nextPosition = startPosition + tokens.length;
      return logitsForGeneratedIndex(0);
    }

    @Override
    public float[] forward(int token, int position) {
      assertThat(position).isEqualTo(nextPosition);
      forwardTokens.add(token);
      nextPosition++;
      return logitsForGeneratedIndex(nextPosition - promptTokens.length);
    }

    @Override
    public int checkpoint() {
      return nextPosition;
    }

    @Override
    public LogitBatch verify(int[] tokens, int startPosition) {
      assertThat(startPosition).isEqualTo(nextPosition);
      verifiedBatches.add(tokens.clone());
      float[] logits = new float[tokens.length * VOCABULARY_SIZE];
      for (int row = 0; row < tokens.length; row++) {
        int generatedIndex = startPosition - promptTokens.length + row + 1;
        int predicted =
            generatedIndex < generatedTokens.length ? generatedTokens[generatedIndex] : 1;
        logits[row * VOCABULARY_SIZE + predicted] = 100.0f;
      }
      nextPosition += tokens.length;
      return new LogitBatch(tokens.length, VOCABULARY_SIZE, logits);
    }

    @Override
    public void rewind(int checkpoint) {
      rewindCheckpoints.add(checkpoint);
      nextPosition = checkpoint;
    }

    @Override
    public void reset() {
      nextPosition = 0;
      verifiedBatches.clear();
      rewindCheckpoints.clear();
      forwardTokens.clear();
    }

    @Override
    public void close() {}

    private float[] logitsForGeneratedIndex(int generatedIndex) {
      float[] logits = new float[VOCABULARY_SIZE];
      int predicted = generatedIndex < generatedTokens.length ? generatedTokens[generatedIndex] : 1;
      logits[predicted] = 100.0f;
      return logits;
    }
  }
}
