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
import static org.assertj.core.api.Assertions.assertThatIllegalStateException;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.models.api.AuxiliaryInferenceBackend;
import com.integrallis.models.api.AuxiliaryTextGenerationModel;
import com.integrallis.models.api.BackendDiagnostics;
import com.integrallis.models.api.BatchInferenceBackend;
import com.integrallis.models.api.InferenceSession;
import com.integrallis.models.api.LogitBatch;
import com.integrallis.models.api.ModelMetadata;
import com.integrallis.models.api.ModelPrompt;
import com.integrallis.models.api.RewindableInferenceBackend;
import com.integrallis.models.api.SamplingOptions;
import com.integrallis.models.api.TokenStream;
import com.integrallis.models.api.Tokenizer;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

class InferencePipelineTest {

  @Test
  void exposesTheMostRecentRuntimeGenerationMetrics() {
    try (InferencePipeline pipeline = new InferencePipeline(new StubBackend())) {
      pipeline.generate("hello", SamplingOptions.builder().temperature(0).maxTokens(1).build());

      assertThat(pipeline.lastGenerationMetrics().available()).isTrue();
      assertThat(pipeline.lastGenerationMetrics().usage().promptTokens()).isPositive();
    }
  }

  @Test
  void exposesTokenizerMetadataAndManagedContextOperations() {
    StubBackend backend = new StubBackend();

    try (InferencePipeline pipeline = new InferencePipeline(backend)) {
      ModelPrompt prompt = ModelPrompt.builder().control("<control>").text("hello").build();

      assertThat(pipeline.metadata().modelName()).isEqualTo("fixture");
      assertThat(pipeline.tokenizer()).isSameAs(backend.tokenizer());
      assertThat(pipeline.contextWindow().capacity()).isEqualTo(8);
      assertThat(pipeline.contextWindow().position()).hasValue(0);
      assertThat(pipeline.tokenize(prompt)).containsExactly(0, 1);
      assertThat(pipeline.prefill(prompt, 0)).containsExactly(0.0f, 0.0f, 10.0f);
      assertThat(pipeline.contextWindow().position()).hasValue(2);
      assertThat(pipeline.contextWindow().remaining()).hasValue(6);

      pipeline.forward(0, 2);
      assertThat(pipeline.checkpoint()).isEqualTo(3);
      pipeline.rewind(1);
      assertThat(pipeline.contextWindow().position()).hasValue(1);
      pipeline.resetContext();
      assertThat(pipeline.contextWindow().position()).hasValue(0);
    }

    assertThat(backend.closeCount).isEqualTo(1);
  }

  @Test
  void preservesStructuredPromptTokenizationDuringGeneration() {
    StubBackend backend = new StubBackend();
    ModelPrompt prompt = ModelPrompt.builder().control("<control>").text("hello").build();

    try (InferencePipeline pipeline = new InferencePipeline(backend)) {
      String generated = pipeline.generate(prompt, deterministicOptions());

      assertThat(generated).isEmpty();
      assertThat(backend.structuredEncodes).isEqualTo(1);
      assertThat(backend.plainEncodes).isZero();
    }
  }

  @Test
  void directContextAccessInvalidatesTheHighLevelPrefixCache() {
    StubBackend backend = new StubBackend();

    try (InferencePipeline pipeline = new InferencePipeline(backend)) {
      pipeline.generate(ModelPrompt.text("hello"), deterministicOptions());
      pipeline.generate(ModelPrompt.text("hello"), deterministicOptions());
      pipeline.resetContext();
      pipeline.generate(ModelPrompt.text("hello"), deterministicOptions());
    }

    assertThat(backend.prefillStartPositions).containsExactly(0, 1, 0);
  }

  @Test
  void exposesConstraintsAndAuxiliaryHeadsWithoutUnwrappingThePipeline() {
    StubBackend backend = new StubBackend();

    try (InferencePipeline pipeline = new InferencePipeline(backend)) {
      assertThat(pipeline).isInstanceOf(ConstrainedTextGenerationModel.class);
      assertThat(pipeline).isInstanceOf(AuxiliaryTextGenerationModel.class);

      AuxiliaryTextGenerationModel auxiliary = (AuxiliaryTextGenerationModel) pipeline;
      assertThat(auxiliary.supportsContrastiveEncoding()).isTrue();
      assertThat(auxiliary.contrastiveDimension()).isEqualTo(2);
      assertThat(auxiliary.encodeContrastive(ModelPrompt.text("tools")))
          .containsExactly(1.0f, 2.0f);
      assertThat(auxiliary.supportsConfidenceScoring()).isTrue();
      assertThat(auxiliary.scoreConfidence(ModelPrompt.text("prompt and output"))).isEqualTo(0.75f);
    }
  }

  @Test
  void closesTheOwnedBackendExactlyOnceAndRejectsFurtherUse() {
    StubBackend backend = new StubBackend();
    InferencePipeline pipeline = new InferencePipeline(backend);

    pipeline.close();
    pipeline.close();

    assertThat(backend.closeCount).isEqualTo(1);
    assertThatIllegalStateException()
        .isThrownBy(pipeline::metadata)
        .withMessageContaining("closed");
  }

  @Test
  void isolatesPromptPrefixStateAcrossExplicitGenerationSessions() {
    SessionBackend backend = new SessionBackend();
    InferencePipeline pipeline = new InferencePipeline(backend);
    TextGenerationSession first = pipeline.openGenerationSession();

    try (pipeline;
        TextGenerationSession second = pipeline.openGenerationSession()) {
      first.generate("ab", deterministicOptions());
      second.generate("xy", deterministicOptions());
      first.generate("abc", deterministicOptions());

      assertThat(first.lastGenerationMetrics().promptCache().cacheReadInputTokens()).isEqualTo(2);
      assertThat(first.lastGenerationMetrics().promptCache().cacheWriteInputTokens()).isEqualTo(1);
      assertThat(second.lastGenerationMetrics().promptCache().cacheReadInputTokens()).isZero();
      assertThat(backend.prefillStartPositions()).containsExactly(List.of(0, 2), List.of(0));

      first.close();
      first.close();
      assertThatIllegalStateException()
          .isThrownBy(() -> first.generate("abc", deterministicOptions()))
          .withMessageContaining("closed");
      assertThat(backend.closedSessions).isEqualTo(1);

      second.generate("xyz", deterministicOptions());
      assertThat(second.lastGenerationMetrics().promptCache().cacheReadInputTokens()).isEqualTo(2);
      assertThat(backend.closeCount).isZero();
    }

    assertThat(backend.closedSessions).isEqualTo(2);
    assertThat(backend.closeCount).isEqualTo(1);
  }

  @Test
  void closesOpenGenerationSessionsWithTheOwningPipeline() {
    SessionBackend backend = new SessionBackend();
    InferencePipeline pipeline = new InferencePipeline(backend);
    TextGenerationSession session = pipeline.openGenerationSession();

    pipeline.close();

    assertThat(session.isClosed()).isTrue();
    assertThat(backend.closedSessions).isEqualTo(1);
    assertThat(backend.closeCount).isEqualTo(1);
    assertThatIllegalStateException()
        .isThrownBy(session::contextWindow)
        .withMessageContaining("closed");
  }

  @Test
  void explainsWhenTheBackendCannotOpenIndependentGenerationState() {
    try (InferencePipeline pipeline = new InferencePipeline(new StubBackend())) {
      assertThat(pipeline.supportsGenerationSessions()).isFalse();
      assertThatThrownBy(pipeline::openGenerationSession)
          .isInstanceOf(UnsupportedOperationException.class)
          .hasMessageContaining("independent inference sessions");
    }
  }

  @Test
  void resetsOneGenerationSessionWithoutChangingAnother() {
    SessionBackend backend = new SessionBackend();

    try (InferencePipeline pipeline = new InferencePipeline(backend);
        TextGenerationSession first = pipeline.openGenerationSession();
        TextGenerationSession second = pipeline.openGenerationSession()) {
      first.generate("ab", deterministicOptions());
      second.generate("xy", deterministicOptions());

      first.resetContext();
      first.generate("abc", deterministicOptions());
      second.generate("xyz", deterministicOptions());

      assertThat(backend.prefillStartPositions()).containsExactly(List.of(0, 0), List.of(0, 2));
    }
  }

  @Test
  void exposesTheCompleteHighLevelGenerationSessionContract() {
    SessionBackend backend = new SessionBackend();

    try (InferencePipeline pipeline = new InferencePipeline(backend);
        TextGenerationSession session = pipeline.openGenerationSession()) {
      TokenStream stream =
          new TokenStream() {
            @Override
            public void onToken(String token) {}

            @Override
            public void onComplete() {}

            @Override
            public void onError(Throwable failure) {
              throw new AssertionError(failure);
            }
          };

      assertThat(session.modelName()).isEqualTo("session-fixture");
      assertThat(session.diagnostics().backend()).isEqualTo("session-stub");
      assertThat(session.tokenizer().vocabSize()).isEqualTo(8);
      assertThat(session.contextWindow().capacity()).isEqualTo(16);
      assertThat(session.contextWindow().position()).hasValue(0);

      assertThat(session.generate(ModelPrompt.text("ab"), deterministicOptions())).isEmpty();
      session.generate("ab", deterministicOptions(), stream);
      session.generate(ModelPrompt.text("ab"), deterministicOptions(), stream);
      session.generate(
          ModelPrompt.text("ab"), deterministicOptions(), stream, TokenConstraint.unrestricted());
    }
  }

  private static SamplingOptions deterministicOptions() {
    return SamplingOptions.builder().temperature(0.0f).maxTokens(1).build();
  }

  private static final class StubBackend
      implements RewindableInferenceBackend, AuxiliaryInferenceBackend {
    private final List<Integer> prefillStartPositions = new ArrayList<>();
    private int position;
    private int plainEncodes;
    private int structuredEncodes;
    private int closeCount;

    private final Tokenizer tokenizer =
        new Tokenizer() {
          @Override
          public int[] encode(String text) {
            plainEncodes++;
            return new int[] {0, 1};
          }

          @Override
          public int[] encode(ModelPrompt prompt) {
            structuredEncodes++;
            return new int[] {0, 1};
          }

          @Override
          public String decode(int[] tokens) {
            return "";
          }

          @Override
          public String decode(int token) {
            return "";
          }

          @Override
          public int vocabSize() {
            return 3;
          }

          @Override
          public int bosToken() {
            return 0;
          }

          @Override
          public int eosToken() {
            return 2;
          }
        };

    @Override
    public String name() {
      return "stub";
    }

    @Override
    public ModelMetadata metadata() {
      return new ModelMetadata("fixture", "fixture", 16, 3, 2, 1, 1, 1);
    }

    @Override
    public int contextCapacity() {
      return 8;
    }

    @Override
    public BackendDiagnostics diagnostics() {
      return BackendDiagnostics.unavailable("stub");
    }

    @Override
    public Tokenizer tokenizer() {
      return tokenizer;
    }

    @Override
    public float[] forward(int token, int tokenPosition) {
      position = tokenPosition + 1;
      return terminalLogits();
    }

    @Override
    public float[] prefill(int[] tokens, int startPosition) {
      prefillStartPositions.add(startPosition);
      position = startPosition + tokens.length;
      return terminalLogits();
    }

    @Override
    public void reset() {
      position = 0;
    }

    @Override
    public int checkpoint() {
      return position;
    }

    @Override
    public void rewind(int checkpoint) {
      position = checkpoint;
    }

    @Override
    public void close() {
      closeCount++;
    }

    @Override
    public boolean supportsContrastiveEncoding() {
      return true;
    }

    @Override
    public int contrastiveDimension() {
      return 2;
    }

    @Override
    public float[] encodeContrastive(int[] tokens) {
      return new float[] {1.0f, 2.0f};
    }

    @Override
    public boolean supportsConfidenceScoring() {
      return true;
    }

    @Override
    public float scoreConfidence(int[] tokens) {
      return 0.75f;
    }

    private static float[] terminalLogits() {
      return new float[] {0.0f, 0.0f, 10.0f};
    }
  }

  private static final class SessionBackend implements BatchInferenceBackend {
    private final List<Session> sessions = new ArrayList<>();
    private int closedSessions;
    private int closeCount;

    @Override
    public String name() {
      return "session-stub";
    }

    @Override
    public ModelMetadata metadata() {
      return new ModelMetadata("fixture", "session-fixture", 16, 8, 8, 1, 1, 1);
    }

    @Override
    public Tokenizer tokenizer() {
      return new Tokenizer() {
        @Override
        public int[] encode(String text) {
          return text.chars().map(character -> 3 + Math.floorMod(character, 5)).toArray();
        }

        @Override
        public String decode(int[] tokens) {
          return "";
        }

        @Override
        public String decode(int token) {
          return "";
        }

        @Override
        public int vocabSize() {
          return 8;
        }

        @Override
        public int bosToken() {
          return 0;
        }

        @Override
        public int eosToken() {
          return 2;
        }
      };
    }

    @Override
    public float[] forward(int token, int position) {
      throw new AssertionError("default backend state must not serve an explicit session");
    }

    @Override
    public int maxBatchSize() {
      return 4;
    }

    @Override
    public InferenceSession openSession() {
      Session session = new Session();
      sessions.add(session);
      return session;
    }

    @Override
    public float[] forward(InferenceSession session, int token, int position) {
      Session state = requireSession(session);
      state.position = position + 1;
      return terminalLogits();
    }

    @Override
    public float[] prefill(InferenceSession session, int[] tokens, int startPosition) {
      Session state = requireSession(session);
      state.prefillStartPositions.add(startPosition);
      state.position = startPosition + tokens.length;
      return terminalLogits();
    }

    @Override
    public LogitBatch forwardBatch(InferenceSession[] sessions, int[] tokens) {
      throw new UnsupportedOperationException();
    }

    @Override
    public void rewind(InferenceSession session, int checkpoint) {
      requireSession(session).position = checkpoint;
    }

    @Override
    public void reset(InferenceSession session) {
      requireSession(session).position = 0;
    }

    @Override
    public void close() {
      closeCount++;
    }

    List<List<Integer>> prefillStartPositions() {
      return sessions.stream().map(session -> List.copyOf(session.prefillStartPositions)).toList();
    }

    private Session requireSession(InferenceSession session) {
      if (!(session instanceof Session state) || state.closed) {
        throw new IllegalStateException("session is closed or does not belong to this backend");
      }
      return state;
    }

    private static float[] terminalLogits() {
      float[] logits = new float[8];
      logits[2] = 10.0f;
      return logits;
    }

    private final class Session implements InferenceSession {
      private final List<Integer> prefillStartPositions = new ArrayList<>();
      private int position;
      private boolean closed;

      @Override
      public int checkpoint() {
        return position;
      }

      @Override
      public boolean isClosed() {
        return closed;
      }

      @Override
      public void close() {
        if (!closed) {
          closed = true;
          closedSessions++;
        }
      }
    }
  }
}
