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
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.models.api.ActivatedAdapterMetadata;
import com.integrallis.models.api.BackendDiagnostics;
import com.integrallis.models.api.InferenceSession;
import com.integrallis.models.api.LogitBatch;
import com.integrallis.models.api.ModelMetadata;
import com.integrallis.models.api.ModelPrompt;
import com.integrallis.models.api.SamplingOptions;
import com.integrallis.models.api.SharedInferencePrefix;
import com.integrallis.models.api.SharedPrefixInferenceBackend;
import com.integrallis.models.api.TokenStream;
import com.integrallis.models.api.Tokenizer;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

class ActivatedToolCallingModelTest {

  @Test
  void recomputesIndependentBranchesBelowTheMeasuredSharingCrossover() {
    ActivatedBackend backend = new ActivatedBackend();

    try (ActivatedToolCallingModel model = new ActivatedToolCallingModel(backend, 3);
        ActivatedToolTurn turn = model.openToolTurn(ModelPrompt.text("abXY"))) {
      assertThat(turn.sharedPrefixTokens()).isZero();
      assertThat(turn.sharedPrefixBytes()).isZero();
      assertThat(turn.physicallySharesPrefix()).isFalse();

      turn.generateToolCall(deterministicOptions(), TokenConstraint.unrestricted());
      turn.generateBaseResponse(ModelPrompt.text("abXYz"), deterministicOptions());
    }

    assertThat(backend.prefills)
        .containsExactly(
            new Prefill(false, 0, List.of((int) 'a', (int) 'b')),
            new Prefill(false, 0, List.of((int) 'a', (int) 'b')),
            new Prefill(true, 2, List.of((int) 'X', (int) 'Y')),
            new Prefill(false, 2, List.of((int) 'X', (int) 'Y', (int) 'z')));
  }

  @Test
  void physicallySharesAtTheMeasuredSharingCrossover() {
    ActivatedBackend backend = new ActivatedBackend();

    try (ActivatedToolCallingModel model = new ActivatedToolCallingModel(backend, 2);
        ActivatedToolTurn turn = model.openToolTurn(ModelPrompt.text("abXY"))) {
      assertThat(turn.sharedPrefixTokens()).isEqualTo(2);
      assertThat(turn.sharedPrefixBytes()).isEqualTo(64);
      assertThat(turn.physicallySharesPrefix()).isTrue();
    }

    assertThat(backend.prefills)
        .containsExactly(new Prefill(false, 0, List.of((int) 'a', (int) 'b')));
  }

  @Test
  void failedSharedTurnConstructionPreservesTheCauseAndClosesBothBranches() {
    ActivatedBackend backend = new ActivatedBackend(true, 2);
    ActivatedToolCallingModel model = new ActivatedToolCallingModel(backend, 1);

    assertThatThrownBy(() -> model.openToolTurn(ModelPrompt.text("abXY")))
        .isInstanceOf(IllegalStateException.class)
        .hasMessage("fixture physical-sharing failure")
        .satisfies(
            failure -> {
              assertThat(failure.getSuppressed())
                  .extracting(Throwable::getMessage)
                  .containsExactly(
                      "fixture session close failure 1", "fixture session close failure 2");
              assertThat(backend.sessionCloseAttempts).isEqualTo(2);
            });

    model.close();
  }

  @Test
  void failedConversationCloseReleasesItsTurnReference() throws Exception {
    ActivatedBackend backend = new ActivatedBackend(false, 2);
    ActivatedToolCallingModel model = new ActivatedToolCallingModel(backend, 1);
    ActivatedToolConversation conversation = model.openConversation();
    conversation.selectTool(
        ModelPrompt.text("abXY"), deterministicOptions(), TokenConstraint.unrestricted());

    assertThatThrownBy(conversation::close)
        .isInstanceOf(IllegalStateException.class)
        .hasMessage("fixture session close failure 1")
        .satisfies(
            failure ->
                assertThat(failure.getSuppressed())
                    .extracting(Throwable::getMessage)
                    .containsExactly("fixture session close failure 2"));

    var activeTurn = ActivatedToolConversation.class.getDeclaredField("activeTurn");
    activeTurn.setAccessible(true);
    assertThat(activeTurn.get(conversation)).isNull();
    assertThat(backend.sessionCloseAttempts).isEqualTo(2);
    model.close();
  }

  @Test
  void explicitSharingOverridesTheAutomaticCrossoverForBenchmarking() {
    ActivatedBackend backend = new ActivatedBackend();

    try (ActivatedToolCallingModel model = new ActivatedToolCallingModel(backend, 100);
        ActivatedToolTurn turn =
            model.openToolTurn(
                ModelPrompt.text("abXY"), ActivatedToolCallingModel.PrefixStrategy.SHARED)) {
      assertThat(turn.sharedPrefixTokens()).isEqualTo(2);
      assertThat(turn.physicallySharesPrefix()).isTrue();
    }
  }

  @Test
  void retainedConversationRecomputesOnlyTheToolBranchBelowTheCrossover() {
    ActivatedBackend backend = new ActivatedBackend();

    try (ActivatedToolCallingModel model = new ActivatedToolCallingModel(backend, 4);
        ActivatedToolConversation conversation = model.openConversation()) {
      conversation.generateBase(ModelPrompt.text("a"), deterministicOptions());
      conversation.generateBase(ModelPrompt.text("ab"), deterministicOptions());

      conversation.selectTool(
          ModelPrompt.text("abqXY"), deterministicOptions(), TokenConstraint.unrestricted());
      assertThat(conversation.sharedPrefixTokens()).isZero();
      assertThat(conversation.sharedPrefixBytes()).isZero();
      assertThat(conversation.physicallySharesPrefix()).isFalse();
      conversation.completeToolResult(ModelPrompt.text("abqXYz"), deterministicOptions());
    }

    assertThat(backend.prefills)
        .containsExactly(
            new Prefill(false, 0, List.of((int) 'a')),
            new Prefill(false, 1, List.of((int) 'b')),
            new Prefill(false, 2, List.of((int) 'q')),
            new Prefill(false, 0, List.of((int) 'a', (int) 'b', (int) 'q')),
            new Prefill(true, 3, List.of((int) 'X', (int) 'Y')),
            new Prefill(false, 3, List.of((int) 'X', (int) 'Y', (int) 'z')));
  }

  @Test
  void forksToolAndBaseTurnsFromOnePhysicalPrefixAndEvaluatesOnlyTheirSuffixes() {
    ActivatedBackend backend = new ActivatedBackend();

    try (ActivatedToolCallingModel model = new ActivatedToolCallingModel(backend, 1);
        ActivatedToolTurn turn = model.openToolTurn(ModelPrompt.text("abXY"))) {
      assertThat(turn.sharedPrefixTokens()).isEqualTo(2);
      assertThat(turn.sharedPrefixBytes()).isEqualTo(64);
      assertThat(turn.physicallySharesPrefix()).isTrue();
      assertThat(turn.baseInferenceStateBytes()).hasValue(96);
      assertThat(turn.toolInferenceStateBytes()).hasValue(96);
      assertThat(turn.uniqueInferenceStateBytes()).hasValue(128);

      turn.generateToolCall(deterministicOptions(), TokenConstraint.unrestricted());
      turn.generateBaseResponse(ModelPrompt.text("abXYz"), deterministicOptions());

      assertThat(turn.toolMetrics().promptCache().cacheReadInputTokens()).isEqualTo(2);
      assertThat(turn.responseMetrics().promptCache().cacheReadInputTokens()).isEqualTo(2);
      assertThat(backend.prefills)
          .containsExactly(
              new Prefill(false, 0, List.of((int) 'a', (int) 'b')),
              new Prefill(true, 2, List.of((int) 'X', (int) 'Y')),
              new Prefill(false, 2, List.of((int) 'X', (int) 'Y', (int) 'z')));
    }
  }

  @Test
  void scoresTheActivatedCallDecisionWithoutDiscardingThePhysicalPrefix() {
    ActivatedBackend backend = new ActivatedBackend();

    try (ActivatedToolCallingModel model = new ActivatedToolCallingModel(backend, 1);
        ActivatedToolTurn turn = model.openToolTurn(ModelPrompt.text("abXY"))) {
      ToolDecisionScore score = turn.scoreToolDecision(2, 3);

      assertThat(score.callTokenId()).isEqualTo(2);
      assertThat(score.callLogit()).isEqualTo(10.0f);
      assertThat(score.noCallTokenId()).isEqualTo(3);
      assertThat(score.noCallLogit()).isZero();
      assertThat(score.callMargin()).isEqualTo(10.0f);
      assertThat(score.shouldCall(9.99f)).isTrue();
      assertThat(score.shouldCall(10.0f)).isFalse();
      assertThat(turn.physicallySharesPrefix()).isTrue();

      turn.generateToolCall(deterministicOptions(), TokenConstraint.unrestricted());
    }

    assertThat(backend.prefills)
        .containsExactly(
            new Prefill(false, 0, List.of((int) 'a', (int) 'b')),
            new Prefill(true, 2, "XY<tool_call>\n".chars().boxed().toList()),
            new Prefill(true, 3, List.of((int) 'Y')));
  }

  @Test
  void scoresAnActivatedHiddenStateWithoutDiscardingThePhysicalPrefix() {
    ActivatedBackend backend = new ActivatedBackend();
    ToolApplicabilityHead head =
        new ToolApplicabilityHead(
            new float[] {1, 2}, new float[] {2, 4}, new float[] {0.5f, -0.25f}, 0.125f);

    try (ActivatedToolCallingModel model = new ActivatedToolCallingModel(backend, 1);
        ActivatedToolTurn turn = model.openToolTurn(ModelPrompt.text("abXY"))) {
      ToolApplicabilityScore score = turn.scoreToolApplicability(head);

      assertThat(score.score()).isEqualTo(0.375);
      assertThat(score.shouldCall()).isTrue();
      assertThat(turn.physicallySharesPrefix()).isTrue();
      turn.generateToolCall(deterministicOptions(), TokenConstraint.unrestricted());
    }

    assertThat(backend.prefills)
        .containsExactly(
            new Prefill(false, 0, List.of((int) 'a', (int) 'b')),
            new Prefill(true, 2, "XY<tool_call>\n".chars().boxed().toList()),
            new Prefill(true, 3, List.of((int) 'Y')));
  }

  @Test
  void rejectsInvalidDecisionTokensBeforeMutatingTheActivatedBranch() {
    ActivatedBackend backend = new ActivatedBackend();

    try (ActivatedToolCallingModel model = new ActivatedToolCallingModel(backend, 1);
        ActivatedToolTurn turn = model.openToolTurn(ModelPrompt.text("abXY"))) {
      assertThatThrownBy(() -> turn.scoreToolDecision(128, 3))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("callTokenId");

      turn.generateToolCall(deterministicOptions(), TokenConstraint.unrestricted());
    }

    assertThat(backend.prefills)
        .containsExactly(
            new Prefill(false, 0, List.of((int) 'a', (int) 'b')),
            new Prefill(true, 2, List.of((int) 'X', (int) 'Y')));
  }

  @Test
  void canRecomputeTheBasePrefixIndependentlyForAnHonestSharingCrossoverComparison() {
    ActivatedBackend backend = new ActivatedBackend();

    try (ActivatedToolCallingModel model = new ActivatedToolCallingModel(backend, 1);
        ActivatedToolTurn turn =
            model.openToolTurn(
                ModelPrompt.text("abXY"), ActivatedToolCallingModel.PrefixStrategy.RECOMPUTED)) {
      assertThat(turn.sharedPrefixTokens()).isZero();
      assertThat(turn.sharedPrefixBytes()).isZero();
      assertThat(turn.physicallySharesPrefix()).isFalse();
      assertThat(turn.baseInferenceStateBytes()).hasValue(96);
      assertThat(turn.toolInferenceStateBytes()).hasValue(96);
      assertThat(turn.uniqueInferenceStateBytes()).hasValue(192);

      turn.generateToolCall(deterministicOptions(), TokenConstraint.unrestricted());
      turn.generateBaseResponse(ModelPrompt.text("abXYz"), deterministicOptions());

      assertThat(turn.toolMetrics().promptCache().cacheReadInputTokens()).isEqualTo(2);
      assertThat(turn.responseMetrics().promptCache().cacheReadInputTokens()).isEqualTo(2);
      assertThat(backend.prefills)
          .containsExactly(
              new Prefill(false, 0, List.of((int) 'a', (int) 'b')),
              new Prefill(false, 0, List.of((int) 'a', (int) 'b')),
              new Prefill(true, 2, List.of((int) 'X', (int) 'Y')),
              new Prefill(false, 2, List.of((int) 'X', (int) 'Y', (int) 'z')));
    }
  }

  @Test
  void anAutomaticRecomputedFirstTurnPromotesToSharingAtTheMeasuredCrossover() {
    ActivatedBackend backend = new ActivatedBackend();

    try (ActivatedToolCallingModel model = new ActivatedToolCallingModel(backend, 6);
        ActivatedToolTurn first = model.openToolTurn(ModelPrompt.text("abXY"))) {
      first.generateToolCall(deterministicOptions(), TokenConstraint.unrestricted());
      first.generateBaseResponse(ModelPrompt.text("abXYz"), deterministicOptions());

      try (SharedToolTurn later = first.continueToolSelection(ModelPrompt.text("abXYzqXY"))) {
        assertThat(later.sharedPrefixTokens()).isEqualTo(6);
        assertThat(later.sharedPrefixBytes()).isEqualTo(64);
        assertThat(later.physicallySharesPrefix()).isTrue();
        later.generateToolCall(deterministicOptions(), TokenConstraint.unrestricted());
      }
    }

    assertThat(backend.prefills)
        .containsExactly(
            new Prefill(false, 0, List.of((int) 'a', (int) 'b')),
            new Prefill(false, 0, List.of((int) 'a', (int) 'b')),
            new Prefill(true, 2, List.of((int) 'X', (int) 'Y')),
            new Prefill(false, 2, List.of((int) 'X', (int) 'Y', (int) 'z')),
            new Prefill(false, 5, List.of((int) 'q')),
            new Prefill(true, 6, List.of((int) 'X', (int) 'Y')));
  }

  @Test
  void explicitRecomputationRemainsIndependentAcrossConsecutiveTurns() {
    ActivatedBackend backend = new ActivatedBackend();

    try (ActivatedToolCallingModel model = new ActivatedToolCallingModel(backend, 1);
        ActivatedToolTurn first =
            model.openToolTurn(
                ModelPrompt.text("abXY"), ActivatedToolCallingModel.PrefixStrategy.RECOMPUTED)) {
      first.generateToolCall(deterministicOptions(), TokenConstraint.unrestricted());
      first.generateBaseResponse(ModelPrompt.text("abXYz"), deterministicOptions());

      try (SharedToolTurn later = first.continueToolSelection(ModelPrompt.text("abXYzqXY"))) {
        assertThat(later.sharedPrefixTokens()).isZero();
        assertThat(later.sharedPrefixBytes()).isZero();
        assertThat(later.physicallySharesPrefix()).isFalse();
        later.generateToolCall(deterministicOptions(), TokenConstraint.unrestricted());
      }
    }

    assertThat(backend.prefills)
        .containsExactly(
            new Prefill(false, 0, List.of((int) 'a', (int) 'b')),
            new Prefill(false, 0, List.of((int) 'a', (int) 'b')),
            new Prefill(true, 2, List.of((int) 'X', (int) 'Y')),
            new Prefill(false, 2, List.of((int) 'X', (int) 'Y', (int) 'z')),
            new Prefill(false, 5, List.of((int) 'q')),
            new Prefill(
                false,
                0,
                List.of((int) 'a', (int) 'b', (int) 'X', (int) 'Y', (int) 'z', (int) 'q')),
            new Prefill(true, 6, List.of((int) 'X', (int) 'Y')));
  }

  @Test
  void rejectsAPromptWithoutThePinnedInvocation() {
    try (ActivatedToolCallingModel model =
        new ActivatedToolCallingModel(new ActivatedBackend(), 1)) {
      assertThatThrownBy(() -> model.openToolTurn(ModelPrompt.text("abX")))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("invocation sequence");
    }
  }

  @Test
  void aFailedToolGenerationCannotBeReusedForBaseSynthesisOrContinuation() {
    TokenConstraint rejectsEveryToken =
        new TokenConstraint() {
          @Override
          public boolean allows(int token) {
            return false;
          }

          @Override
          public void accept(int token) {}
        };

    try (ActivatedToolCallingModel model =
            new ActivatedToolCallingModel(new ActivatedBackend(), 1);
        ActivatedToolTurn turn = model.openToolTurn(ModelPrompt.text("abXY"))) {
      assertThatThrownBy(() -> turn.generateToolCall(deterministicOptions(), rejectsEveryToken))
          .hasMessageContaining("token constraint rejected every token");

      assertThatThrownBy(
              () -> turn.generateBaseResponse(ModelPrompt.text("abXYz"), deterministicOptions()))
          .isInstanceOf(IllegalStateException.class)
          .hasMessageContaining("failed");
      assertThatThrownBy(() -> turn.continueToolSelection(ModelPrompt.text("abXYzXY")))
          .isInstanceOf(IllegalStateException.class)
          .hasMessageContaining("failed");
    }
  }

  @Test
  void aStreamedToolFailureCannotBeReusedWhenTheConsumerAcceptsTheErrorCallback() {
    TokenConstraint rejectsEveryToken =
        new TokenConstraint() {
          @Override
          public boolean allows(int token) {
            return false;
          }

          @Override
          public void accept(int token) {}
        };
    List<Throwable> failures = new ArrayList<>();
    TokenStream acceptingStream =
        new TokenStream() {
          @Override
          public void onToken(String token) {}

          @Override
          public void onComplete() {}

          @Override
          public void onError(Throwable failure) {
            failures.add(failure);
          }
        };

    try (ActivatedToolCallingModel model =
            new ActivatedToolCallingModel(new ActivatedBackend(), 1);
        ActivatedToolTurn turn = model.openToolTurn(ModelPrompt.text("abXY"))) {
      turn.generateToolCall(deterministicOptions(), acceptingStream, rejectsEveryToken);

      assertThat(failures).singleElement().satisfies(failure -> assertThat(failure).isNotNull());
      assertThatThrownBy(
              () -> turn.generateBaseResponse(ModelPrompt.text("abXYz"), deterministicOptions()))
          .isInstanceOf(IllegalStateException.class)
          .hasMessageContaining("failed");
      assertThatThrownBy(() -> turn.continueToolSelection(ModelPrompt.text("abXYzXY")))
          .isInstanceOf(IllegalStateException.class)
          .hasMessageContaining("failed");
    }
  }

  @Test
  void aStreamedBaseResponseFailureCannotResumeTheConversation() {
    List<Throwable> failures = new ArrayList<>();
    TokenStream failingCompletion =
        new TokenStream() {
          @Override
          public void onToken(String token) {}

          @Override
          public void onComplete() {
            throw new IllegalStateException("completion consumer failed");
          }

          @Override
          public void onError(Throwable failure) {
            failures.add(failure);
          }
        };

    try (ActivatedToolCallingModel model =
            new ActivatedToolCallingModel(new ActivatedBackend(), 1);
        ActivatedToolTurn turn = model.openToolTurn(ModelPrompt.text("abXY"))) {
      turn.generateToolCall(deterministicOptions(), TokenConstraint.unrestricted());
      turn.generateBaseResponse(
          ModelPrompt.text("abXYz"), deterministicOptions(), failingCompletion);

      assertThat(failures)
          .singleElement()
          .satisfies(
              failure ->
                  assertThat(failure)
                      .isInstanceOf(IllegalStateException.class)
                      .hasMessageContaining("completion consumer failed"));
      assertThatThrownBy(turn::consumeBaseBranch)
          .isInstanceOf(IllegalStateException.class)
          .hasMessageContaining("failed");
      assertThatThrownBy(() -> turn.continueToolSelection(ModelPrompt.text("abXYzXY")))
          .isInstanceOf(IllegalStateException.class)
          .hasMessageContaining("failed");
    }
  }

  @Test
  void aConversationDoesNotResumeItsBaseBranchAfterAStreamedToolResultFailure() {
    List<Throwable> failures = new ArrayList<>();
    TokenStream failingCompletion =
        new TokenStream() {
          @Override
          public void onToken(String token) {}

          @Override
          public void onComplete() {
            throw new IllegalStateException("completion consumer failed");
          }

          @Override
          public void onError(Throwable failure) {
            failures.add(failure);
          }
        };

    try (ActivatedToolCallingModel model =
            new ActivatedToolCallingModel(new ActivatedBackend(), 1);
        ActivatedToolConversation conversation = model.openConversation()) {
      conversation.selectTool(
          ModelPrompt.text("abXY"), deterministicOptions(), TokenConstraint.unrestricted());

      conversation.completeToolResult(
          ModelPrompt.text("abXYz"), deterministicOptions(), failingCompletion);

      assertThat(failures).hasSize(1);
      assertThat(conversation.hasActiveToolTurn()).isTrue();
      assertThatThrownBy(
              () -> conversation.generateBase(ModelPrompt.text("abXYzw"), deterministicOptions()))
          .isInstanceOf(IllegalStateException.class)
          .hasMessageContaining("complete the active tool turn");
    }
  }

  @Test
  void keepsTemplateControlTokensAfterTheInvocationOnTheActivatedBranch() {
    ActivatedBackend backend = new ActivatedBackend();

    try (ActivatedToolCallingModel model = new ActivatedToolCallingModel(backend, 1);
        ActivatedToolTurn turn = model.openToolTurn(ModelPrompt.text("abXYn"))) {
      assertThat(turn.sharedPrefixTokens()).isEqualTo(2);
      turn.generateToolCall(deterministicOptions(), TokenConstraint.unrestricted());
    }

    assertThat(backend.prefills)
        .containsExactly(
            new Prefill(false, 0, List.of((int) 'a', (int) 'b')),
            new Prefill(true, 2, List.of((int) 'X', (int) 'Y', (int) 'n')));
  }

  @Test
  void extendsTheBaseBranchIntoAConsecutiveToolTurnWithoutReevaluatingPriorContext() {
    ActivatedBackend backend = new ActivatedBackend();

    try (ActivatedToolCallingModel model = new ActivatedToolCallingModel(backend, 1);
        ActivatedToolTurn first = model.openToolTurn(ModelPrompt.text("abXY"))) {
      first.generateToolCall(deterministicOptions(), TokenConstraint.unrestricted());

      try (SharedToolTurn second = first.continueToolSelection(ModelPrompt.text("abXYzXY"))) {
        assertThat(second.sharedPrefixTokens()).isEqualTo(5);
        assertThat(second.physicallySharesPrefix()).isTrue();
        second.generateToolCall(deterministicOptions(), TokenConstraint.unrestricted());
        second.generateBaseResponse(ModelPrompt.text("abXYzXYw"), deterministicOptions());

        assertThat(second.toolMetrics().promptCache().cacheReadInputTokens()).isEqualTo(5);
        assertThat(second.responseMetrics().promptCache().cacheReadInputTokens()).isEqualTo(5);
      }
    }

    assertThat(backend.prefills)
        .containsExactly(
            new Prefill(false, 0, List.of((int) 'a', (int) 'b')),
            new Prefill(true, 2, List.of((int) 'X', (int) 'Y')),
            new Prefill(false, 2, List.of((int) 'X', (int) 'Y', (int) 'z')),
            new Prefill(true, 5, List.of((int) 'X', (int) 'Y')),
            new Prefill(false, 5, List.of((int) 'X', (int) 'Y', (int) 'w')));
  }

  @Test
  void extendsTheBaseBranchAfterItsConversationalResponseIntoALaterToolTurn() {
    ActivatedBackend backend = new ActivatedBackend();

    try (ActivatedToolCallingModel model = new ActivatedToolCallingModel(backend, 1);
        ActivatedToolTurn first = model.openToolTurn(ModelPrompt.text("abXY"))) {
      first.generateToolCall(deterministicOptions(), TokenConstraint.unrestricted());
      first.generateBaseResponse(ModelPrompt.text("abXYz"), deterministicOptions());

      try (SharedToolTurn second = first.continueToolSelection(ModelPrompt.text("abXYzqXY"))) {
        assertThat(second.sharedPrefixTokens()).isEqualTo(6);
        assertThat(second.physicallySharesPrefix()).isTrue();
        second.generateToolCall(deterministicOptions(), TokenConstraint.unrestricted());
        second.generateBaseResponse(ModelPrompt.text("abXYzqXYw"), deterministicOptions());

        assertThat(second.toolMetrics().promptCache().cacheReadInputTokens()).isEqualTo(6);
        assertThat(second.responseMetrics().promptCache().cacheReadInputTokens()).isEqualTo(6);
      }
    }

    assertThat(backend.prefills)
        .containsExactly(
            new Prefill(false, 0, List.of((int) 'a', (int) 'b')),
            new Prefill(true, 2, List.of((int) 'X', (int) 'Y')),
            new Prefill(false, 2, List.of((int) 'X', (int) 'Y', (int) 'z')),
            new Prefill(false, 5, List.of((int) 'q')),
            new Prefill(true, 6, List.of((int) 'X', (int) 'Y')),
            new Prefill(false, 6, List.of((int) 'X', (int) 'Y', (int) 'w')));
  }

  @Test
  void reconcilesTemplateControlChangesAfterAResponseWithoutRebuildingTheSharedPrefix() {
    ActivatedBackend backend = new ActivatedBackend();

    try (ActivatedToolCallingModel model = new ActivatedToolCallingModel(backend, 1);
        ActivatedToolTurn first = model.openToolTurn(ModelPrompt.text("abXY"))) {
      first.generateToolCall(deterministicOptions(), TokenConstraint.unrestricted());
      first.generateBaseResponse(ModelPrompt.text("abXYn"), deterministicOptions());

      try (SharedToolTurn later = first.continueToolSelection(ModelPrompt.text("abXYrqXY"))) {
        assertThat(later.sharedPrefixTokens()).isEqualTo(6);
        assertThat(later.physicallySharesPrefix()).isTrue();
        later.generateToolCall(deterministicOptions(), TokenConstraint.unrestricted());
      }
    }

    assertThat(backend.prefills)
        .containsExactly(
            new Prefill(false, 0, List.of((int) 'a', (int) 'b')),
            new Prefill(true, 2, List.of((int) 'X', (int) 'Y')),
            new Prefill(false, 2, List.of((int) 'X', (int) 'Y', (int) 'n')),
            new Prefill(false, 4, List.of((int) 'r', (int) 'q')),
            new Prefill(true, 6, List.of((int) 'X', (int) 'Y')));
  }

  @Test
  void keepsOneBaseCacheLineageAcrossProseToolAndLaterProseTurns() {
    ActivatedBackend backend = new ActivatedBackend();

    try (ActivatedToolCallingModel model = new ActivatedToolCallingModel(backend, 1);
        ActivatedToolConversation conversation = model.openConversation()) {
      conversation.generateBase(ModelPrompt.text("a"), deterministicOptions());
      conversation.generateBase(ModelPrompt.text("ab"), deterministicOptions());

      conversation.selectTool(
          ModelPrompt.text("abqXY"), deterministicOptions(), TokenConstraint.unrestricted());
      assertThat(conversation.sharedPrefixTokens()).isEqualTo(3);
      assertThat(conversation.sharedPrefixBytes()).isEqualTo(64);
      assertThat(conversation.physicallySharesPrefix()).isTrue();
      conversation.completeToolResult(ModelPrompt.text("abqXYz"), deterministicOptions());

      conversation.generateBase(ModelPrompt.text("abqXYzw"), deterministicOptions());
    }

    assertThat(backend.prefills)
        .containsExactly(
            new Prefill(false, 0, List.of((int) 'a')),
            new Prefill(false, 1, List.of((int) 'b')),
            new Prefill(false, 2, List.of((int) 'q')),
            new Prefill(true, 3, List.of((int) 'X', (int) 'Y')),
            new Prefill(false, 3, List.of((int) 'X', (int) 'Y', (int) 'z')),
            new Prefill(false, 6, List.of((int) 'w')));
  }

  private static SamplingOptions deterministicOptions() {
    return SamplingOptions.builder().temperature(0).maxTokens(1).build();
  }

  private record Prefill(boolean activated, int startPosition, List<Integer> tokens) {}

  private static final class ActivatedBackend implements SharedPrefixInferenceBackend {
    private final List<Prefill> prefills = new ArrayList<>();
    private final Tokenizer tokenizer = new CharacterTokenizer();
    private int nextPrefix;
    private int defaultPosition;
    private final boolean failPhysicalSharingCheck;
    private int sessionCloseFailuresRemaining;
    private int sessionCloseAttempts;
    private boolean closed;

    private ActivatedBackend() {
      this(false, 0);
    }

    private ActivatedBackend(boolean failPhysicalSharingCheck, int sessionCloseFailuresRemaining) {
      this.failPhysicalSharingCheck = failPhysicalSharingCheck;
      this.sessionCloseFailuresRemaining = sessionCloseFailuresRemaining;
    }

    @Override
    public String name() {
      return "activated-fixture";
    }

    @Override
    public ModelMetadata metadata() {
      return new ModelMetadata("llama", "activated-fixture", 128, 128, 8, 1, 1, 1);
    }

    @Override
    public BackendDiagnostics diagnostics() {
      return BackendDiagnostics.unavailable(name());
    }

    @Override
    public Tokenizer tokenizer() {
      return tokenizer;
    }

    @Override
    public float[] forward(int token, int position) {
      defaultPosition = position + 1;
      return terminalLogits();
    }

    @Override
    public void reset() {
      defaultPosition = 0;
    }

    @Override
    public int maxBatchSize() {
      return 4;
    }

    @Override
    public InferenceSession openSession() {
      return new Session(false, 0, 0);
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
      if (state.activated && state.position == state.prefixLength) {
        int relative = startPosition - state.prefixLength;
        for (int index = 0; index < Math.min(tokens.length, 2 - relative); index++) {
          int expected = relative + index == 0 ? 'X' : 'Y';
          if (tokens[index] != expected) {
            throw new IllegalArgumentException("wrong activated invocation token");
          }
        }
      }
      prefills.add(
          new Prefill(
              state.activated, startPosition, java.util.Arrays.stream(tokens).boxed().toList()));
      state.position = startPosition + tokens.length;
      return terminalLogits();
    }

    @Override
    public boolean supportsHiddenState() {
      return true;
    }

    @Override
    public float[] prefillHiddenState(InferenceSession session, int[] tokens, int startPosition) {
      prefill(session, tokens, startPosition);
      return new float[] {3, 6};
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
    public boolean supportsActivatedBranch() {
      return true;
    }

    @Override
    public void activateAdapter(InferenceSession session) {
      Session state = requireSession(session);
      if (state.activated) {
        throw new IllegalStateException("adapter is already active");
      }
      state.activated = true;
      state.prefixLength = state.position;
    }

    @Override
    public Optional<ActivatedAdapterMetadata> activatedAdapter() {
      return Optional.of(
          new ActivatedAdapterMetadata(
              "test/base",
              "a".repeat(40),
              "b".repeat(64),
              java.util.Map.of("tokenizer.json", "d".repeat(64)),
              "c".repeat(64),
              1,
              1,
              List.of((int) 'X', (int) 'Y'),
              testProvenance()));
    }

    private static ActivatedAdapterMetadata.TrainingProvenance testProvenance() {
      return new ActivatedAdapterMetadata.TrainingProvenance(
          "test/dataset",
          "e".repeat(40),
          "train.jsonl",
          "f".repeat(64),
          "1".repeat(64),
          "2".repeat(64),
          "3".repeat(64),
          "formatter.java",
          "4".repeat(64));
    }

    @Override
    public SharedInferencePrefix freezePrefix(InferenceSession source) {
      Session state = requireSession(source);
      if (state.position == 0) {
        throw new IllegalArgumentException("empty prefix");
      }
      state.closed = true;
      return new Prefix(state.position, ++nextPrefix);
    }

    @Override
    public InferenceSession fork(SharedInferencePrefix prefix, Branch branch) {
      if (!(prefix instanceof Prefix owned)) {
        throw new IllegalArgumentException("foreign prefix");
      }
      return new Session(branch == Branch.ACTIVATED_ADAPTER, owned.checkpoint, owned.id);
    }

    @Override
    public boolean sharesPrefixStorage(InferenceSession first, InferenceSession second) {
      if (failPhysicalSharingCheck) {
        throw new IllegalStateException("fixture physical-sharing failure");
      }
      Session left = requireSession(first);
      Session right = requireSession(second);
      return left.prefixId != 0 && left.prefixId == right.prefixId;
    }

    @Override
    public void close() {
      closed = true;
    }

    private Session requireSession(InferenceSession session) {
      if (closed || !(session instanceof Session state) || state.closed) {
        throw new IllegalStateException("closed or foreign session");
      }
      return state;
    }

    private static float[] terminalLogits() {
      float[] logits = new float[128];
      logits[2] = 10;
      return logits;
    }

    private final class Session implements InferenceSession {
      private boolean activated;
      private int prefixLength;
      private final int prefixId;
      private int position;
      private boolean closed;

      private Session(boolean activated, int prefixLength, int prefixId) {
        this.activated = activated;
        this.prefixLength = prefixLength;
        this.prefixId = prefixId;
        this.position = prefixLength;
      }

      @Override
      public int checkpoint() {
        return position;
      }

      @Override
      public boolean isClosed() {
        return closed;
      }

      @Override
      public java.util.OptionalLong allocatedStateBytes() {
        return java.util.OptionalLong.of(96);
      }

      @Override
      public void close() {
        sessionCloseAttempts++;
        closed = true;
        if (sessionCloseFailuresRemaining > 0) {
          int ordinal = 3 - sessionCloseFailuresRemaining;
          sessionCloseFailuresRemaining--;
          throw new IllegalStateException("fixture session close failure " + ordinal);
        }
      }
    }

    private record Prefix(int checkpoint, int id) implements SharedInferencePrefix {
      @Override
      public long sharedBytes() {
        return 64;
      }
    }
  }

  private static final class CharacterTokenizer implements Tokenizer {
    @Override
    public int[] encode(String text) {
      return text.chars().toArray();
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
      return 128;
    }

    @Override
    public int bosToken() {
      return 0;
    }

    @Override
    public int eosToken() {
      return 2;
    }
  }
}
