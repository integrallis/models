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
package com.integrallis.models.decisions;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.models.api.InferenceSession;
import com.integrallis.models.api.LogitBatch;
import com.integrallis.models.api.ModelMetadata;
import com.integrallis.models.api.SharedInferencePrefix;
import com.integrallis.models.api.SharedPrefixInferenceBackend;
import com.integrallis.models.api.Tokenizer;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * The economic claim of this whole tier is that N decisions over one state cost one prefill and N
 * head reads, because every branch forks the same physical prefix. These tests hold the evaluator
 * to that, and hold it in both directions: a backend that cannot actually share is refused rather
 * than quietly served, so the capability can never be inert everywhere and still look compliant.
 */
@Tag("unit")
class SharedPrefixDecisionEvaluatorTest {

  private static final int[] STATE = {10, 11, 12, 13};

  @Test
  void everyQuestionGetsItsOwnVerdictInOrder() {
    StubBackend backend = new StubBackend();
    DecisionBatch batch = evaluator(backend).evaluate(STATE, List.of(noul("a"), noul("b")));

    assertThat(batch.verdicts()).hasSize(2);
    assertThat(batch.verdicts().get(0).space().question()).isEqualTo("a");
    assertThat(batch.verdicts().get(1).space().question()).isEqualTo("b");
  }

  @Test
  void theStateIsPrefilledAndFrozenExactlyOnceHoweverManyQuestionsAreAsked() {
    StubBackend backend = new StubBackend();

    evaluator(backend).evaluate(STATE, List.of(noul("a"), noul("b"), noul("c"), noul("d")));

    assertThat(backend.freezeCount).isEqualTo(1);
    assertThat(backend.statePrefillCount).isEqualTo(1);
    assertThat(backend.forkCount).isEqualTo(4);
  }

  @Test
  void theReportProvesSharingRatherThanInferringItFromTiming() {
    StubBackend backend = new StubBackend();

    DecisionBatch batch = evaluator(backend).evaluate(STATE, List.of(noul("a"), noul("b")));

    assertThat(batch.prefixFreezes()).isEqualTo(1);
    assertThat(batch.questionCount()).isEqualTo(2);
    assertThat(batch.allSharedPrefixStorage()).isTrue();
  }

  @Test
  void aBackendWhoseForksDoNotActuallyShareIsReportedRatherThanAccepted() {
    StubBackend backend = new StubBackend();
    backend.pretendSharing = false;

    DecisionBatch batch = evaluator(backend).evaluate(STATE, List.of(noul("a"), noul("b")));

    assertThat(batch.allSharedPrefixStorage()).isFalse();
  }

  @Test
  void aBackendThatCannotShareAPrefixIsRefusedUpFront() {
    StubBackend backend = new StubBackend();
    backend.supportsPrefixes = false;

    assertThatThrownBy(() -> evaluator(backend).evaluate(STATE, List.of(noul("a"))))
        .isInstanceOf(IllegalStateException.class)
        .hasMessageContaining("prefix");
  }

  @Test
  void aBackendWithoutHiddenStatesIsRefusedUpFront() {
    StubBackend backend = new StubBackend();
    backend.supportsHidden = false;

    assertThatThrownBy(() -> evaluator(backend).evaluate(STATE, List.of(noul("a"))))
        .isInstanceOf(IllegalStateException.class)
        .hasMessageContaining("hidden");
  }

  @Test
  void aQuestionContinuesFromTheEndOfTheStateRatherThanRestartingAtZero() {
    StubBackend backend = new StubBackend();

    evaluator(backend).evaluate(STATE, List.of(noul("a")));

    assertThat(backend.questionStartPositions).containsExactly(STATE.length);
  }

  @Test
  void noDecisionEverAsksTheBackendToGenerate() {
    StubBackend backend = new StubBackend();

    evaluator(backend).evaluate(STATE, List.of(noul("a"), noul("b")));

    assertThat(backend.generationCalls).isZero();
  }

  @Test
  void everySessionIsClosedIncludingWhenAHeadRejectsItsHiddenState() {
    StubBackend backend = new StubBackend();
    Question poisoned =
        new Question(
            new Noul("wrong width"),
            new int[] {99},
            new LinearDecisionHead(
                new Noul("wrong width"), new double[][] {{1.0}, {1.0}}, new double[] {0.0, 0.0}),
            1.0);

    assertThatThrownBy(() -> evaluator(backend).evaluate(STATE, List.of(poisoned)))
        .isInstanceOf(IllegalArgumentException.class);

    assertThat(backend.openSessions).isEmpty();
  }

  @Test
  void anEmptyQuestionListNeverTouchesTheBackend() {
    StubBackend backend = new StubBackend();

    DecisionBatch batch = evaluator(backend).evaluate(STATE, List.of());

    assertThat(batch.verdicts()).isEmpty();
    assertThat(backend.freezeCount).isZero();
  }

  private static SharedPrefixDecisionEvaluator evaluator(StubBackend backend) {
    return new SharedPrefixDecisionEvaluator(backend);
  }

  /** A Noul whose head reads the stub's two-wide hidden state. */
  private static Question noul(String proposition) {
    Noul space = new Noul(proposition);
    return new Question(
        space,
        new int[] {7},
        new LinearDecisionHead(
            space, new double[][] {{1.0, 0.0}, {0.0, 1.0}}, new double[] {0.0, 0.0}),
        1.0);
  }

  /** Records what the evaluator actually asked the backend to do. */
  private static final class StubBackend implements SharedPrefixInferenceBackend {

    private static final int HIDDEN_WIDTH = 2;

    boolean supportsPrefixes = true;
    boolean supportsHidden = true;
    boolean pretendSharing = true;
    int freezeCount;
    int forkCount;
    int statePrefillCount;
    int generationCalls;
    final List<Integer> questionStartPositions = new ArrayList<>();
    final Set<InferenceSession> openSessions = new HashSet<>();

    private final Set<InferenceSession> forked = new HashSet<>();

    @Override
    public String name() {
      return "stub";
    }

    @Override
    public boolean supportsSharedPrefixes() {
      return supportsPrefixes;
    }

    @Override
    public boolean supportsHiddenState() {
      return supportsHidden;
    }

    @Override
    public boolean supportsActivatedBranch() {
      return false;
    }

    @Override
    public int maxBatchSize() {
      return 8;
    }

    @Override
    public InferenceSession openSession() {
      StubSession session = new StubSession(this);
      openSessions.add(session);
      return session;
    }

    @Override
    public SharedInferencePrefix freezePrefix(InferenceSession source) {
      freezeCount++;
      openSessions.remove(source);
      return new StubPrefix(((StubSession) source).position);
    }

    @Override
    public InferenceSession fork(SharedInferencePrefix prefix, Branch branch) {
      forkCount++;
      StubSession session = new StubSession(this);
      session.position = ((StubPrefix) prefix).length;
      openSessions.add(session);
      forked.add(session);
      return session;
    }

    @Override
    public boolean sharesPrefixStorage(InferenceSession first, InferenceSession second) {
      return pretendSharing && forked.contains(first) && forked.contains(second);
    }

    @Override
    public float[] prefillHiddenState(InferenceSession session, int[] tokens, int startPosition) {
      if (forked.contains(session)) {
        questionStartPositions.add(startPosition);
      } else {
        statePrefillCount++;
      }
      ((StubSession) session).position = startPosition + tokens.length;
      return new float[] {0.25f, 0.75f};
    }

    @Override
    public float[] forward(InferenceSession session, int token, int position) {
      generationCalls++;
      throw new AssertionError("a decision must never ask the backend to generate");
    }

    @Override
    public float[] forward(int token, int position) {
      generationCalls++;
      throw new AssertionError("a decision must never ask the backend to generate");
    }

    @Override
    public LogitBatch forwardBatch(InferenceSession[] sessions, int[] tokens) {
      generationCalls++;
      throw new AssertionError("a decision must never ask the backend to generate");
    }

    @Override
    public ModelMetadata metadata() {
      throw new UnsupportedOperationException("not needed by these tests");
    }

    @Override
    public Tokenizer tokenizer() {
      throw new UnsupportedOperationException("not needed by these tests");
    }

    @Override
    public void rewind(InferenceSession session, int position) {
      ((StubSession) session).position = position;
    }

    @Override
    public void reset(InferenceSession session) {
      ((StubSession) session).position = 0;
    }

    @Override
    public void close() {
      // nothing to release
    }

    int hiddenWidth() {
      return HIDDEN_WIDTH;
    }
  }

  private static final class StubSession implements InferenceSession {
    private final StubBackend backend;
    int position;
    private boolean closed;

    StubSession(StubBackend backend) {
      this.backend = backend;
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
    public void close() {
      closed = true;
      backend.openSessions.remove(this);
    }
  }

  private record StubPrefix(int length) implements SharedInferencePrefix {
    @Override
    public int checkpoint() {
      return length;
    }

    @Override
    public long sharedBytes() {
      return length * 1024L;
    }
  }
}
