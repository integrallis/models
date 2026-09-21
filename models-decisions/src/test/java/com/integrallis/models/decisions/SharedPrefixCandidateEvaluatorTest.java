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
import static org.assertj.core.api.Assertions.within;

import com.integrallis.models.api.InferenceSession;
import com.integrallis.models.api.LogitBatch;
import com.integrallis.models.api.ModelMetadata;
import com.integrallis.models.api.SharedInferencePrefix;
import com.integrallis.models.api.SharedPrefixInferenceBackend;
import com.integrallis.models.api.Tokenizer;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Scoring candidates by their token log-probabilities over a state prefilled once.
 *
 * <p>This is where the architecture pays. The state is evaluated a single time; each candidate then
 * costs only its own few tokens. A hosted service must re-ingest the state for every question and
 * every candidate, and no amount of hardware closes that gap because it is a consequence of being
 * outside the process that already computed the state.
 */
@Tag("unit")
class SharedPrefixCandidateEvaluatorTest {

  private static final int[] STATE = {10, 11, 12, 13};

  @Test
  void theStateIsEvaluatedOnceHoweverManyCandidatesAreScored() {
    StubBackend backend = new StubBackend();
    evaluator(backend)
        .evaluate(
            STATE,
            new Choice("q", List.of("a", "b", "c")),
            List.of(new int[] {20}, new int[] {21}, new int[] {22}));

    // Three candidates, and not one extra forward pass. The prefix logits already hold the
    // distribution over every single-token candidate, so scoring them is free once the state is
    // evaluated. Only multi-token candidates cost anything beyond their first token.
    assertThat(backend.statePrefills).isEqualTo(1);
    assertThat(backend.candidateForwards).isZero();
  }

  @Test
  void aStrongerCandidateWinsAndTheDistributionIsProper() {
    StubBackend backend = new StubBackend();
    backend.favour(21, 6.0);

    Verdict verdict =
        evaluator(backend)
            .evaluate(
                STATE,
                new Choice("q", List.of("a", "b", "c")),
                List.of(new int[] {20}, new int[] {21}, new int[] {22}));

    assertThat(verdict.winner()).isEqualTo("b");
    double total = 0.0;
    for (double p : verdict.probabilities()) {
      total += p;
    }
    assertThat(total).isCloseTo(1.0, within(1e-9));
  }

  @Test
  void aLongerCandidateIsNotPenalisedMerelyForBeingLonger() {
    // Length normalisation: two candidates whose per-token evidence is identical must score the
    // same even when one spans more tokens. Without it the shortest label wins by default.
    StubBackend backend = new StubBackend();

    Verdict verdict =
        evaluator(backend)
            .evaluate(
                STATE,
                new Choice("q", List.of("short", "long")),
                List.of(new int[] {30}, new int[] {30, 30, 30}));

    assertThat(verdict.probabilityOf("short"))
        .isCloseTo(verdict.probabilityOf("long"), within(1e-9));
  }

  @Test
  void everyCandidateIsScoredFromTheSamePrefixState() {
    // Scoring candidate three must not be affected by having scored one and two first: the session
    // is rewound to the prefix checkpoint between candidates.
    StubBackend backend = new StubBackend();
    Verdict together =
        evaluator(backend)
            .evaluate(
                STATE,
                new Choice("q", List.of("a", "b")),
                List.of(new int[] {40, 41}, new int[] {42}));

    StubBackend alone = new StubBackend();
    Verdict reversed =
        evaluator(alone)
            .evaluate(
                STATE,
                new Choice("q", List.of("b", "a")),
                List.of(new int[] {42}, new int[] {40, 41}));

    assertThat(reversed.probabilityOf("a")).isCloseTo(together.probabilityOf("a"), within(1e-9));
    // Only the multi-token candidate advanced the branch, so only it needed rewinding.
    assertThat(backend.rewinds).isEqualTo(1);
  }

  @Test
  void theSharingIsProvenRatherThanAssumed() {
    StubBackend backend = new StubBackend();
    CandidateBatch batch =
        evaluator(backend)
            .evaluateBatch(
                STATE,
                new Choice("q", List.of("a", "b")),
                List.of(new int[] {20, 20, 20}, new int[] {21}));

    assertThat(batch.statePrefills()).isEqualTo(1);
    // Three-token candidate costs two forwards beyond its free first token; the single-token
    // candidate costs none.
    assertThat(batch.candidateTokensEvaluated()).isEqualTo(2);
    assertThat(batch.sharedPrefixProven()).isTrue();
  }

  @Test
  void aBackendWhoseForkDoesNotShareIsReportedNotHidden() {
    StubBackend backend = new StubBackend();
    backend.pretendSharing = false;

    CandidateBatch batch =
        evaluator(backend)
            .evaluateBatch(
                STATE, new Choice("q", List.of("a", "b")), List.of(new int[] {20}, new int[] {21}));

    assertThat(batch.sharedPrefixProven()).isFalse();
  }

  private static SharedPrefixCandidateEvaluator evaluator(StubBackend backend) {
    return new SharedPrefixCandidateEvaluator(backend, 1.0);
  }

  /** Uniform logits except for favoured tokens, so expected scores are computable by hand. */
  private static final class StubBackend implements SharedPrefixInferenceBackend {
    private static final int VOCAB = 64;
    int statePrefills;
    int candidateForwards;
    int rewinds;
    boolean pretendSharing = true;
    private final java.util.Map<Integer, Double> favoured = new java.util.HashMap<>();
    private final java.util.Set<InferenceSession> forked = new java.util.HashSet<>();

    void favour(int token, double bonus) {
      favoured.put(token, bonus);
    }

    private float[] logits() {
      float[] l = new float[VOCAB];
      for (var e : favoured.entrySet()) {
        l[e.getKey()] = e.getValue().floatValue();
      }
      return l;
    }

    @Override
    public String name() {
      return "stub";
    }

    @Override
    public boolean supportsHiddenState() {
      return true;
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
      return new StubSession();
    }

    @Override
    public float[] prefill(InferenceSession s, int[] tokens, int start) {
      statePrefills++;
      ((StubSession) s).position = start + tokens.length;
      return logits();
    }

    @Override
    public float[] forward(InferenceSession s, int token, int position) {
      candidateForwards++;
      ((StubSession) s).position = position + 1;
      return logits();
    }

    @Override
    public void rewind(InferenceSession s, int checkpoint) {
      rewinds++;
      ((StubSession) s).position = checkpoint;
    }

    @Override
    public SharedInferencePrefix freezePrefix(InferenceSession source) {
      return new StubPrefix(((StubSession) source).position);
    }

    @Override
    public InferenceSession fork(SharedInferencePrefix prefix, Branch branch) {
      StubSession s = new StubSession();
      s.position = ((StubPrefix) prefix).length;
      forked.add(s);
      return s;
    }

    @Override
    public boolean sharesPrefixStorage(InferenceSession a, InferenceSession b) {
      return pretendSharing && forked.contains(a) && forked.contains(b);
    }

    @Override
    public void reset(InferenceSession s) {
      ((StubSession) s).position = 0;
    }

    @Override
    public float[] forward(int t, int p) {
      throw new AssertionError("no generation");
    }

    @Override
    public LogitBatch forwardBatch(InferenceSession[] s, int[] t) {
      throw new AssertionError("no generation");
    }

    @Override
    public ModelMetadata metadata() {
      throw new UnsupportedOperationException();
    }

    @Override
    public Tokenizer tokenizer() {
      throw new UnsupportedOperationException();
    }

    @Override
    public void close() {}
  }

  private static final class StubSession implements InferenceSession {
    int position;
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
      closed = true;
    }
  }

  private record StubPrefix(int length) implements SharedInferencePrefix {
    @Override
    public int checkpoint() {
      return length;
    }

    @Override
    public long sharedBytes() {
      return length * 512L;
    }
  }

  private static List<int[]> tokens(int... ids) {
    List<int[]> out = new ArrayList<>();
    for (int id : ids) {
      out.add(new int[] {id});
    }
    return out;
  }
}
