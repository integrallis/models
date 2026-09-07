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
package com.integrallis.models.api;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;

class BatchInferenceBackendTest {

  @Test
  void defaultRaggedPrefillKeepsIndependentPositionsAndReturnsOneFinalRowPerSession() {
    StubBackend backend = new StubBackend();
    StubSession first = new StubSession();
    StubSession second = new StubSession();

    LogitBatch logits =
        backend.prefillBatch(
            new InferenceSession[] {first, second}, new int[][] {{2, 3}, {5, 7, 11}});

    assertThat(logits.tokenCount()).isEqualTo(2);
    assertThat(logits.copyRow(0)).containsExactly(0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0);
    assertThat(logits.copyRow(1)).containsExactly(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1);
    assertThat(first.checkpoint()).isEqualTo(2);
    assertThat(second.checkpoint()).isEqualTo(3);
    assertThat(backend.supportsRaggedPrefillBatch()).isFalse();
  }

  private static final class StubBackend implements BatchInferenceBackend {
    @Override
    public int maxBatchSize() {
      return 2;
    }

    @Override
    public InferenceSession openSession() {
      return new StubSession();
    }

    @Override
    public float[] forward(InferenceSession session, int token, int position) {
      StubSession state = (StubSession) session;
      state.position = position + 1;
      float[] logits = new float[12];
      logits[Math.floorMod(token, logits.length)] = 1;
      return logits;
    }

    @Override
    public LogitBatch forwardBatch(InferenceSession[] sessions, int[] tokens) {
      throw new UnsupportedOperationException();
    }

    @Override
    public void rewind(InferenceSession session, int checkpoint) {
      ((StubSession) session).position = checkpoint;
    }

    @Override
    public void reset(InferenceSession session) {
      ((StubSession) session).position = 0;
    }

    @Override
    public String name() {
      return "stub";
    }

    @Override
    public ModelMetadata metadata() {
      return new ModelMetadata("stub", "stub", 12, 12, 1, 1, 1, 1);
    }

    @Override
    public Tokenizer tokenizer() {
      throw new UnsupportedOperationException();
    }

    @Override
    public float[] forward(int token, int position) {
      throw new UnsupportedOperationException();
    }

    @Override
    public void close() {}
  }

  private static final class StubSession implements InferenceSession {
    private int position;

    @Override
    public int checkpoint() {
      return position;
    }

    @Override
    public boolean isClosed() {
      return false;
    }

    @Override
    public void close() {}
  }
}
