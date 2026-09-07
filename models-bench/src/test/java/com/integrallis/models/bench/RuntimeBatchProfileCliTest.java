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
package com.integrallis.models.bench;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.models.api.BatchInferenceBackend;
import com.integrallis.models.api.InferenceSession;
import com.integrallis.models.api.LogitBatch;
import com.integrallis.models.api.ModelMetadata;
import com.integrallis.models.api.Tokenizer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class RuntimeBatchProfileCliTest {

  @TempDir Path directory;

  @Test
  void serializedAndContinuousModesPreserveOutputsAndMeasurePhysicalBatches() throws Exception {
    Path model = Files.write(directory.resolve("fixture.gguf"), new byte[] {1, 2, 3});
    var serialized = configuration(model, RuntimeBatchProfileCli.Mode.SERIALIZED);
    var continuous = configuration(model, RuntimeBatchProfileCli.Mode.CONTINUOUS);
    var serializedBackend = new FakeBatchBackend();
    var continuousBackend = new FakeBatchBackend();

    var serializedResult = RuntimeBatchProfileCli.profile(serializedBackend, serialized);
    var continuousResult = RuntimeBatchProfileCli.profile(continuousBackend, continuous);

    assertThat(continuousResult.outputSha256()).isEqualTo(serializedResult.outputSha256());
    assertThat(continuousResult.successfulRequests()).isEqualTo(4);
    assertThat(continuousResult.completionTokens()).isEqualTo(8);
    assertThat(continuousResult.scheduler().largestBatch()).isEqualTo(2);
    assertThat(continuousResult.scheduler().meanBatchSize()).isEqualTo(2.0);
    assertThat(serializedResult.scheduler()).isNull();
    assertThat(serializedBackend.batchSizes()).isEmpty();
    assertThat(continuousBackend.batchSizes()).hasSize(3).allMatch(size -> size == 2);
    assertThat(continuousBackend.prefillCalls()).isEqualTo(6);
  }

  private RuntimeBatchProfileCli.Configuration configuration(
      Path model, RuntimeBatchProfileCli.Mode mode) {
    return new RuntimeBatchProfileCli.Configuration(
        new PureJavaModelSource(model.toString(), model),
        "profile prompt",
        64,
        2,
        1,
        2,
        2,
        mode,
        Duration.ofMillis(10),
        directory.resolve(mode.externalName() + ".json"));
  }

  private static final class FakeBatchBackend implements BatchInferenceBackend {
    private static final int VOCABULARY_SIZE = 32;

    private final List<Integer> batchSizes = new ArrayList<>();
    private int nextSession;
    private int prefillCalls;

    List<Integer> batchSizes() {
      return List.copyOf(batchSizes);
    }

    int prefillCalls() {
      return prefillCalls;
    }

    @Override
    public int maxBatchSize() {
      return 4;
    }

    @Override
    public InferenceSession openSession() {
      return new FakeSession(nextSession++);
    }

    @Override
    public float[] forward(InferenceSession session, int token, int position) {
      FakeSession state = checked(session);
      if (position != state.position) {
        throw new IllegalArgumentException("non-sequential position");
      }
      state.position++;
      return logits(token, state.id);
    }

    @Override
    public float[] prefill(InferenceSession session, int[] tokens, int startPosition) {
      prefillCalls++;
      FakeSession state = checked(session);
      state.position = startPosition + tokens.length;
      return logits(tokens[tokens.length - 1], state.id);
    }

    @Override
    public LogitBatch forwardBatch(InferenceSession[] sessions, int[] tokens) {
      batchSizes.add(sessions.length);
      float[] values = new float[sessions.length * VOCABULARY_SIZE];
      for (int index = 0; index < sessions.length; index++) {
        FakeSession state = checked(sessions[index]);
        float[] row = forward(state, tokens[index], state.position);
        System.arraycopy(row, 0, values, index * VOCABULARY_SIZE, VOCABULARY_SIZE);
      }
      return new LogitBatch(sessions.length, VOCABULARY_SIZE, values);
    }

    @Override
    public void rewind(InferenceSession session, int checkpoint) {
      checked(session).position = checkpoint;
    }

    @Override
    public void reset(InferenceSession session) {
      checked(session).position = 0;
    }

    @Override
    public String name() {
      return "fake-runtime-batch";
    }

    @Override
    public ModelMetadata metadata() {
      return new ModelMetadata("fake", "batch", 128, VOCABULARY_SIZE, 1, 1, 1, 1);
    }

    @Override
    public Tokenizer tokenizer() {
      return new Tokenizer() {
        @Override
        public int[] encode(String text) {
          return new int[] {1, 2};
        }

        @Override
        public String decode(int[] tokens) {
          return Integer.toString(tokens.length);
        }

        @Override
        public String decode(int token) {
          return Character.toString((char) ('A' + Math.floorMod(token, 26)));
        }

        @Override
        public int eosToken() {
          return -1;
        }

        @Override
        public int bosToken() {
          return -1;
        }

        @Override
        public int vocabSize() {
          return VOCABULARY_SIZE;
        }
      };
    }

    @Override
    public float[] forward(int token, int position) {
      throw new UnsupportedOperationException("default sequence is not used");
    }

    @Override
    public void close() {}

    private static FakeSession checked(InferenceSession session) {
      return (FakeSession) session;
    }

    private static float[] logits(int token, int session) {
      float[] values = new float[VOCABULARY_SIZE];
      values[Math.floorMod(token + session + 1, VOCABULARY_SIZE)] = 1;
      return values;
    }
  }

  private static final class FakeSession implements InferenceSession {
    private final int id;
    private int position;
    private boolean closed;

    private FakeSession(int id) {
      this.id = id;
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
    }
  }
}
