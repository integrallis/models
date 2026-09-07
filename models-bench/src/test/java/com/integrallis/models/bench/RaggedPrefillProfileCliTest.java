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
import java.util.concurrent.atomic.AtomicLong;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class RaggedPrefillProfileCliTest {

  @TempDir Path directory;

  @Test
  void sequentialAndRaggedModesRetainTheSameResultTrace() throws Exception {
    Path model = Files.write(directory.resolve("fixture.gguf"), new byte[] {1, 2, 3});
    var sequentialBackend = new FakeBackend();
    var raggedBackend = new FakeBackend();

    var sequential =
        RaggedPrefillProfileCli.profile(
            sequentialBackend,
            configuration(model, RaggedPrefillProfileCli.Mode.SEQUENTIAL),
            ticker());
    var ragged =
        RaggedPrefillProfileCli.profile(
            raggedBackend, configuration(model, RaggedPrefillProfileCli.Mode.RAGGED), ticker());

    assertThat(ragged.outputSha256()).isEqualTo(sequential.outputSha256());
    assertThat(ragged.promptTokens()).isEqualTo(5);
    assertThat(ragged.elapsedNanos()).isEqualTo(100);
    assertThat(raggedBackend.raggedPrefillCalls).isEqualTo(1);
    assertThat(sequentialBackend.raggedPrefillCalls).isZero();
    assertThat(sequentialBackend.sequentialPrefillCalls).isEqualTo(2);
    assertThat(raggedBackend.sequentialPrefillCalls).isZero();
  }

  private RaggedPrefillProfileCli.Configuration configuration(
      Path model, RaggedPrefillProfileCli.Mode mode) {
    return new RaggedPrefillProfileCli.Configuration(
        new PureJavaModelSource(model.toString(), model),
        "profile",
        64,
        2,
        0,
        1,
        mode,
        directory.resolve(mode.externalName() + ".json"));
  }

  private static java.util.function.LongSupplier ticker() {
    AtomicLong time = new AtomicLong();
    return () -> time.getAndAdd(100);
  }

  private static final class FakeBackend implements BatchInferenceBackend {
    private int sequentialPrefillCalls;
    private int raggedPrefillCalls;

    @Override
    public int maxBatchSize() {
      return 2;
    }

    @Override
    public boolean supportsRaggedPrefillBatch() {
      return true;
    }

    @Override
    public InferenceSession openSession() {
      return new FakeSession();
    }

    @Override
    public float[] forward(InferenceSession session, int token, int position) {
      FakeSession state = (FakeSession) session;
      state.position = position + 1;
      return logits(token);
    }

    @Override
    public float[] prefill(InferenceSession session, int[] tokens, int startPosition) {
      sequentialPrefillCalls++;
      FakeSession state = (FakeSession) session;
      state.position = startPosition + tokens.length;
      return logits(tokens[tokens.length - 1]);
    }

    @Override
    public LogitBatch prefillBatch(InferenceSession[] sessions, int[][] tokenBatches) {
      raggedPrefillCalls++;
      float[] values = new float[sessions.length * 16];
      for (int index = 0; index < sessions.length; index++) {
        FakeSession state = (FakeSession) sessions[index];
        state.position += tokenBatches[index].length;
        System.arraycopy(
            logits(tokenBatches[index][tokenBatches[index].length - 1]), 0, values, index * 16, 16);
      }
      return new LogitBatch(sessions.length, 16, values);
    }

    @Override
    public LogitBatch forwardBatch(InferenceSession[] sessions, int[] tokens) {
      float[] values = new float[sessions.length * 16];
      for (int index = 0; index < sessions.length; index++) {
        float[] row = forward(sessions[index], tokens[index], sessions[index].checkpoint());
        System.arraycopy(row, 0, values, index * 16, 16);
      }
      return new LogitBatch(sessions.length, 16, values);
    }

    @Override
    public void rewind(InferenceSession session, int checkpoint) {
      ((FakeSession) session).position = checkpoint;
    }

    @Override
    public void reset(InferenceSession session) {
      ((FakeSession) session).position = 0;
    }

    @Override
    public String name() {
      return "fake-ragged-prefill";
    }

    @Override
    public ModelMetadata metadata() {
      return new ModelMetadata("fake", "ragged", 64, 16, 1, 1, 1, 1);
    }

    @Override
    public Tokenizer tokenizer() {
      return new Tokenizer() {
        @Override
        public int[] encode(String text) {
          return text.contains("Request 1") ? new int[] {1, 2, 3} : new int[] {1, 2};
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
        public int eosToken() {
          return -1;
        }

        @Override
        public int bosToken() {
          return -1;
        }

        @Override
        public int vocabSize() {
          return 16;
        }
      };
    }

    @Override
    public float[] forward(int token, int position) {
      throw new UnsupportedOperationException();
    }

    @Override
    public void close() {}

    private static float[] logits(int token) {
      float[] values = new float[16];
      values[Math.floorMod(token + 1, values.length)] = 1;
      return values;
    }
  }

  private static final class FakeSession implements InferenceSession {
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
