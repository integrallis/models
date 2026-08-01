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
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.models.api.BatchInferenceBackend;
import com.integrallis.models.api.InferenceSession;
import com.integrallis.models.api.LogitBatch;
import com.integrallis.models.api.ModelMetadata;
import com.integrallis.models.api.Tokenizer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;
import java.util.Map;
import java.util.function.LongSupplier;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class SessionBatchProfileCliTest {

  @TempDir Path directory;

  @Test
  void sequentialAndBatchedModesProduceTheSameMeasuredTokenTrace() throws Exception {
    Path model = Files.write(directory.resolve("fixture.gguf"), new byte[] {1, 2, 3});
    SessionBatchProfileCli.Configuration sequential =
        configuration(model, SessionBatchProfileCli.Mode.SEQUENTIAL, 2, 1, 3);
    SessionBatchProfileCli.Configuration batched =
        configuration(model, SessionBatchProfileCli.Mode.BATCHED, 2, 1, 3);
    FakeBatchBackend sequentialBackend = new FakeBatchBackend(4);
    FakeBatchBackend batchedBackend = new FakeBatchBackend(4);

    SessionBatchProfileCli.Result sequentialResult =
        SessionBatchProfileCli.profile(sequentialBackend, sequential, ticker(100, 1_100));
    SessionBatchProfileCli.Result batchedResult =
        SessionBatchProfileCli.profile(batchedBackend, batched, ticker(200, 1_200));

    assertThat(sequentialResult.outputTokenSha256()).isEqualTo(batchedResult.outputTokenSha256());
    assertThat(sequentialResult.generatedTokens()).isEqualTo(6);
    assertThat(sequentialResult.elapsedNanos()).isEqualTo(1_000);
    assertThat(sequentialResult.aggregateTokensPerSecond()).isEqualTo(6_000_000.0);
    assertThat(sequentialResult.perRequestTokenMillis()).isEqualTo(1.0 / 3_000.0);
    assertThat(sequentialBackend.batchSizes()).isEmpty();
    assertThat(sequentialBackend.forwardCalls()).isEqualTo(8);
    assertThat(batchedBackend.batchSizes()).containsExactly(2, 2, 2, 2);
    assertThat(batchedBackend.forwardCalls()).isZero();
    assertThat(sequentialBackend.closedSessions()).isEqualTo(4);
    assertThat(batchedBackend.closedSessions()).isEqualTo(4);
  }

  @Test
  void rejectsConcurrencyBeyondTheLoadedModelsBatchCapacity() throws Exception {
    Path model = Files.write(directory.resolve("fixture.gguf"), new byte[] {1});
    SessionBatchProfileCli.Configuration configuration =
        configuration(model, SessionBatchProfileCli.Mode.BATCHED, 3, 0, 1);

    assertThatThrownBy(
            () ->
                SessionBatchProfileCli.profile(
                    new FakeBatchBackend(2), configuration, ticker(0, 1)))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessage("concurrency 3 exceeds backend batch capacity 2");
  }

  @Test
  void capturesJvmMemoryBeforeMeasuredSessionsClose() throws Exception {
    Path model = Files.write(directory.resolve("fixture.gguf"), new byte[] {1});
    SessionBatchProfileCli.Configuration configuration =
        configuration(model, SessionBatchProfileCli.Mode.BATCHED, 2, 1, 1);
    FakeBatchBackend backend = new FakeBatchBackend(4);
    JvmMemorySnapshot before = memorySnapshot(100);
    JvmMemorySnapshot active = memorySnapshot(200);
    Deque<JvmMemorySnapshot> snapshots = new ArrayDeque<>(List.of(before, active));

    SessionBatchProfileCli.Result result =
        SessionBatchProfileCli.profile(
            backend,
            configuration,
            ticker(100, 1_100),
            () -> {
              assertThat(backend.closedSessions()).isEqualTo(2);
              return snapshots.removeFirst();
            });

    assertThat(result.schemaVersion()).isEqualTo(3);
    assertThat(result.jvmMemoryBefore()).isEqualTo(before);
    assertThat(result.jvmMemoryActive()).isEqualTo(active);
    assertThat(result.currentRssBytes()).isGreaterThanOrEqualTo(0);
    assertThat(result.anonymousRssBytes()).isGreaterThanOrEqualTo(0);
    assertThat(result.fileRssBytes()).isGreaterThanOrEqualTo(0);
    assertThat(result.sharedMemoryRssBytes()).isGreaterThanOrEqualTo(0);
    assertThat(snapshots).isEmpty();
    assertThat(backend.closedSessions()).isEqualTo(4);
  }

  private SessionBatchProfileCli.Configuration configuration(
      Path model,
      SessionBatchProfileCli.Mode mode,
      int concurrency,
      int warmupSteps,
      int measuredSteps) {
    return new SessionBatchProfileCli.Configuration(
        new PureJavaModelSource(model.toString(), model),
        "profile prompt",
        64,
        concurrency,
        warmupSteps,
        measuredSteps,
        mode,
        directory.resolve(mode.name().toLowerCase() + ".json"));
  }

  private static LongSupplier ticker(long... values) {
    return new LongSupplier() {
      private int index;

      @Override
      public long getAsLong() {
        return values[index++];
      }
    };
  }

  private static JvmMemorySnapshot memorySnapshot(long base) {
    return new JvmMemorySnapshot(
        base, base + 1, base + 2, base + 3, Map.of(), NativeMemoryTracking.Summary.unavailable());
  }

  private static final class FakeBatchBackend implements BatchInferenceBackend {
    private static final int VOCABULARY_SIZE = 32;

    private final int maxBatchSize;
    private final List<Integer> batchSizes = new ArrayList<>();
    private int nextSessionId;
    private int forwardCalls;
    private int closedSessions;

    private FakeBatchBackend(int maxBatchSize) {
      this.maxBatchSize = maxBatchSize;
    }

    @Override
    public int maxBatchSize() {
      return maxBatchSize;
    }

    @Override
    public InferenceSession openSession() {
      return new FakeSession(nextSessionId++);
    }

    @Override
    public float[] forward(InferenceSession session, int token, int position) {
      forwardCalls++;
      return logits(checked(session), token, position);
    }

    @Override
    public float[] prefill(InferenceSession session, int[] tokens, int startPosition) {
      FakeSession state = checked(session);
      state.position = startPosition + tokens.length;
      return logitsFor(state, tokens[tokens.length - 1], state.position - 1);
    }

    @Override
    public LogitBatch forwardBatch(InferenceSession[] sessions, int[] tokens) {
      batchSizes.add(sessions.length);
      float[] values = new float[sessions.length * VOCABULARY_SIZE];
      for (int index = 0; index < sessions.length; index++) {
        FakeSession session = checked(sessions[index]);
        float[] row = logits(session, tokens[index], session.position);
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
      return "fake-batch";
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
          return new int[] {1, Math.floorMod(text.hashCode(), VOCABULARY_SIZE)};
        }

        @Override
        public String decode(int[] tokens) {
          return Integer.toString(tokens.length);
        }

        @Override
        public String decode(int token) {
          return Integer.toString(token);
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

    private float[] logits(FakeSession session, int token, int position) {
      if (position != session.position) {
        throw new IllegalArgumentException("non-sequential position");
      }
      float[] logits = logitsFor(session, token, position);
      session.position++;
      return logits;
    }

    private float[] logitsFor(FakeSession session, int token, int position) {
      float[] logits = new float[VOCABULARY_SIZE];
      logits[Math.floorMod(token + session.id + position, VOCABULARY_SIZE)] = 1.0f;
      return logits;
    }

    private FakeSession checked(InferenceSession session) {
      if (!(session instanceof FakeSession state) || state.closed) {
        throw new IllegalArgumentException("invalid session");
      }
      return state;
    }

    private List<Integer> batchSizes() {
      return batchSizes;
    }

    private int forwardCalls() {
      return forwardCalls;
    }

    private int closedSessions() {
      return closedSessions;
    }

    private final class FakeSession implements InferenceSession {
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
        if (!closed) {
          closed = true;
          closedSessions++;
        }
      }
    }
  }
}
