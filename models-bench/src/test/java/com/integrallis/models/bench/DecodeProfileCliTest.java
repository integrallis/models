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

import com.integrallis.models.api.InferenceBackend;
import com.integrallis.models.api.ModelMetadata;
import com.integrallis.models.api.Tokenizer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.modeljars.ModelJarRegistry;

class DecodeProfileCliTest {

  @TempDir Path directory;

  @Test
  void resolvesModelJarAliasWithoutDiscardingItsDescriptor() throws Exception {
    Path model = Files.write(directory.resolve("fixture.gguf"), new byte[] {1, 2, 3});
    var descriptor = ModelJarTestFixtures.descriptor("fixture", model);

    DecodeProfileCli.Configuration configuration =
        DecodeProfileCli.parse(
            new String[] {"--modeljar", "fixture"}, ModelJarRegistry.of(List.of(descriptor)));

    assertThat(configuration.model().artifact()).isEqualTo(model);
    assertThat(configuration.model().descriptor()).contains(descriptor);
  }

  @Test
  void recordsOnlyDecodeCallsAfterPromptAndWarmup() throws Exception {
    Path model = Files.write(directory.resolve("unused.gguf"), new byte[] {1});
    List<String> events = new ArrayList<>();
    FakeBackend backend = new FakeBackend(events);
    DecodeProfileCli.Configuration configuration =
        new DecodeProfileCli.Configuration(
            new PureJavaModelSource(model.toString(), model, Optional.empty()),
            "profile prompt",
            5,
            7,
            2,
            3,
            Path.of("decode.jfr"));

    DecodeProfileCli.Result result =
        DecodeProfileCli.profile(
            backend,
            configuration,
            new FakeRecording(events),
            new FakeGcMetrics(
                events,
                new DecodeProfileCli.GcMetrics(12, 400),
                new DecodeProfileCli.GcMetrics(15, 407)));

    assertThat(events)
        .containsExactly(
            "reset",
            "encode:profile prompt",
            "prefill:1,2@0",
            "forward:7@2",
            "forward:7@3",
            "reset",
            "prefill:1,2@0",
            "record:start",
            "gc:snapshot",
            "forward:7@2",
            "forward:7@3",
            "forward:7@4",
            "gc:snapshot",
            "record:stop",
            "record:dump:decode.jfr");
    assertThat(result.promptTokens()).isEqualTo(2);
    assertThat(result.warmupTokens()).isEqualTo(2);
    assertThat(result.measuredTokens()).isEqualTo(3);
    assertThat(result.logitChecksum()).isEqualTo(9.0);
    assertThat(result.gcCollections()).isEqualTo(3);
    assertThat(result.gcPauseMillis()).isEqualTo(7);
  }

  private static final class FakeGcMetrics implements DecodeProfileCli.GcMetricsSource {
    private final List<String> events;
    private final List<DecodeProfileCli.GcMetrics> metrics;
    private int index;

    private FakeGcMetrics(List<String> events, DecodeProfileCli.GcMetrics... metrics) {
      this.events = events;
      this.metrics = List.of(metrics);
    }

    @Override
    public DecodeProfileCli.GcMetrics snapshot() {
      events.add("gc:snapshot");
      return metrics.get(index++);
    }
  }

  private static final class FakeRecording implements DecodeProfileCli.ProfileRecording {
    private final List<String> events;

    private FakeRecording(List<String> events) {
      this.events = events;
    }

    @Override
    public void start() {
      events.add("record:start");
    }

    @Override
    public void stop() {
      events.add("record:stop");
    }

    @Override
    public void dump(Path output) {
      events.add("record:dump:" + output);
    }

    @Override
    public void close() {}
  }

  private static final class FakeBackend implements InferenceBackend {
    private final List<String> events;
    private final Tokenizer tokenizer =
        new Tokenizer() {
          @Override
          public int[] encode(String text) {
            events.add("encode:" + text);
            return new int[] {1, 2};
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
            return 32;
          }
        };

    private FakeBackend(List<String> events) {
      this.events = events;
    }

    @Override
    public String name() {
      return "fake";
    }

    @Override
    public ModelMetadata metadata() {
      return new ModelMetadata("fake", "test", 128, 32, 1, 1, 1, 1);
    }

    @Override
    public Tokenizer tokenizer() {
      return tokenizer;
    }

    @Override
    public float[] forward(int token, int position) {
      events.add("forward:" + token + "@" + position);
      return new float[] {position};
    }

    @Override
    public float[] prefill(int[] tokens, int startPosition) {
      events.add("prefill:" + tokens[0] + "," + tokens[1] + "@" + startPosition);
      return new float[] {tokens.length};
    }

    @Override
    public void reset() {
      events.add("reset");
    }

    @Override
    public void close() {}
  }
}
