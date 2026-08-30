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
package com.integrallis.models.backend.purejava.qwen35;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.models.backend.purejava.PureJavaBackend;
import com.integrallis.models.backend.purejava.fixture.ModelFixtureDescriptor;
import com.integrallis.models.backend.purejava.fixture.ModelFixtureRegistry;
import com.integrallis.models.backend.purejava.fixture.ModelFixtureRequirement;
import com.integrallis.models.backend.purejava.gguf.GgufParser;
import com.integrallis.models.backend.purejava.tokenizer.GgufTokenizer;
import java.lang.foreign.Arena;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/** Real-model compatibility gate for the first pure-Java Qwen3.5 dense graph. */
@Tag("integration")
class Qwen35ForwardPassIntegrationTest {

  private static final ModelFixtureRequirement QWEN35_08B_Q4_K_M =
      ModelFixtureRequirement.of("hf://unsloth/Qwen3.5-0.8B-GGUF")
          .version("[3.5.0,3.6.0)")
          .variant("q4_k_m")
          .backend("pure-java")
          .capability("chat");

  @Test
  void matchesThePinnedLlamaCppFirstGreedyToken() throws Exception {
    ModelFixtureDescriptor descriptor =
        ModelFixtureRegistry.fromClasspath().resolve(QWEN35_08B_Q4_K_M).orElseThrow();

    try (Arena arena = Arena.ofConfined()) {
      var file = GgufParser.parse(descriptor.localPath().orElseThrow(), arena);
      var tokenizer = GgufTokenizer.fromMetadata(file.metadata());
      assertThat(tokenizer.encode("The")).containsExactly(760);

      float[] logits = Qwen35ForwardPass.fromGgufFile(file).forwardFresh(760);

      assertThat(logits).hasSize(248_320);
      assertThat(allFinite(logits)).isTrue();
      assertThat(argmax(logits))
          .as("llama.cpp a58222229 produces token 2614 (' following') from fresh token 760")
          .isEqualTo(2_614);
    }
  }

  @Test
  void matchesThePinnedLlamaCppPromptGreedyToken() throws Exception {
    ModelFixtureDescriptor descriptor =
        ModelFixtureRegistry.fromClasspath().resolve(QWEN35_08B_Q4_K_M).orElseThrow();

    try (Arena arena = Arena.ofConfined()) {
      var file = GgufParser.parse(descriptor.localPath().orElseThrow(), arena);
      var tokenizer = GgufTokenizer.fromMetadata(file.metadata());
      int[] prompt = tokenizer.encode("The quick brown fox");
      assertThat(prompt).containsExactly(760, 3_841, 13_477, 37_550);

      float[] logits = Qwen35ForwardPass.fromGgufFile(file).forward(prompt);

      assertThat(allFinite(logits)).isTrue();
      assertThat(argmax(logits))
          .as(
              "llama.cpp a58222229 produces token 321 (' and') after the four-token prompt "
                  + "'The quick brown fox'")
          .isEqualTo(321);
    }
  }

  @Test
  void generatesThroughThePublicBackendContract() {
    ModelFixtureDescriptor descriptor =
        ModelFixtureRegistry.fromClasspath().resolve(QWEN35_08B_Q4_K_M).orElseThrow();
    String previous = System.getProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY);
    System.setProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY, "32");

    try (PureJavaBackend backend = PureJavaBackend.load(descriptor.localPath().orElseThrow())) {
      assertThat(backend.metadata().modelFamily()).isEqualTo("qwen35");
      assertThat(backend.contextCapacity()).isEqualTo(32);
      int[] prompt = backend.tokenizer().encode("The quick brown fox");

      float[] logits = backend.prefill(prompt, 0);
      assertThat(argmax(logits)).isEqualTo(321);
      assertThat(backend.checkpoint()).isEqualTo(4);
      assertThat(argmax(backend.forward(321, 4)))
          .as("llama.cpp a58222229 continues with token 279 (' the')")
          .isEqualTo(279);

      backend.rewind(2);
      assertThat(backend.checkpoint()).isEqualTo(2);
      assertThat(argmax(backend.prefill(new int[] {13_477, 37_550}, 2))).isEqualTo(321);

      try (var session = backend.openSession()) {
        assertThat(argmax(backend.prefill(session, prompt, 0))).isEqualTo(321);
        assertThat(session.checkpoint()).isEqualTo(4);
      }
    } finally {
      if (previous == null) {
        System.clearProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY);
      } else {
        System.setProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY, previous);
      }
    }
  }

  private static boolean allFinite(float[] values) {
    for (float value : values) {
      if (!Float.isFinite(value)) {
        return false;
      }
    }
    return true;
  }

  private static int argmax(float[] values) {
    int result = 0;
    for (int index = 1; index < values.length; index++) {
      if (values[index] > values[result]) {
        result = index;
      }
    }
    return result;
  }
}
