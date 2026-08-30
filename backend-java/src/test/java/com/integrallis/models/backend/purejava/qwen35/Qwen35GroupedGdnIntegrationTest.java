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

import com.integrallis.models.backend.purejava.fixture.ModelFixtureDescriptor;
import com.integrallis.models.backend.purejava.fixture.ModelFixtureRegistry;
import com.integrallis.models.backend.purejava.fixture.ModelFixtureRequirement;
import com.integrallis.models.backend.purejava.gguf.GgufParser;
import com.integrallis.models.backend.purejava.tokenizer.GgufTokenizer;
import java.lang.foreign.Arena;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/** Real-model gate for Qwen3.5 grouped Gated DeltaNet heads. */
@Tag("integration")
class Qwen35GroupedGdnIntegrationTest {

  private static final ModelFixtureRequirement QWEN35_4B_Q4_K_M =
      ModelFixtureRequirement.of("hf://unsloth/Qwen3.5-4B-GGUF")
          .version("[3.5.0,3.6.0)")
          .variant("q4_k_m")
          .backend("pure-java")
          .capability("chat");

  @Test
  void matchesPinnedLlamaCppTokensWithGroupedGdnHeads() throws Exception {
    ModelFixtureDescriptor descriptor =
        ModelFixtureRegistry.fromClasspath().resolve(QWEN35_4B_Q4_K_M).orElseThrow();

    try (Arena arena = Arena.ofConfined()) {
      var file = GgufParser.parse(descriptor.localPath().orElseThrow(), arena);
      Qwen35Config config = Qwen35Config.fromMetadata(file.metadata());
      assertThat(config.gdnKeyHeads()).isEqualTo(16);
      assertThat(config.gdnValueHeads()).isEqualTo(32);

      var tokenizer = GgufTokenizer.fromMetadata(file.metadata());
      assertThat(tokenizer.encode("The")).containsExactly(760);
      float[] logits = Qwen35ForwardPass.fromGgufFile(file).forwardFresh(760);

      assertThat(argmax(logits))
          .as("llama.cpp a58222229 produces token 2614 (' following') from fresh token 760")
          .isEqualTo(2_614);

      int[] prompt = tokenizer.encode("The quick brown fox");
      assertThat(prompt).containsExactly(760, 3_841, 13_477, 37_550);
      logits = Qwen35ForwardPass.fromGgufFile(file).forward(prompt);

      assertThat(argmax(logits))
          .as(
              "llama.cpp a58222229 produces token 33075 (' jumps') after the four-token prompt "
                  + "'The quick brown fox'")
          .isEqualTo(33_075);
    }
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
