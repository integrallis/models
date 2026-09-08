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
package com.integrallis.models.backend.purejava;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.models.api.ModelPrompt;
import com.integrallis.models.api.ToolSpec;
import com.integrallis.models.runtime.chat.ChatMessage;
import com.integrallis.models.runtime.chat.ChatTemplate;
import com.integrallis.models.runtime.chat.ToolCallScanner;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;

/** Real-weight regression for GGUF YaRN semantics and the Hammer 2.1 tool protocol. */
@Tag("integration")
@EnabledIfSystemProperty(named = HammerYarnModelIntegrationTest.FIXTURE_PROPERTY, matches = ".+")
class HammerYarnModelIntegrationTest {

  static final String FIXTURE_PROPERTY = "models.fixtures.hammer21Gguf";

  private static final ToolSpec WEATHER =
      new ToolSpec(
          "get-weather-for-zipcode",
          "Gets weather for a given zipcode",
          "{\"type\":\"object\",\"properties\":{\"zipcode\":{\"type\":\"string\","
              + "\"description\":\"The zipcode to get weather for\"}},\"required\":[\"zipcode\"]}");

  // Pinned against llama.cpp b9960 using the exact prompt rendered below and the artifact digest
  // 190676fe7ac430ac6680b32c216d4eb7413002e2af0c140b8280a0f74fcc6a4d.
  private static final int[] EXPECTED_GREEDY_TOKENS = {
    13874, 3989, 58, 13608, 606, 1210, 364, 455, 12, 15206,
    15193, 9141, 573, 1851, 516, 364, 16370, 1210, 5360, 59629,
    1210, 364, 23, 23, 17, 20, 17, 22892, 921, 73594
  };

  @Test
  void matchesThePinnedHammerToolCallThroughYarnScaledInference() {
    Path model = Path.of(System.getProperty(FIXTURE_PROPERTY));
    ModelPrompt prompt =
        ChatTemplate.HAMMER.render(
            List.of(ChatMessage.user("What is the weather for 88252?")), List.of(WEATHER));
    String previous = System.getProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY);
    System.setProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY, "512");

    try (PureJavaBackend backend = PureJavaBackend.load(model)) {
      int[] promptTokens = backend.tokenizer().encode(prompt);
      assertThat(promptTokens).hasSize(379);

      int[] generated = greedyTokens(backend, promptTokens, EXPECTED_GREEDY_TOKENS.length);

      assertThat(generated)
          .as("greedy token IDs must match llama.cpp for the pinned YaRN GGUF")
          .containsExactly(EXPECTED_GREEDY_TOKENS);
      ToolCallScanner.Result scanned =
          ToolCallScanner.scan(
              backend.tokenizer().decode(generated), ChatTemplate.HAMMER.toolSyntax());
      assertThat(scanned.toolCalls())
          .singleElement()
          .satisfies(
              call -> {
                assertThat(call.name()).isEqualTo("get-weather-for-zipcode");
                assertThat(call.argumentsJson()).isEqualTo("{\"zipcode\": \"88252\"}");
              });
    } finally {
      restoreSystemProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY, previous);
    }
  }

  private static int[] greedyTokens(PureJavaBackend backend, int[] promptTokens, int count) {
    backend.reset();
    float[] logits = backend.prefill(promptTokens, 0);
    int position = promptTokens.length;
    int[] generated = new int[count];
    for (int index = 0; index < count; index++) {
      int token = argmax(logits);
      generated[index] = token;
      logits = backend.forward(token, position++);
    }
    return generated;
  }

  private static int argmax(float[] values) {
    int best = 0;
    for (int index = 1; index < values.length; index++) {
      if (values[index] > values[best]) {
        best = index;
      }
    }
    return best;
  }

  private static void restoreSystemProperty(String name, String previous) {
    if (previous == null) {
      System.clearProperty(name);
    } else {
      System.setProperty(name, previous);
    }
  }
}
