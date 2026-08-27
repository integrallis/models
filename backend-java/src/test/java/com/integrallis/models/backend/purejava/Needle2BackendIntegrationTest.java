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
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.integrallis.models.api.ModelPrompt;
import com.integrallis.models.api.SamplingOptions;
import com.integrallis.models.api.ToolSpec;
import com.integrallis.models.runtime.GenerationLoop;
import com.integrallis.models.runtime.chat.ChatMessage;
import com.integrallis.models.runtime.chat.ChatTemplate;
import com.integrallis.models.runtime.chat.ToolCallScanner;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("integration")
class Needle2BackendIntegrationTest {

  @Test
  void loadsCactThroughThePublicBackendContractWhenProvided() throws IOException {
    String configured = System.getProperty("models.fixtures.needle2Cact", "");
    assumeTrue(!configured.isBlank(), "set -Dmodels.fixtures.needle2Cact=<needle2.cact>");
    Path path = Path.of(configured);
    assumeTrue(Files.isRegularFile(path), "Needle 2 fixture is not installed");

    try (PureJavaBackend backend = PureJavaBackend.load(path)) {
      assertThat(backend.metadata().modelFamily()).isEqualTo("needle2");
      assertThat(backend.metadata().vocabSize()).isEqualTo(8192);
      assertThat(backend.contextCapacity()).isEqualTo(2048);
      assertThat(backend.tokenizer().encode("Name one JVM language.")).startsWith(2, 449, 471);
      assertThat(backend.diagnostics().environment())
          .containsEntry("artifact-format", "cact")
          .containsEntry("weight-encoding", "rotated-codebook");
      assertThat(backend.diagnostics().optimization("cact-rotated-codebook")).isPresent();

      assertThat(argmax(backend.prefill(new int[] {2}, 0))).isEqualTo(2);
      assertThat(backend.checkpoint()).isEqualTo(1);
      backend.rewind(0);
      assertThat(backend.checkpoint()).isZero();

      try (var session = backend.openSession()) {
        assertThat(argmax(backend.prefill(session, new int[] {2}, 0))).isEqualTo(2);
        assertThat(session.checkpoint()).isEqualTo(1);
      }
    }
  }

  @Test
  void generatesTheOfficialWeatherToolCallWhenProvided() throws IOException {
    String configured = System.getProperty("models.fixtures.needle2Cact", "");
    assumeTrue(!configured.isBlank(), "set -Dmodels.fixtures.needle2Cact=<needle2.cact>");
    Path path = Path.of(configured);
    assumeTrue(Files.isRegularFile(path), "Needle 2 fixture is not installed");
    ToolSpec weather =
        new ToolSpec(
            "get_weather",
            "Get the current weather for a city.",
            "{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},"
                + "\"required\":[\"city\"]}");
    ModelPrompt prompt =
        ChatTemplate.NEEDLE2.render(
            List.of(ChatMessage.user("weather in Lagos")), List.of(weather));

    try (PureJavaBackend backend = PureJavaBackend.load(path)) {
      String generated =
          new GenerationLoop(backend)
              .generate(prompt, SamplingOptions.builder().temperature(0.0f).maxTokens(64).build());

      ToolCallScanner.Result result =
          ToolCallScanner.scan(generated, ChatTemplate.NEEDLE2.toolSyntax());
      assertThat(result.toolCalls())
          .singleElement()
          .satisfies(
              call -> {
                assertThat(call.name()).isEqualTo("get_weather");
                assertThat(call.argumentsJson()).isEqualTo("{\"city\":\"Lagos\"}");
              });
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
