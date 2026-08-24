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
import com.integrallis.models.runtime.GenerationLoop;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("integration")
class Needle2BackendIntegrationTest {

  private static final String WEATHER_TOOLS =
      "[{\"name\":\"get_weather\",\"description\":\"Get the current weather for a city.\","
          + "\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},"
          + "\"required\":[\"city\"]}}]";

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
    ModelPrompt prompt =
        ModelPrompt.builder()
            .control("<|im_start|>")
            .text("user\n")
            .control("<tools>")
            .text(WEATHER_TOOLS)
            .control("</tools>")
            .text("\nweather in Lagos")
            .control("<|im_end|>\n<|im_start|>")
            .text("assistant\n")
            .build();

    try (PureJavaBackend backend = PureJavaBackend.load(path)) {
      String generated =
          new GenerationLoop(backend)
              .generate(prompt, SamplingOptions.builder().temperature(0.0f).maxTokens(64).build());

      assertThat(generated).contains("get_weather").contains("\"city\":\"Lagos\"");
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
