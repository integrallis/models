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
package com.integrallis.models.langchain4j;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.integrallis.models.api.SamplingOptions;
import com.integrallis.models.backend.purejava.PureJavaBackend;
import com.integrallis.models.runtime.RuntimeTextGenerationModel;
import com.integrallis.models.runtime.chat.ChatTemplate;
import dev.langchain4j.agent.tool.P;
import dev.langchain4j.agent.tool.Tool;
import dev.langchain4j.service.AiServices;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/** Real-weight LangChain4j tool round trip through Qwen3 1.7B and the pure-Java backend. */
@Tag("integration")
class Qwen3LangChain4jToolCallingIntegrationTest {

  interface Assistant {
    String chat(String message);
  }

  record Weather(String zipcode, String conditions, int temperatureInFahrenheit) {}

  static final class WeatherTools {
    private final AtomicInteger invocations = new AtomicInteger();
    private final AtomicReference<String> lastZipcode = new AtomicReference<>();

    @Tool(name = "get-weather-for-zipcode", value = "Gets weather for a given zipcode")
    Weather getWeatherForZipcode(@P("The zipcode to get weather for") String zipcode) {
      invocations.incrementAndGet();
      lastZipcode.set(zipcode);
      return new Weather(zipcode, "Raining cats and dogs", 78);
    }
  }

  @Test
  void invokesTheToolAndSynthesizesTheResult() {
    Path artifact = fixture();
    try (PureJavaBackend backend = PureJavaBackend.load(artifact)) {
      var adapter =
          new ModelsChatModel(
              new RuntimeTextGenerationModel(backend),
              ChatTemplate.CHATML_NO_THINK,
              SamplingOptions.builder().temperature(0.0f).maxTokens(128).build());
      WeatherTools weatherTools = new WeatherTools();
      Assistant assistant =
          AiServices.builder(Assistant.class).chatModel(adapter).tools(weatherTools).build();

      String answer = assistant.chat("What is the weather for 88252?");

      assertThat(weatherTools.invocations).as("model answer: %s", answer).hasValue(1);
      assertThat(weatherTools.lastZipcode).hasValue("88252");
      assertThat(answer).contains("78").containsIgnoringCase("rain");
      assertThat(answer).doesNotContain("<tool_call>", "<think>");
    }
  }

  private static Path fixture() {
    String configured = System.getProperty("models.fixtures.qwen317b", "");
    assumeTrue(!configured.isBlank(), "set models.fixtures.qwen317b");
    Path artifact = Path.of(configured).toAbsolutePath().normalize();
    assumeTrue(Files.isRegularFile(artifact), "Qwen3 1.7B fixture is not installed");
    return artifact;
  }
}
