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
import com.integrallis.models.runtime.ActivatedToolCallingModel;
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

/** Real LangChain4j loop over the base-aligned Qwen3 activated adapter. */
@Tag("activated-integration")
class Qwen3ActivatedLangChain4jToolCallingIntegrationTest {

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
  void invokesOneToolNarratesItsResultAndAbstainsOnOrdinaryChat() {
    Path modelPath = fixture("models.fixtures.qwen317b", false);
    Path adapterPath = fixture("models.fixtures.activatedLoraDirectory", true);
    try (ModelsChatModel adapter =
        new ModelsChatModel(
            new ActivatedToolCallingModel(
                PureJavaBackend.loadActivatedAdapter(modelPath, adapterPath), 1),
            ChatTemplate.CHATML_NO_THINK,
            SamplingOptions.builder().temperature(0.0f).maxTokens(128).build())) {
      WeatherTools weatherTools = new WeatherTools();
      Assistant assistant =
          AiServices.builder(Assistant.class).chatModel(adapter).tools(weatherTools).build();

      String weather = assistant.chat("What is the weather for 88252?");
      String greeting = assistant.chat("Reply with exactly: READY");

      assertThat(weatherTools.invocations)
          .as("weather=%s greeting=%s", weather, greeting)
          .hasValue(1);
      assertThat(weatherTools.lastZipcode).hasValue("88252");
      assertThat(weather).contains("78").containsIgnoringCase("rain");
      assertThat(greeting).containsIgnoringCase("READY");
      assertThat(weather).doesNotContain("<tool_call>", "<think>");
      assertThat(greeting).doesNotContain("<tool_call>", "<think>");
    }
  }

  private static Path fixture(String property, boolean directory) {
    String configured = System.getProperty(property, "");
    assumeTrue(!configured.isBlank(), "set " + property);
    Path path = Path.of(configured).toAbsolutePath().normalize();
    assumeTrue(
        directory ? Files.isDirectory(path) : Files.isRegularFile(path),
        "fixture is not installed: " + path);
    return path;
  }
}
