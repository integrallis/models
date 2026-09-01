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
package com.integrallis.models.spring.ai;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.integrallis.models.api.SamplingOptions;
import com.integrallis.models.backend.purejava.PureJavaBackend;
import com.integrallis.models.runtime.RuntimeTextGenerationModel;
import com.integrallis.models.runtime.chat.ChatTemplate;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.tool.annotation.Tool;
import org.springframework.ai.tool.annotation.ToolParam;

/** End-to-end reproduction of the first Needle 2 Spring AI user report. */
@Tag("integration")
class Needle2SpringAiToolCallingIntegrationTest {

  private record Weather(String zipcode, String conditions, int temperature) {}

  private static final class WeatherTools {
    private final AtomicInteger invocations = new AtomicInteger();
    private final AtomicReference<String> lastZipcode = new AtomicReference<>();

    @Tool(name = "get-weather-for-zipcode", description = "Gets weather for a given zipcode")
    Weather getWeatherForZipcode(
        @ToolParam(description = "The zipcode to get weather for") String zipcode) {
      invocations.incrementAndGet();
      lastZipcode.set(zipcode);
      return new Weather(zipcode, "Raining cats and dogs", 78);
    }
  }

  @Test
  void invokesTheUsersHyphenatedZipcodeToolWithTheExactArgument() {
    String configured = System.getProperty("models.fixtures.needle2Cact", "");
    assumeTrue(!configured.isBlank(), "set models.fixtures.needle2Cact");
    Path artifact = Path.of(configured).toAbsolutePath().normalize();
    assumeTrue(Files.isRegularFile(artifact), "Needle 2 fixture is not installed");

    try (PureJavaBackend backend = PureJavaBackend.load(artifact)) {
      var runtime = new RuntimeTextGenerationModel(backend);
      var adapter =
          new ModelsSpringAiChatModel(
              runtime,
              "needle2",
              ChatTemplate.NEEDLE2,
              SamplingOptions.builder().maxTokens(128).build(),
              Set.of("chat", "text-generation", "tool-calling"));
      WeatherTools weatherTools = new WeatherTools();
      ChatClient client = ChatClient.builder(adapter).defaultTools(weatherTools).build();

      String answer = client.prompt().user("What is the weather for 88252?").call().content();

      assertThat(weatherTools.invocations).as("model answer: %s", answer).hasValue(1);
      assertThat(weatherTools.lastZipcode).hasValue("88252");
      assertThat(answer)
          .isEqualTo(
              "{\"zipcode\":\"88252\",\"conditions\":\"Raining cats and dogs\","
                  + "\"temperature\":78}");
    }
  }
}
