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
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.tool.annotation.Tool;
import org.springframework.ai.tool.annotation.ToolParam;

/** End-to-end reproduction of the first GPT-OSS Spring AI user report. */
@Tag("integration")
class GptOssOfficialToolCallingIntegrationTest {

  private record Weather(String zipcode, String conditions, int temperature) {}

  private static final class WeatherTools {
    private final AtomicInteger invocations = new AtomicInteger();

    @Tool(name = "get-weather-for-zipcode", description = "Gets weather for a given zipcode")
    Weather getWeatherForZipcode(
        @ToolParam(description = "The zipcode to get weather for") String zipcode) {
      invocations.incrementAndGet();
      return new Weather(zipcode, "Raining cats and dogs", 78);
    }
  }

  @Test
  void invokesTheHyphenatedWeatherToolAndCompletesTheConversation() {
    Path checkpoint = fixtureDirectory();
    String previous = System.getProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY);
    System.setProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY, "512");
    try (PureJavaBackend backend = PureJavaBackend.load(checkpoint)) {
      var runtime = new RuntimeTextGenerationModel(backend);
      var adapter =
          new ModelsSpringAiChatModel(
              runtime,
              "gpt-oss-20b",
              ChatTemplate.GPT_OSS,
              SamplingOptions.builder().temperature(0.0f).maxTokens(128).build(),
              Set.of("chat", "text-generation", "tool-calling"));
      WeatherTools weatherTools = new WeatherTools();
      ChatClient client = ChatClient.builder(adapter).defaultTools(weatherTools).build();

      String answer = client.prompt().user("What is the weather for 88252?").call().content();

      assertThat(weatherTools.invocations).hasValue(1);
      assertThat(answer).contains("88252", "78");
      assertThat(runtime.lastGenerationMetrics().usage().completionTokens()).isPositive();
    } finally {
      if (previous == null) {
        System.clearProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY);
      } else {
        System.setProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY, previous);
      }
    }
  }

  private static Path fixtureDirectory() {
    String configured = System.getProperty("models.fixtures.gptOssHuggingFaceDirectory", "");
    assumeTrue(!configured.isBlank(), "set models.fixtures.gptOssHuggingFaceDirectory");
    Path directory = Path.of(configured).toAbsolutePath().normalize();
    assumeTrue(Files.isDirectory(directory), "GPT-OSS Hugging Face fixture is not installed");
    return directory;
  }
}
