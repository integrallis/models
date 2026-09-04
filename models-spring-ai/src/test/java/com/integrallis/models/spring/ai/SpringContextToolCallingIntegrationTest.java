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
import static org.junit.jupiter.api.Assertions.assertTimeoutPreemptively;

import com.integrallis.models.api.BackendDiagnostics;
import com.integrallis.models.api.ModelPrompt;
import com.integrallis.models.api.SamplingOptions;
import com.integrallis.models.api.TextGenerationModel;
import com.integrallis.models.api.TokenStream;
import com.integrallis.models.runtime.chat.ChatTemplate;
import java.time.Duration;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.client.advisor.MessageChatMemoryAdvisor;
import org.springframework.ai.chat.memory.ChatMemory;
import org.springframework.ai.chat.memory.MessageWindowChatMemory;
import org.springframework.ai.tool.annotation.Tool;
import org.springframework.ai.tool.annotation.ToolParam;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.context.annotation.AnnotationConfigApplicationContext;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * Exercises Craig Walls' ChatClient, customizer, memory, and tool wiring in a real Spring context.
 */
@Tag("unit")
class SpringContextToolCallingIntegrationTest {

  @Test
  void contextStartsAndQwenToolRoundTripCompletes() {
    try (var context = new AnnotationConfigApplicationContext(TestConfiguration.class)) {
      String answer =
          assertTimeoutPreemptively(
              Duration.ofSeconds(5),
              () ->
                  context
                      .getBean(ChatClient.class)
                      .prompt("What is the weather in 88252?")
                      .advisors(spec -> spec.param(ChatMemory.CONVERSATION_ID, "DEMO"))
                      .call()
                      .content());

      assertThat(answer).isEqualTo("It is raining cats and dogs and 78 degrees in 88252.");
      assertThat(context.getBean(WeatherTools.class).invocations).hasValue(1);
      assertThat(context.getBean(ScriptedQwenModel.class).prompts())
          .hasSize(2)
          .last()
          .asString()
          .contains("88252", "Raining cats and dogs", "78");
    }
  }

  @Configuration(proxyBeanMethods = false)
  static class TestConfiguration {

    @Bean
    ScriptedQwenModel modelRuntime() {
      return new ScriptedQwenModel(
          "<tool_call>{\"name\":\"get-weather-for-zipcode\",\"arguments\":{\"zipcode\":\"88252\"}}</tool_call><|im_end|>",
          "It is raining cats and dogs and 78 degrees in 88252.");
    }

    @Bean
    ModelsSpringAiChatModel model(ScriptedQwenModel runtime) {
      return new ModelsSpringAiChatModel(
          runtime,
          "qwen3-1.7b",
          ChatTemplate.CHATML_NO_THINK,
          SamplingOptions.builder().temperature(0).maxTokens(128).build(),
          Set.of("chat", "text-generation", "tool-calling"));
    }

    @Bean
    ChatClient.Builder chatClientBuilder(
        ModelsSpringAiChatModel model, ObjectProvider<TestChatClientCustomizer> customizers) {
      ChatClient.Builder builder = ChatClient.builder(model);
      customizers.orderedStream().forEach(customizer -> customizer.customize(builder));
      return builder;
    }

    @Bean
    ChatClient chatClient(ChatClient.Builder builder) {
      return builder.build();
    }

    @Bean
    TestChatClientCustomizer chatMemoryCustomizer() {
      return builder ->
          builder.defaultAdvisors(
              MessageChatMemoryAdvisor.builder(
                      MessageWindowChatMemory.builder().maxMessages(500).build())
                  .build());
    }

    @Bean
    TestChatClientCustomizer toolCustomizer(WeatherTools weatherTools) {
      return builder -> builder.defaultTools(weatherTools);
    }

    @Bean
    WeatherTools weatherTools() {
      return new WeatherTools();
    }
  }

  @FunctionalInterface
  interface TestChatClientCustomizer {
    void customize(ChatClient.Builder builder);
  }

  static final class ScriptedQwenModel implements TextGenerationModel {
    private final ArrayDeque<String> completions;
    private final List<String> prompts = new ArrayList<>();

    ScriptedQwenModel(String... completions) {
      this.completions = new ArrayDeque<>(List.of(completions));
    }

    @Override
    public String modelName() {
      return "scripted-qwen";
    }

    @Override
    public BackendDiagnostics diagnostics() {
      return BackendDiagnostics.unavailable("spring-context-integration-test");
    }

    @Override
    public void generate(ModelPrompt prompt, SamplingOptions options, TokenStream stream) {
      prompts.add(prompt.text());
      stream.onToken(completions.removeFirst());
      stream.onComplete();
    }

    @Override
    public void generate(String prompt, SamplingOptions options, TokenStream stream) {
      prompts.add(prompt);
      stream.onToken(completions.removeFirst());
      stream.onComplete();
    }

    List<String> prompts() {
      return List.copyOf(prompts);
    }
  }

  record Weather(String zipcode, String conditions, int temperatureInFahrenheit) {}

  static final class WeatherTools {
    private final AtomicInteger invocations = new AtomicInteger();

    @Tool(name = "get-weather-for-zipcode", description = "Gets weather for a given zipcode")
    Weather getWeatherForZipcode(
        @ToolParam(description = "The zipcode to get weather for") String zipcode) {
      invocations.incrementAndGet();
      return new Weather(zipcode, "Raining cats and dogs", 78);
    }
  }
}
