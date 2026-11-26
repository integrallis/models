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
package com.integrallis.models.spring.boot.decision;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.integrallis.models.api.InferenceBackend;
import com.integrallis.models.backend.purejava.PureJavaBackend;
import com.integrallis.models.spring.ai.decision.ModelsSpringAiDecisionModel;
import com.integrallis.models.spring.boot.ModelsAutoConfiguration;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.decision.DecisionModel;
import org.springframework.ai.decision.DecisionResponse;
import org.springframework.ai.decision.question.ChoiceQuestion;
import org.springframework.ai.decision.question.DecisionQuestion;
import org.springframework.ai.decision.question.YesNoQuestion;
import org.springframework.boot.autoconfigure.AutoConfigurations;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;

/**
 * The auto-configured DecisionModel over a real model, through Spring Boot: the application
 * declares one backend, sets one property, and gets a chat model and a decision model over it.
 *
 * <p>Not an accuracy measurement. It checks the wiring end to end on real weights and that the
 * answer through Boot is the adapter's answer on the same backend. Skipped without the fixture.
 */
@Tag("integration")
class RealModelDecisionAutoConfigurationIntegrationTest {

  private static final String INPUT =
      "Support ticket: I was billed twice for my March subscription and I want one charge"
          + " refunded today.";

  private static final String MODEL_FILE =
      System.getProperty("models.fixtures.decisionModel", "Qwen3-1.7B-Q8_0.gguf");

  private static Path fixture() {
    return Path.of(
        System.getProperty(
            "models.fixtures.directory",
            Path.of(System.getProperty("user.home"), ".jvllm", "models").toString()),
        MODEL_FILE);
  }

  @Test
  void oneBackendOnePropertyGivesAChatModelAndADecisionModelThatAgreesWithTheAdapter()
      throws Exception {
    Path model = fixture();
    assumeTrue(Files.isRegularFile(model), "no model fixture at " + model);
    try (PureJavaBackend backend = PureJavaBackend.load(model)) {
      Map<String, DecisionQuestion> questions = new LinkedHashMap<>();
      questions.put(
          "team",
          ChoiceQuestion.builder()
              .text("Which team should handle this ticket?")
              .option("billing", "charges, refunds and invoices")
              .option("shipping", "delivery and tracking of physical goods")
              .option("technical", "bugs, outages and login problems")
              .build());
      questions.put("billedTwice", YesNoQuestion.of("Was the customer billed more than once?"));
      DecisionResponse direct =
          new ModelsSpringAiDecisionModel(backend, MODEL_FILE).call(INPUT, questions);
      new ApplicationContextRunner()
          .withConfiguration(
              AutoConfigurations.of(
                  ModelsAutoConfiguration.class, ModelsDecisionAutoConfiguration.class))
          .withBean(InferenceBackend.class, () -> backend)
          .withPropertyValues("integrallis.models.decision.enabled=true")
          .run(
              context -> {
                assertThat(context).hasSingleBean(ChatModel.class);
                assertThat(context).hasSingleBean(DecisionModel.class);
                long start = System.nanoTime();
                DecisionResponse viaBoot =
                    context.getBean(DecisionModel.class).call(INPUT, questions);
                System.out.printf(
                    "REAL-BOOT %s team=%s %s p(yes|billed twice)=%.4f prefills=%s %.0f ms%n",
                    MODEL_FILE,
                    viaBoot.getChoice("team").getValue(),
                    viaBoot.getChoice("team").getProbabilities(),
                    viaBoot.getYesNo("billedTwice").getProbability(),
                    viaBoot.getMetadata().get(ModelsSpringAiDecisionModel.STATE_PREFILLS),
                    (System.nanoTime() - start) / 1e6);
                assertThat(viaBoot.getChoice("team").getValue()).isEqualTo("billing");
                assertThat(viaBoot.getChoice("team").getProbabilities())
                    .isEqualTo(direct.getChoice("team").getProbabilities());
                assertThat(viaBoot.getYesNo("billedTwice").getProbability())
                    .isEqualTo(direct.getYesNo("billedTwice").getProbability());
              });
    }
  }
}
