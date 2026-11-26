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

import com.integrallis.models.api.InferenceBackend;
import com.integrallis.models.spring.ai.decision.ModelsSpringAiDecisionModel;
import com.integrallis.models.spring.boot.ModelsAutoConfiguration;
import io.micrometer.observation.ObservationRegistry;
import java.lang.reflect.Proxy;
import java.util.Map;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.springframework.ai.decision.DecisionModel;
import org.springframework.ai.decision.DecisionResponse;
import org.springframework.ai.decision.question.ChoiceQuestion;
import org.springframework.ai.decision.question.YesNoQuestion;
import org.springframework.boot.autoconfigure.AutoConfigurations;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;

@Tag("unit")
class ModelsDecisionAutoConfigurationTest {

  private static final String INPUT = "Ticket: billed twice in March, wants one charge refunded.";

  private final ApplicationContextRunner runner =
      new ApplicationContextRunner()
          .withConfiguration(
              AutoConfigurations.of(
                  ModelsAutoConfiguration.class, ModelsDecisionAutoConfiguration.class));

  private static Map<String, org.springframework.ai.decision.question.DecisionQuestion>
      questions() {
    return Map.of(
        "billedTwice",
        YesNoQuestion.of("Was the customer billed more than once?"),
        "team",
        ChoiceQuestion.builder()
            .text("Which team should handle this ticket?")
            .option("billing", "charges and refunds")
            .option("shipping", "deliveries")
            .build());
  }

  @Test
  void contributesNothingUnlessEnabled() {
    runner
        .withBean(ScriptedPrefixBackend.class, ScriptedPrefixBackend::new)
        .run(context -> assertThat(context).doesNotHaveBean(DecisionModel.class));
    runner
        .withBean(ScriptedPrefixBackend.class, ScriptedPrefixBackend::new)
        .withPropertyValues("integrallis.models.decision.enabled=false")
        .run(context -> assertThat(context).doesNotHaveBean(DecisionModel.class));
  }

  @Test
  void whenEnabledAnswersThroughTheApplicationsBackendExactlyAsTheAdapterDoes() {
    runner
        .withBean(ScriptedPrefixBackend.class, ScriptedPrefixBackend::new)
        .withPropertyValues("integrallis.models.decision.enabled=true")
        .run(
            context -> {
              assertThat(context).hasSingleBean(DecisionModel.class);
              assertThat(context)
                  .getBean(ModelsDecisionAutoConfiguration.MODELS_DECISION_MODEL_BEAN_NAME)
                  .isInstanceOf(ModelsSpringAiDecisionModel.class);
              DecisionResponse viaBoot =
                  context.getBean(DecisionModel.class).call(INPUT, questions());
              DecisionResponse direct =
                  new ModelsSpringAiDecisionModel(new ScriptedPrefixBackend())
                      .call(INPUT, questions());
              assertThat(viaBoot.getYesNo("billedTwice").getProbability())
                  .isEqualTo(direct.getYesNo("billedTwice").getProbability());
              assertThat(viaBoot.getChoice("team").getProbabilities())
                  .isEqualTo(direct.getChoice("team").getProbabilities());
              // The same backend as the chat model: the scripted backend saw the questions.
              assertThat(context.getBean(ScriptedPrefixBackend.class).scoredPrompts).isNotEmpty();
              assertThat(context).hasBean(ModelsAutoConfiguration.MODELS_CHAT_MODEL_BEAN_NAME);
            });
  }

  @Test
  void temperatureAndModelNameComeFromConfiguration() {
    runner
        .withBean(ScriptedPrefixBackend.class, ScriptedPrefixBackend::new)
        .withPropertyValues(
            "integrallis.models.decision.enabled=true",
            "integrallis.models.decision.temperature=4.0",
            "integrallis.models.decision.model-name=support-triage")
        .run(
            context -> {
              DecisionResponse response =
                  context.getBean(DecisionModel.class).call(INPUT, questions());
              DecisionResponse atFour =
                  new ModelsSpringAiDecisionModel(
                          new ScriptedPrefixBackend(),
                          "support-triage",
                          4.0,
                          ObservationRegistry.NOOP)
                      .call(INPUT, questions());
              DecisionResponse atOne =
                  new ModelsSpringAiDecisionModel(new ScriptedPrefixBackend())
                      .call(INPUT, questions());
              assertThat(response.getYesNo("billedTwice").getProbability())
                  .isEqualTo(atFour.getYesNo("billedTwice").getProbability())
                  .isNotEqualTo(atOne.getYesNo("billedTwice").getProbability());
              assertThat(response.getMetadata().getModel()).isEqualTo("support-triage");
            });
  }

  @Test
  void anApplicationDecisionModelWins() {
    DecisionModel own = new ModelsSpringAiDecisionModel(new ScriptedPrefixBackend(), "own");
    runner
        .withBean(ScriptedPrefixBackend.class, ScriptedPrefixBackend::new)
        .withBean("ownDecisionModel", DecisionModel.class, () -> own)
        .withPropertyValues("integrallis.models.decision.enabled=true")
        .run(
            context -> {
              assertThat(context).hasSingleBean(DecisionModel.class);
              assertThat(context.getBean(DecisionModel.class)).isSameAs(own);
              assertThat(context)
                  .doesNotHaveBean(ModelsDecisionAutoConfiguration.MODELS_DECISION_MODEL_BEAN_NAME);
            });
  }

  @Test
  void aNamedBackendIsUsedWhenThereAreSeveral() {
    ScriptedPrefixBackend chosen = new ScriptedPrefixBackend();
    ScriptedPrefixBackend other = new ScriptedPrefixBackend();
    runner
        // The chat model takes the primary backend; decisions are answered by the one named.
        .withBean(
            "chatBackend", ScriptedPrefixBackend.class, () -> other, bd -> bd.setPrimary(true))
        .withBean("judgeBackend", ScriptedPrefixBackend.class, () -> chosen)
        .withPropertyValues(
            "integrallis.models.decision.enabled=true",
            "integrallis.models.decision.backend=judgeBackend")
        .run(
            context -> {
              context.getBean(DecisionModel.class).call(INPUT, questions());
              assertThat(chosen.scoredPrompts).isNotEmpty();
              assertThat(other.scoredPrompts).isEmpty();
            });
  }

  @Test
  void enabledWithNoUsableBackendFailsToStartAndSaysWhy() {
    runner
        .withPropertyValues("integrallis.models.decision.enabled=true")
        .run(
            context ->
                assertThat(context)
                    .hasFailed()
                    .getFailure()
                    .rootCause()
                    .hasMessageContaining("there is no InferenceBackend bean"));
    runner
        .withBean("a", ScriptedPrefixBackend.class, ScriptedPrefixBackend::new)
        .withBean("b", ScriptedPrefixBackend.class, ScriptedPrefixBackend::new)
        .withPropertyValues("integrallis.models.decision.enabled=true")
        .run(
            context ->
                assertThat(context)
                    .hasFailed()
                    .getFailure()
                    .rootCause()
                    .hasMessageContaining("several InferenceBackend beans")
                    .hasMessageContaining("integrallis.models.decision.backend"));
    runner
        .withBean(ScriptedPrefixBackend.class, ScriptedPrefixBackend::new)
        .withPropertyValues(
            "integrallis.models.decision.enabled=true",
            "integrallis.models.decision.backend=missing")
        .run(
            context ->
                assertThat(context)
                    .hasFailed()
                    .getFailure()
                    .rootCause()
                    .hasMessageContaining("names missing, but there is no such bean"));
    InferenceBackend plain =
        (InferenceBackend)
            Proxy.newProxyInstance(
                getClass().getClassLoader(),
                new Class<?>[] {InferenceBackend.class},
                (proxy, method, args) ->
                    switch (method.getName()) {
                      case "name" -> "plain";
                      case "toString" -> "plain";
                      case "hashCode" -> 7;
                      case "equals" -> proxy == args[0];
                      default -> throw new UnsupportedOperationException(method.getName());
                    });
    // Only the decision configuration here: the chat model would refuse this backend first.
    new ApplicationContextRunner()
        .withConfiguration(AutoConfigurations.of(ModelsDecisionAutoConfiguration.class))
        .withBean(InferenceBackend.class, () -> plain)
        .withPropertyValues("integrallis.models.decision.enabled=true")
        .run(
            context ->
                assertThat(context)
                    .hasFailed()
                    .getFailure()
                    .rootCause()
                    .hasMessageContaining("is not a SharedPrefixInferenceBackend"));
  }

  @Test
  void aTemperatureTheAdapterCannotUseFailsAtBinding() {
    runner
        .withBean(ScriptedPrefixBackend.class, ScriptedPrefixBackend::new)
        .withPropertyValues(
            "integrallis.models.decision.enabled=true", "integrallis.models.decision.temperature=0")
        .run(
            context ->
                assertThat(context)
                    .hasFailed()
                    .getFailure()
                    .rootCause()
                    .hasMessageContaining("must be positive and finite"));
  }
}
