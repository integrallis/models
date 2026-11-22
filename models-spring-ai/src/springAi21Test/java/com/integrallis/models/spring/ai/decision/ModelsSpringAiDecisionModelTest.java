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
package com.integrallis.models.spring.ai.decision;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

import com.integrallis.models.decisions.AnswerSpace;
import com.integrallis.models.decisions.Choice;
import com.integrallis.models.decisions.LetterDecisionEvaluator;
import com.integrallis.models.decisions.Noul;
import com.integrallis.models.decisions.Score;
import io.micrometer.observation.Observation;
import io.micrometer.observation.ObservationHandler;
import io.micrometer.observation.ObservationRegistry;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.content.Media;
import org.springframework.ai.decision.DecisionModel;
import org.springframework.ai.decision.DecisionPrompt;
import org.springframework.ai.decision.DecisionResponse;
import org.springframework.ai.decision.answer.ChoiceAnswer;
import org.springframework.ai.decision.answer.ScoreAnswer;
import org.springframework.ai.decision.observation.DecisionModelObservationContext;
import org.springframework.ai.decision.question.ChoiceQuestion;
import org.springframework.ai.decision.question.DecisionQuestion;
import org.springframework.ai.decision.question.ScoreQuestion;
import org.springframework.ai.decision.question.YesNoQuestion;

/**
 * The adapter through Spring AI's own types: every question type, every input type, read back
 * through {@link DecisionResponse}'s typed accessors, which reject an answer of the wrong kind.
 * Expected probabilities are computed here from the prompt the model should have been shown, not by
 * the code under test.
 */
@Tag("unit")
class ModelsSpringAiDecisionModelTest {

  private static final String STATE = "The invoice was paid twice.";

  private final ScriptedPrefixBackend backend = new ScriptedPrefixBackend();
  private final DecisionModel model = new ModelsSpringAiDecisionModel(backend, "scripted-1");

  /** Softmax over the answer letters for the prompt {@code state + suffix(space)}. */
  private static double[] expected(String state, AnswerSpace space) {
    float[] logits =
        ScriptedPrefixBackend.logitsAfter(state + LetterDecisionEvaluator.questionSuffix(space));
    double[] weights = new double[space.size()];
    double total = 0;
    for (int index = 0; index < weights.length; index++) {
      weights[index] = Math.exp(logits['A' + index]);
      total += weights[index];
    }
    for (int index = 0; index < weights.length; index++) {
      weights[index] /= total;
    }
    return weights;
  }

  private static int argmax(double[] values) {
    int best = 0;
    for (int index = 1; index < values.length; index++) {
      if (values[index] > values[best]) {
        best = index;
      }
    }
    return best;
  }

  @Test
  void aYesNoQuestionIsAnsweredWithTheProbabilityOfYes() {
    DecisionResponse response =
        model.call(STATE, Map.of("dup", YesNoQuestion.of("Was the customer charged twice?")));

    double[] expected = expected(STATE, new Noul("Was the customer charged twice?"));
    assertThat(response.getYesNo("dup").getProbability()).isCloseTo(expected[1], within(1e-12));
  }

  @Test
  void yesNoCriteriaReachTheModelAsTheRubric() {
    YesNoQuestion question =
        YesNoQuestion.builder()
            .text("Was the customer charged twice?")
            .whenYes("two or more charges for one order")
            .whenNo("a single charge")
            .build();

    DecisionResponse response = model.call(STATE, Map.of("dup", question));

    Noul space =
        new Noul(
            "Was the customer charged twice?",
            "two or more charges for one order",
            "a single charge");
    assertThat(backend.scoredPrompts.get(0))
        .contains("- true: two or more charges for one order")
        .contains("- false: a single charge");
    assertThat(response.getYesNo("dup").getProbability())
        .isCloseTo(expected(STATE, space)[1], within(1e-12));
  }

  @Test
  void aChoiceQuestionIsAnsweredOverExactlyTheOptionsItOffered() {
    ChoiceQuestion question =
        ChoiceQuestion.builder()
            .text("Which team owns this?")
            .option("billing", "money was charged or refunded")
            .option("support")
            .option("fraud", "   ")
            .build();

    ChoiceAnswer answer = model.call(STATE, Map.of("route", question)).getChoice("route");

    Choice space =
        new Choice(
            "Which team owns this?",
            List.of("billing", "support", "fraud"),
            Map.of("billing", "money was charged or refunded"));
    double[] expected = expected(STATE, space);
    assertThat(answer.getProbabilities().keySet()).containsExactly("billing", "support", "fraud");
    assertThat(answer.getProbability("billing")).isCloseTo(expected[0], within(1e-12));
    assertThat(answer.getProbability("support")).isCloseTo(expected[1], within(1e-12));
    assertThat(answer.getProbability("fraud")).isCloseTo(expected[2], within(1e-12));
    assertThat(answer.getValue()).isEqualTo(space.options().get(argmax(expected)));
    assertThat(answer.getConfidence()).isBetween(0.0, 1.0);
  }

  @Test
  void aScoreQuestionIsAnsweredWithTheExpectedLevelAndEachLevelsProbability() {
    ScoreQuestion question =
        ScoreQuestion.builder()
            .text("How urgent is this?")
            .level("low")
            .level("medium", "needs an answer this week")
            .level("high")
            .build();

    ScoreAnswer answer = model.call(STATE, Map.of("urgency", question)).getScore("urgency");

    Score space =
        new Score(
            "How urgent is this?",
            List.of("low", "medium", "high"),
            Map.of("medium", "needs an answer this week"));
    double[] expected = expected(STATE, space);
    assertThat(answer.getProbabilities()).hasSize(3);
    for (int level = 0; level < 3; level++) {
      assertThat(answer.getProbabilities().get(level)).isCloseTo(expected[level], within(1e-12));
    }
    assertThat(answer.getValue())
        .isCloseTo(0 * expected[0] + 1 * expected[1] + 2 * expected[2], within(1e-12))
        .isBetween(0.0, 2.0);
  }

  @Test
  void severalQuestionsReadTheInputOnceAndEachGetsItsOwnKindOfAnswer() {
    Map<String, DecisionQuestion> questions = new LinkedHashMap<>();
    questions.put("dup", YesNoQuestion.of("Was the customer charged twice?"));
    questions.put("route", ChoiceQuestion.of("Which team owns this?", "billing", "support"));
    questions.put("urgency", ScoreQuestion.of("How urgent is this?", "low", "medium", "high"));

    DecisionResponse response = model.call(STATE, questions);

    assertThat(backend.freezes).isEqualTo(1);
    assertThat(backend.scoredPrompts.get(0)).isEqualTo(STATE);
    assertThat(backend.scoredPrompts.stream().filter(STATE::equals)).hasSize(1);
    assertThat(response.getYesNo("dup").getProbability())
        .isCloseTo(expected(STATE, new Noul("Was the customer charged twice?"))[1], within(1e-12));
    assertThat(response.getChoice("route").getProbability("billing"))
        .isCloseTo(
            expected(STATE, new Choice("Which team owns this?", List.of("billing", "support")))[0],
            within(1e-12));
    assertThat(response.getScore("urgency").getProbabilities()).hasSize(3);
    assertThat(response.getResults())
        .extracting(result -> result.getName())
        .containsExactly("dup", "route", "urgency");
  }

  @Test
  void theResponseNamesTheModelAndCountsTheTokensItRead() {
    DecisionResponse response =
        model.call(STATE, Map.of("dup", YesNoQuestion.of("Was the customer charged twice?")));

    String prompt =
        STATE + LetterDecisionEvaluator.questionSuffix(new Noul("Was the customer charged twice?"));
    assertThat(response.getMetadata().getModel()).isEqualTo("scripted-1");
    assertThat(response.getMetadata().getUsage().getPromptTokens()).isEqualTo(prompt.length());
    assertThat(response.getMetadata().getUsage().getCompletionTokens()).isZero();
  }

  @Test
  void fieldsAreShownToTheModelAsAJsonObject() {
    Map<String, Object> fields = new LinkedHashMap<>();
    fields.put("subject", "Charged twice");
    fields.put("amount", 42);
    fields.put("tags", List.of("refund", "vip"));

    DecisionResponse response =
        model.call(
            DecisionPrompt.builder()
                .fields(fields)
                .question("dup", YesNoQuestion.of("Was the customer charged twice?"))
                .build());

    String state = "{\"subject\":\"Charged twice\",\"amount\":42,\"tags\":[\"refund\",\"vip\"]}";
    assertThat(backend.scoredPrompts.get(0)).startsWith(state);
    assertThat(response.getYesNo("dup").getProbability())
        .isCloseTo(expected(state, new Noul("Was the customer charged twice?"))[1], within(1e-12));
  }

  @Test
  void aConversationIsShownToTheModelAsATranscript() {
    DecisionResponse response =
        model.call(
            DecisionPrompt.builder()
                .conversation(
                    List.of(
                        new UserMessage("I was charged twice."),
                        new AssistantMessage("I'm sorry, let me check.")))
                .question("dup", YesNoQuestion.of("Was the customer charged twice?"))
                .build());

    String state = "user: I was charged twice.\nassistant: I'm sorry, let me check.";
    assertThat(backend.scoredPrompts.get(0)).startsWith(state + "\n");
    assertThat(response.getYesNo("dup").getProbability())
        .isCloseTo(expected(state, new Noul("Was the customer charged twice?"))[1], within(1e-12));
  }

  @Test
  void anInputThatIsNotTextIsRefusedRatherThanSilentlyDropped() {
    Media image =
        Media.builder()
            .mimeType(org.springframework.util.MimeTypeUtils.IMAGE_PNG)
            .data(new byte[] {1})
            .build();

    assertThatThrownBy(
            () ->
                model.call(
                    DecisionPrompt.builder()
                        .conversation(
                            List.of(UserMessage.builder().text("see this").media(image).build()))
                        .question("dup", YesNoQuestion.of("Was the customer charged twice?"))
                        .build()))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("text");
  }

  @Test
  void moreOptionsThanThereAreAnswerLettersIsRefusedByName() {
    String[] options = new String[40];
    for (int index = 0; index < options.length; index++) {
      options[index] = "option-" + index;
    }

    assertThatThrownBy(
            () -> model.call(STATE, Map.of("many", ChoiceQuestion.of("Which one?", options))))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("many");
  }

  @Test
  void everyCallIsObserved() {
    List<Observation.Context> seen = new ArrayList<>();
    ObservationRegistry registry = ObservationRegistry.create();
    registry
        .observationConfig()
        .observationHandler(
            new ObservationHandler<Observation.Context>() {
              @Override
              public boolean supportsContext(Observation.Context context) {
                return true;
              }

              @Override
              public void onStop(Observation.Context context) {
                seen.add(context);
              }
            });
    DecisionModel observed = new ModelsSpringAiDecisionModel(backend, "scripted-1", 1.0, registry);

    DecisionResponse response =
        observed.call(STATE, Map.of("dup", YesNoQuestion.of("Was the customer charged twice?")));

    assertThat(seen).hasSize(1);
    assertThat(seen.get(0)).isInstanceOf(DecisionModelObservationContext.class);
    DecisionModelObservationContext context = (DecisionModelObservationContext) seen.get(0);
    assertThat(context.getResponse()).isSameAs(response);
    assertThat(context.getOperationMetadata().provider()).isEqualTo("integrallis");
  }

  @Test
  void overlappingCallsNeverEnterTheBackendTogether() throws Exception {
    backend.readDelayMillis = 15;
    ExecutorService pool = Executors.newFixedThreadPool(4);
    try {
      CountDownLatch start = new CountDownLatch(1);
      List<Future<DecisionResponse>> calls = new ArrayList<>();
      for (int index = 0; index < 4; index++) {
        calls.add(
            pool.submit(
                () -> {
                  start.await();
                  return model.call(
                      STATE,
                      Map.of(
                          "dup", YesNoQuestion.of("Was the customer charged twice?"),
                          "route", ChoiceQuestion.of("Which team?", "billing", "support")));
                }));
      }
      start.countDown();
      for (Future<DecisionResponse> call : calls) {
        assertThat(call.get(30, TimeUnit.SECONDS).getAnswers()).hasSize(2);
      }
    } finally {
      pool.shutdownNow();
    }

    assertThat(backend.maxConcurrent.get()).isEqualTo(1);
  }

  @Test
  void theTemperatureIsAppliedAcrossTheLetters() {
    ChoiceQuestion question = ChoiceQuestion.of("Which team owns this?", "billing", "support");
    double cold = model.call(STATE, Map.of("q", question)).getChoice("q").getProbability("billing");

    double warm =
        new ModelsSpringAiDecisionModel(
                new ScriptedPrefixBackend(), "scripted-1", 8.0, ObservationRegistry.NOOP)
            .call(STATE, Map.of("q", question))
            .getChoice("q")
            .getProbability("billing");

    assertThat(Math.abs(warm - 0.5)).isLessThan(Math.abs(cold - 0.5));
  }

  @Test
  void aFieldHoldingTextPartsIsWrittenAsItsText() {
    Map<String, Object> fields = new LinkedHashMap<>();
    fields.put("subject", org.springframework.ai.chat.messages.part.TextPart.of("Charged twice"));
    fields.put(
        "body",
        List.of(
            org.springframework.ai.chat.messages.part.TextPart.of("Two charges, "),
            org.springframework.ai.chat.messages.part.TextPart.of("one order.")));

    model.call(
        DecisionPrompt.builder()
            .fields(fields)
            .question("dup", YesNoQuestion.of("Was the customer charged twice?"))
            .build());

    assertThat(backend.scoredPrompts.get(0))
        .startsWith("{\"subject\":\"Charged twice\",\"body\":\"Two charges, one order.\"}");
  }

  @Test
  void aFieldHoldingAnImageIsRefusedByName() {
    Media image =
        Media.builder()
            .mimeType(org.springframework.util.MimeTypeUtils.IMAGE_PNG)
            .data(new byte[] {1})
            .build();
    Map<String, Object> fields = new LinkedHashMap<>();
    fields.put("receipt", org.springframework.ai.chat.messages.part.MediaPart.of(image));

    assertThatThrownBy(
            () ->
                model.call(
                    DecisionPrompt.builder()
                        .fields(fields)
                        .question("dup", YesNoQuestion.of("Was the customer charged twice?"))
                        .build()))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("receipt")
        .hasMessageContaining("text only");
  }

  @Test
  void theModelNameDefaultsToTheBackendsAndCannotBeBlank() {
    DecisionResponse response =
        new ModelsSpringAiDecisionModel(backend)
            .call(STATE, Map.of("dup", YesNoQuestion.of("Was the customer charged twice?")));

    assertThat(response.getMetadata().getModel()).isEqualTo("scripted");
    assertThatThrownBy(() -> new ModelsSpringAiDecisionModel(backend, "  "))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("modelName");
  }

  @Test
  void anInputWithNoTextAtAllIsRefused() {
    assertThatThrownBy(
            () ->
                model.call("", Map.of("dup", YesNoQuestion.of("Was the customer charged twice?"))))
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void aFieldThatCannotBeWrittenAsJsonIsRefused() {
    Map<String, Object> fields = new LinkedHashMap<>();
    fields.put("nested", Map.of("opaque", new Object()));

    assertThatThrownBy(
            () ->
                model.call(
                    DecisionPrompt.builder()
                        .fields(fields)
                        .question("dup", YesNoQuestion.of("Was the customer charged twice?"))
                        .build()))
        .isInstanceOf(IllegalArgumentException.class);
  }
}
