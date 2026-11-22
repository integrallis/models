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
import static org.assertj.core.api.Assertions.within;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.integrallis.models.backend.purejava.PureJavaBackend;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.springframework.ai.decision.DecisionResponse;
import org.springframework.ai.decision.answer.ChoiceAnswer;
import org.springframework.ai.decision.answer.ScoreAnswer;
import org.springframework.ai.decision.question.ChoiceQuestion;
import org.springframework.ai.decision.question.DecisionQuestion;
import org.springframework.ai.decision.question.ScoreQuestion;
import org.springframework.ai.decision.question.YesNoQuestion;

/**
 * The adapter over a real model, end to end: real tokenizer, real sessions, a real shared prefix.
 *
 * <p>This shows that the path works on a loaded model, shared prefix or not, and that asking a
 * question alone gives the same distribution as asking it among others. It is not an accuracy
 * measurement: the gated assertions are coarse sanity checks, and the numbers are printed. Skipped
 * when the fixture is absent.
 */
@Tag("integration")
class RealModelDecisionIntegrationTest {

  private static final String STATE =
      "Support ticket: I was billed twice for my March subscription and I want one charge"
          + " refunded today. This is the third time I have written about it.";

  private static final String MODEL_FILE =
      System.getProperty("models.fixtures.decisionModel", "Qwen3-1.7B-Q8_0.gguf");

  private static Path fixture() {
    String name = MODEL_FILE;
    return Path.of(
        System.getProperty(
            "models.fixtures.directory",
            Path.of(System.getProperty("user.home"), ".jvllm", "models").toString()),
        name);
  }

  @Test
  void answersEveryQuestionTypeAndSharingThePrefixDoesNotChangeAnAnswer() throws Exception {
    Path model = fixture();
    assumeTrue(Files.isRegularFile(model), "no model fixture at " + model);

    try (PureJavaBackend backend = PureJavaBackend.load(model)) {
      boolean shares = backend.supportsSharedPrefixes();
      ModelsSpringAiDecisionModel decisions = new ModelsSpringAiDecisionModel(backend, MODEL_FILE);

      YesNoQuestion billedTwice = YesNoQuestion.of("Was the customer billed more than once?");
      YesNoQuestion wantsCancel = YesNoQuestion.of("Is the customer asking about a late delivery?");
      ChoiceQuestion team =
          ChoiceQuestion.builder()
              .text("Which team should handle this ticket?")
              .option("billing", "charges, refunds and invoices")
              .option("shipping", "delivery and tracking of physical goods")
              .option("technical", "bugs, outages and login problems")
              .build();
      ScoreQuestion urgency =
          ScoreQuestion.builder()
              .text("How urgent is this ticket?")
              .level("low", "no time pressure and no repeated contact")
              .level("medium", "wants an answer soon")
              .level("high", "wants action today or has written repeatedly")
              .build();

      Map<String, DecisionQuestion> all = new LinkedHashMap<>();
      all.put("billedTwice", billedTwice);
      all.put("lateDelivery", wantsCancel);
      all.put("team", team);
      all.put("urgency", urgency);

      long start = System.nanoTime();
      DecisionResponse together = decisions.call(STATE, all);
      double togetherMs = (System.nanoTime() - start) / 1e6;

      double yesTrue = together.getYesNo("billedTwice").getProbability();
      double yesFalse = together.getYesNo("lateDelivery").getProbability();
      ChoiceAnswer route = together.getChoice("team");
      ScoreAnswer score = together.getScore("urgency");
      System.out.printf(
          "REAL %s | shared: prefills=%s proven=%s tokens=%d %.0f ms%n",
          MODEL_FILE,
          together.getMetadata().get(ModelsSpringAiDecisionModel.STATE_PREFILLS),
          together.getMetadata().get(ModelsSpringAiDecisionModel.SHARED_PREFIX_PROVEN),
          together.getMetadata().getUsage().getPromptTokens(),
          togetherMs);
      System.out.printf(
          "REAL p(yes | billed twice)=%.4f  p(yes | late delivery)=%.4f%n", yesTrue, yesFalse);
      System.out.printf("REAL team=%s %s%n", route.getValue(), route.getProbabilities());
      System.out.printf(
          "REAL urgency expected level=%.3f %s%n", score.getValue(), score.getProbabilities());

      // A family that can share a KV prefix reads the ticket once and proves it; one that cannot
      // (a recurrent architecture, for instance) reads it once per question and says so.
      assertThat(together.getMetadata().<Object>get(ModelsSpringAiDecisionModel.STATE_PREFILLS))
          .isEqualTo(shares ? 1 : all.size());
      assertThat(
              together.getMetadata().<Object>get(ModelsSpringAiDecisionModel.SHARED_PREFIX_PROVEN))
          .isEqualTo(shares);

      // Sanity with wide margins on both models this was run on (2026-11-21, Qwen3 1.7B Q8_0 and
      // Qwen3.5 4B Q4_K_M): a billing ticket routes to billing, and a ticket asking for action
      // today sits in the upper half of the urgency scale. The yes/no pair is printed and not
      // gated: Qwen3.5 4B separated the true statement from the false one (0.696 against 0.035),
      // Qwen3 1.7B answered yes to both (0.999 against 0.984), which is the model, not this code.
      assertThat(route.getValue()).isEqualTo("billing");
      assertThat(score.getValue()).isBetween(1.0, 2.0);
      assertThat(yesTrue).isBetween(0.0, 1.0);
      assertThat(yesFalse).isBetween(0.0, 1.0);
      assertThat(route.getProbabilities().values().stream().mapToDouble(Double::doubleValue).sum())
          .isCloseTo(1.0, within(1e-9));

      // Asking one question alone re-reads the ticket; the answer must be the same distribution.
      start = System.nanoTime();
      DecisionResponse alone = decisions.call(STATE, Map.of("team", team));
      double aloneMs = (System.nanoTime() - start) / 1e6;
      System.out.printf(
          "REAL alone: prefills=%s %.0f ms; max |shared - alone| over team options = %.2e%n",
          alone.getMetadata().get(ModelsSpringAiDecisionModel.STATE_PREFILLS),
          aloneMs,
          team.getOptionNames().stream()
              .mapToDouble(
                  option ->
                      Math.abs(
                          route.getProbability(option)
                              - alone.getChoice("team").getProbability(option)))
              .max()
              .orElseThrow());
      for (String option : team.getOptionNames()) {
        assertThat(alone.getChoice("team").getProbability(option))
            .isCloseTo(route.getProbability(option), within(1e-3));
      }
      assertThat(alone.getChoice("team").getValue()).isEqualTo(route.getValue());
    }
  }
}
