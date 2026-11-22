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

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.integrallis.models.api.SharedPrefixInferenceBackend;
import com.integrallis.models.decisions.AnswerSpace;
import com.integrallis.models.decisions.Choice;
import com.integrallis.models.decisions.LetterDecisionBatch;
import com.integrallis.models.decisions.LetterDecisionEvaluator;
import com.integrallis.models.decisions.Noul;
import com.integrallis.models.decisions.Score;
import com.integrallis.models.decisions.Verdict;
import io.micrometer.observation.ObservationRegistry;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.locks.ReentrantLock;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.part.MessagePart;
import org.springframework.ai.chat.messages.part.TextPart;
import org.springframework.ai.chat.metadata.DefaultUsage;
import org.springframework.ai.decision.DecisionModel;
import org.springframework.ai.decision.DecisionPrompt;
import org.springframework.ai.decision.DecisionResponse;
import org.springframework.ai.decision.DecisionResponseMetadata;
import org.springframework.ai.decision.answer.ChoiceAnswer;
import org.springframework.ai.decision.answer.DecisionAnswer;
import org.springframework.ai.decision.answer.ScoreAnswer;
import org.springframework.ai.decision.answer.YesNoAnswer;
import org.springframework.ai.decision.input.ConversationInput;
import org.springframework.ai.decision.input.DecisionInput;
import org.springframework.ai.decision.input.FieldsInput;
import org.springframework.ai.decision.input.TextInput;
import org.springframework.ai.decision.observation.DecisionModelObservationContext;
import org.springframework.ai.decision.observation.DecisionModelObservationConvention;
import org.springframework.ai.decision.observation.DecisionModelObservationDocumentation;
import org.springframework.ai.decision.observation.DefaultDecisionModelObservationConvention;
import org.springframework.ai.decision.question.ChoiceQuestion;
import org.springframework.ai.decision.question.DecisionQuestion;
import org.springframework.ai.decision.question.ScoreLevel;
import org.springframework.ai.decision.question.ScoreQuestion;
import org.springframework.ai.decision.question.YesNoQuestion;

/**
 * Spring AI {@link DecisionModel} answered in-process by a Models backend.
 *
 * <p>Each Spring AI question type is one of this library's answer spaces, criteria included:
 *
 * <ul>
 *   <li>{@link YesNoQuestion} is a {@link Noul}, answered with a {@link YesNoAnswer} carrying the
 *       probability of yes;
 *   <li>{@link ChoiceQuestion} is a {@link Choice} over exactly the options offered, answered with
 *       a {@link ChoiceAnswer} carrying every option's probability;
 *   <li>{@link ScoreQuestion} is a {@link Score}, answered with a {@link ScoreAnswer} carrying each
 *       level's probability and the expected level.
 * </ul>
 *
 * <p>Questions are not known in advance, so they are answered by {@link LetterDecisionEvaluator}:
 * the options are shown to the model as a lettered list and the answer is the distribution over the
 * letters. Nothing is generated. With several questions in one prompt the input is read once and
 * every question forks it, when the backend can share a prefix; the response metadata says whether
 * that happened ({@value #STATE_PREFILLS} and {@value #SHARED_PREFIX_PROVEN}).
 *
 * <p><b>What the model is shown.</b> A text input as it is; a fields input as a JSON object; a
 * conversation as one {@code role: text} line per message. Only text is supported: an image or any
 * other non-text part is refused, not dropped.
 *
 * <p><b>What the probabilities are.</b> The model's distribution over the answer letters,
 * renormalised across them and scaled by this adapter's temperature. They are not fitted to any
 * dataset by this class; {@link ChoiceAnswer#getConfidence()} is how concentrated the distribution
 * is, not how likely the answer is to be right.
 *
 * <p>{@link DecisionPrompt#getOptions()} is not consulted: the adapter answers with the one model
 * it was built over. Calls are serialised, because a backend is stateful; a backend shared with
 * another component must be serialised by the application as well.
 *
 * <p>Requires Spring AI 2.1 or later. On an earlier line this class is simply never loaded.
 */
public final class ModelsSpringAiDecisionModel implements DecisionModel {

  /** Response-metadata key: how many times the input was read. One when a prefix was shared. */
  public static final String STATE_PREFILLS = "state_prefills";

  /** Response-metadata key: whether the backend proved every question shared one prefix. */
  public static final String SHARED_PREFIX_PROVEN = "shared_prefix_proven";

  private static final String PROVIDER = "integrallis";
  private static final ObjectMapper JSON = new ObjectMapper();
  private static final DecisionModelObservationConvention DEFAULT_OBSERVATION_CONVENTION =
      new DefaultDecisionModelObservationConvention();

  private final LetterDecisionEvaluator evaluator;
  private final String modelName;
  private final ObservationRegistry observationRegistry;
  private final ReentrantLock backendLock = new ReentrantLock();

  /**
   * Creates an adapter named after the backend, with temperature 1 and no observation.
   *
   * @param backend the loaded model
   */
  public ModelsSpringAiDecisionModel(SharedPrefixInferenceBackend backend) {
    this(backend, Objects.requireNonNull(backend, "backend").name());
  }

  /**
   * Creates an adapter with temperature 1 and no observation.
   *
   * @param backend the loaded model
   * @param modelName the name reported in response metadata and observations
   */
  public ModelsSpringAiDecisionModel(SharedPrefixInferenceBackend backend, String modelName) {
    this(backend, modelName, 1.0, ObservationRegistry.NOOP);
  }

  /**
   * Creates an adapter.
   *
   * @param backend the loaded model
   * @param modelName the name reported in response metadata and observations
   * @param temperature positive, finite temperature applied across the answer letters
   * @param observationRegistry where each call is observed
   */
  public ModelsSpringAiDecisionModel(
      SharedPrefixInferenceBackend backend,
      String modelName,
      double temperature,
      ObservationRegistry observationRegistry) {
    this.evaluator = new LetterDecisionEvaluator(backend, temperature);
    this.modelName = requireText(modelName, "modelName");
    this.observationRegistry = Objects.requireNonNull(observationRegistry, "observationRegistry");
  }

  @Override
  public DecisionResponse call(DecisionPrompt prompt) {
    Objects.requireNonNull(prompt, "prompt");
    DecisionModelObservationContext context =
        DecisionModelObservationContext.builder().decisionPrompt(prompt).provider(PROVIDER).build();
    return DecisionModelObservationDocumentation.DECISION_MODEL_OPERATION
        .observation(null, DEFAULT_OBSERVATION_CONVENTION, () -> context, observationRegistry)
        .observe(
            () -> {
              DecisionResponse response = answer(prompt);
              context.setResponse(response);
              return response;
            });
  }

  private DecisionResponse answer(DecisionPrompt prompt) {
    String state = DecisionInputs.render(prompt.getInstructions());
    List<String> names = new ArrayList<>(prompt.getQuestions().size());
    List<AnswerSpace> spaces = new ArrayList<>(prompt.getQuestions().size());
    for (Map.Entry<String, DecisionQuestion> entry : prompt.getQuestions().entrySet()) {
      names.add(entry.getKey());
      spaces.add(answerSpace(entry.getKey(), entry.getValue()));
    }

    LetterDecisionBatch batch;
    backendLock.lock();
    try {
      batch = evaluator.evaluate(state, spaces);
    } finally {
      backendLock.unlock();
    }

    DecisionResponse.Builder response = DecisionResponse.builder(prompt);
    for (int index = 0; index < names.size(); index++) {
      response.answer(names.get(index), answer(batch.verdicts().get(index)));
    }
    DecisionResponseMetadata metadata =
        DecisionResponseMetadata.builder()
            .model(modelName)
            .usage(new DefaultUsage(batch.inputTokens(), 0))
            .build();
    metadata.put(STATE_PREFILLS, batch.statePrefills());
    metadata.put(SHARED_PREFIX_PROVEN, batch.sharedPrefixProven());
    return response.metadata(metadata).build();
  }

  /**
   * The answer space a Spring AI question asks over. A question this library cannot ask — too many
   * options for the answer letters, a duplicate option — is refused with the question's name.
   */
  static AnswerSpace answerSpace(String name, DecisionQuestion question) {
    try {
      AnswerSpace space = toAnswerSpace(question);
      // Rendering is where the letter limit is enforced; do it now so the refusal names the
      // question instead of surfacing from the middle of a batch.
      LetterDecisionEvaluator.questionSuffix(space);
      return space;
    } catch (IllegalArgumentException refused) {
      throw new IllegalArgumentException(
          "question '" + name + "' cannot be asked: " + refused.getMessage(), refused);
    }
  }

  // DecisionQuestion is sealed, so this switch is exhaustive: a question type added by a later
  // Spring AI fails compilation here instead of falling through to a guess.
  private static AnswerSpace toAnswerSpace(DecisionQuestion question) {
    return switch (question) {
      case YesNoQuestion yesNo -> {
        Map<String, String> criteria = new LinkedHashMap<>();
        putIfText(criteria, "true", yesNo.getWhenYes());
        putIfText(criteria, "false", yesNo.getWhenNo());
        yield new Noul(yesNo.getText(), criteria);
      }
      case ChoiceQuestion choice -> {
        Map<String, String> criteria = new LinkedHashMap<>();
        choice
            .getOptions()
            .forEach((option, description) -> putIfText(criteria, option, description));
        yield new Choice(choice.getText(), choice.getOptionNames(), criteria);
      }
      case ScoreQuestion score -> {
        Map<String, String> criteria = new LinkedHashMap<>();
        for (ScoreLevel level : score.getLevels()) {
          putIfText(criteria, level.getLabel(), level.getDescription());
        }
        yield new Score(score.getText(), score.getLevelLabels(), criteria);
      }
    };
  }

  private static DecisionAnswer answer(Verdict verdict) {
    double[] probabilities = verdict.probabilities();
    AnswerSpace space = verdict.space();
    if (space instanceof Noul) {
      return YesNoAnswer.builder().probability(verdict.probabilityOfTrue()).build();
    }
    if (space instanceof Score) {
      List<Double> perLevel = new ArrayList<>(probabilities.length);
      double expectedLevel = 0;
      for (int level = 0; level < probabilities.length; level++) {
        perLevel.add(probabilities[level]);
        expectedLevel += level * probabilities[level];
      }
      return ScoreAnswer.builder()
          .value(expectedLevel)
          .probabilities(perLevel)
          .confidence(verdict.confidence())
          .build();
    }
    Map<String, Double> perOption = new LinkedHashMap<>();
    List<String> labels = space.labels();
    for (int index = 0; index < probabilities.length; index++) {
      perOption.put(labels.get(index), probabilities[index]);
    }
    return ChoiceAnswer.builder()
        .value(verdict.winner())
        .probabilities(perOption)
        .confidence(verdict.confidence())
        .build();
  }

  private static void putIfText(Map<String, String> criteria, String label, String text) {
    if (text != null && !text.isBlank()) {
      criteria.put(label, text);
    }
  }

  private static String requireText(String value, String name) {
    Objects.requireNonNull(value, name);
    if (value.isBlank()) {
      throw new IllegalArgumentException(name + " must not be blank");
    }
    return value;
  }

  /** Turns a Spring AI decision input into the text the model reads. */
  static final class DecisionInputs {

    private DecisionInputs() {}

    // DecisionInput is sealed too; see toAnswerSpace.
    static String render(DecisionInput input) {
      return switch (input) {
        case TextInput text -> text(text.getParts(), "the text input");
        case FieldsInput fields -> json(fields.getFields());
        case ConversationInput conversation -> transcript(conversation);
      };
    }

    private static String transcript(ConversationInput conversation) {
      StringBuilder transcript = new StringBuilder();
      for (Message message : conversation.getMessages()) {
        if (transcript.length() > 0) {
          transcript.append('\n');
        }
        String role = message.getMessageType().getValue();
        transcript
            .append(role)
            .append(": ")
            .append(text(ConversationInput.parts(message), "a " + role + " message"));
      }
      return transcript.toString();
    }

    private static String json(Map<String, Object> fields) {
      Map<String, Object> plain = new LinkedHashMap<>();
      for (Map.Entry<String, Object> field : fields.entrySet()) {
        Object value = field.getValue();
        if (value instanceof MessagePart part) {
          value = text(List.of(part), "field '" + field.getKey() + "'");
        } else if (FieldsInput.isPartList(value)) {
          List<MessagePart> parts = new ArrayList<>();
          for (Object part : (List<?>) value) {
            parts.add((MessagePart) part);
          }
          value = text(parts, "field '" + field.getKey() + "'");
        }
        plain.put(field.getKey(), value);
      }
      try {
        return JSON.writeValueAsString(plain);
      } catch (JsonProcessingException unserialisable) {
        throw new IllegalArgumentException(
            "the fields input cannot be written as JSON: " + unserialisable.getOriginalMessage(),
            unserialisable);
      }
    }

    private static String text(List<MessagePart> parts, String what) {
      StringBuilder text = new StringBuilder();
      for (MessagePart part : parts) {
        if (!(part instanceof TextPart textPart)) {
          throw new IllegalArgumentException(
              what
                  + " contains a "
                  + part.getClass().getSimpleName()
                  + "; this decision model reads text only");
        }
        text.append(textPart.text());
      }
      return text.toString();
    }
  }
}
