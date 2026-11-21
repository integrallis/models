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

import com.integrallis.models.api.HiddenStateInferenceBackend;
import com.integrallis.models.api.InferenceBackend;
import com.integrallis.models.decisions.AnswerSpace;
import com.integrallis.models.decisions.DecisionHead;
import com.integrallis.models.decisions.Verdict;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import org.springframework.ai.decision.DecisionModel;
import org.springframework.ai.decision.DecisionPrompt;
import org.springframework.ai.decision.DecisionResponse;
import org.springframework.ai.decision.answer.ChoiceAnswer;
import org.springframework.ai.decision.input.DecisionInput;
import org.springframework.ai.decision.input.TextInput;
import org.springframework.ai.decision.question.DecisionQuestion;

/**
 * Spring AI 2.1 {@link DecisionModel} backed by Models calibrated decision heads.
 *
 * <p>Each named question in a {@link DecisionPrompt} is looked up in a configured {@link
 * DecisionHead} map. The input text is tokenized, the backend prefill produces a hidden state, and
 * the head scores it into a calibrated {@link Verdict}. The verdict is mapped to a {@link
 * ChoiceAnswer} with every label's probability so callers can inspect confidence, not just the
 * winner.
 *
 * <p>The backend must implement {@link HiddenStateInferenceBackend} and return {@code true} from
 * {@code supportsHiddenState()}. Construction fails fast if it does not.
 */
public final class ModelsSpringAiDecisionModel implements DecisionModel {

  private final InferenceBackend backend;
  private final Map<String, DecisionHead> heads;
  private final double defaultTemperature;

  /**
   * Creates a decision model over the given backend and heads.
   *
   * @param backend a backend that exposes hidden states
   * @param heads map from question name (the key in {@code DecisionPrompt.getQuestions()}) to the
   *     calibrated head that answers it
   * @param defaultTemperature temperature passed to {@link DecisionHead#decide} when no per-head
   *     override is configured
   */
  public ModelsSpringAiDecisionModel(
      InferenceBackend backend, Map<String, DecisionHead> heads, double defaultTemperature) {
    this.backend = Objects.requireNonNull(backend, "backend");
    this.heads = Map.copyOf(Objects.requireNonNull(heads, "heads"));
    this.defaultTemperature = defaultTemperature;
    if (backend instanceof HiddenStateInferenceBackend h && !h.supportsHiddenState()) {
      throw new IllegalArgumentException(
          "backend " + backend.name() + " does not support hidden states");
    } else if (!(backend instanceof HiddenStateInferenceBackend)) {
      throw new IllegalArgumentException(
          "backend " + backend.name() + " does not implement HiddenStateInferenceBackend");
    }
  }

  /**
   * Creates a decision model with temperature {@code 1.0}.
   *
   * @param backend a backend that exposes hidden states
   * @param heads map from question name to calibrated head
   */
  public ModelsSpringAiDecisionModel(InferenceBackend backend, Map<String, DecisionHead> heads) {
    this(backend, heads, 1.0);
  }

  @Override
  public DecisionResponse call(DecisionPrompt prompt) {
    String stateText = extractText(prompt.getInstructions());
    int[] stateTokens = backend.tokenizer().encode(stateText);

    DecisionResponse.Builder responseBuilder = DecisionResponse.builder(prompt);

    for (Map.Entry<String, DecisionQuestion> entry : prompt.getQuestions().entrySet()) {
      String name = entry.getKey();
      DecisionQuestion question = entry.getValue();
      DecisionHead head = heads.get(name);
      if (head == null) {
        throw new IllegalArgumentException(
            "No DecisionHead configured for question '" + name + "'. Available: " + heads.keySet());
      }

      String questionText = question.getText();
      int[] questionTokens =
          questionText == null || questionText.isEmpty()
              ? new int[0]
              : backend.tokenizer().encode(questionText);

      float[] hidden = evaluateHiddenState(stateTokens, questionTokens);
      Verdict verdict = head.decide(hidden, defaultTemperature);

      ChoiceAnswer answer = toChoiceAnswer(verdict);
      responseBuilder.answer(name, answer);
    }

    return responseBuilder.build();
  }

  /** Visible for testing. */
  static String extractText(DecisionInput input) {
    if (input instanceof TextInput textInput) {
      String text = textInput.getText();
      return text != null ? text : "";
    }
    throw new UnsupportedOperationException(
        "DecisionInput type not supported: " + input.getClass().getName());
  }

  private float[] evaluateHiddenState(int[] stateTokens, int[] questionTokens) {
    HiddenStateInferenceBackend hs = (HiddenStateInferenceBackend) backend;
    backend.reset();
    int[] combined = new int[stateTokens.length + questionTokens.length];
    System.arraycopy(stateTokens, 0, combined, 0, stateTokens.length);
    System.arraycopy(questionTokens, 0, combined, stateTokens.length, questionTokens.length);
    return hs.prefillHiddenState(combined, 0);
  }

  private static ChoiceAnswer toChoiceAnswer(Verdict verdict) {
    AnswerSpace space = verdict.space();
    Map<String, Double> probabilities = new LinkedHashMap<>();
    for (String label : space.labels()) {
      probabilities.put(label, verdict.probabilityOf(label));
    }
    return ChoiceAnswer.builder()
        .value(verdict.winner())
        .probabilities(probabilities)
        .confidence(verdict.confidence())
        .build();
  }
}
