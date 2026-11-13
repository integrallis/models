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
package com.integrallis.models.tinystruct;

import com.integrallis.models.api.InferenceSession;
import com.integrallis.models.api.SharedInferencePrefix;
import com.integrallis.models.api.SharedPrefixInferenceBackend;
import com.integrallis.models.decisions.AnswerSpace;
import com.integrallis.models.decisions.Choice;
import com.integrallis.models.decisions.LetterLogitScorer;
import com.integrallis.models.decisions.Noul;
import com.integrallis.models.decisions.Score;
import com.integrallis.models.decisions.Verdict;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.BiFunction;
import org.tinystruct.ApplicationException;
import org.tinystruct.data.component.Builder;
import org.tinystruct.typesafe.client.RoutingRequest;
import org.tinystruct.typesafe.client.RoutingResult;
import org.tinystruct.typesafe.client.TypesafeClient;

/**
 * In-process {@link TypesafeClient} backed by Models typed decisions ({@link Choice}, {@link Noul},
 * and {@link Score}) over a {@link SharedPrefixInferenceBackend} or a caller-supplied decision
 * evaluator.
 *
 * <p>Unlike {@code HttpTypesafeClient}, which sends every natural-language instruction and its
 * action/parameter questions over the network to a hosted System One endpoint, this client answers
 * every question inside the JVM with no decode loop. When backed by a {@link
 * SharedPrefixInferenceBackend} that supports shared prefixes and a request carries multiple
 * questions, the user state is prefilled and frozen once and each question forks that prefix.
 */
public final class ModelsTypesafeClient implements TypesafeClient, AutoCloseable {

  /** Default model version reported when none is explicitly configured. */
  public static final String DEFAULT_MODEL_VERSION = "models-decisions-local";

  private final SharedPrefixInferenceBackend backend;
  private final BiFunction<String, AnswerSpace, Verdict> evaluator;
  private final LetterLogitScorer letterScorer;
  private final String modelVersion;
  private final Object backendMonitor = new Object();

  /**
   * Creates an in-process TypeSafe client over the given backend with unit temperature ({@code
   * 1.0}).
   *
   * @param backend the in-process inference backend
   */
  public ModelsTypesafeClient(SharedPrefixInferenceBackend backend) {
    this(backend, defaultModelName(backend), 1.0);
  }

  /**
   * Creates an in-process TypeSafe client over the given backend with a calibration temperature.
   *
   * @param backend the in-process inference backend
   * @param modelVersion the model identifier recorded in {@link RoutingResult#getModelVersion()}
   * @param temperature positive, finite calibration temperature for {@link LetterLogitScorer}
   */
  public ModelsTypesafeClient(
      SharedPrefixInferenceBackend backend, String modelVersion, double temperature) {
    this.backend = Objects.requireNonNull(backend, "backend");
    this.modelVersion = requireText(modelVersion, "modelVersion");
    this.letterScorer = new LetterLogitScorer(temperature);
    this.evaluator = null;
  }

  /**
   * Creates an in-process TypeSafe client backed by a custom decision function (for example, one
   * dispatching to calibrated {@code DecisionArtifact} heads or {@code ModernBertDecisionModel}).
   *
   * @param evaluator function taking {@code (state, answerSpace)} and returning a {@link Verdict}
   * @param modelVersion the model identifier recorded in {@link RoutingResult#getModelVersion()}
   */
  public ModelsTypesafeClient(
      BiFunction<String, AnswerSpace, Verdict> evaluator, String modelVersion) {
    this.backend = null;
    this.letterScorer = null;
    this.evaluator = Objects.requireNonNull(evaluator, "evaluator");
    this.modelVersion = requireText(modelVersion, "modelVersion");
  }

  /** Returns the configured model version reported by this client. */
  public String modelVersion() {
    return modelVersion;
  }

  @Override
  public RoutingResult classify(RoutingRequest request) throws ApplicationException {
    Objects.requireNonNull(request, "request");
    String state = extractStateText(request.getState());
    Builder questions = request.getQuestions();
    if (questions == null || questions.isEmpty()) {
      throw new ApplicationException("TypeSafe routing request must contain at least one question");
    }

    Map<String, ParsedQuestion> parsedQuestions = parseQuestions(questions);
    String effectiveModel =
        request.getModel() != null && !request.getModel().isBlank()
            ? request.getModel()
            : modelVersion;

    if (evaluator != null) {
      return classifyWithEvaluator(state, parsedQuestions, effectiveModel);
    }
    return classifyWithBackend(state, parsedQuestions, effectiveModel);
  }

  /**
   * Parses a single TypeSafe question descriptor {@link Builder} into a Models {@link AnswerSpace}.
   *
   * @param questionId the question identifier for error reporting
   * @param question the question descriptor containing {@code type}, {@code instructions}, and
   *     optional {@code criteria}
   * @return the corresponding {@link AnswerSpace}
   * @throws ApplicationException if the descriptor is malformed or uses an unsupported type
   */
  public static AnswerSpace toAnswerSpace(String questionId, Builder question)
      throws ApplicationException {
    Objects.requireNonNull(questionId, "questionId");
    if (question == null) {
      throw new ApplicationException("Question '" + questionId + "' descriptor must not be null");
    }
    Object rawType = question.get("type");
    if (rawType == null || rawType.toString().isBlank()) {
      throw new ApplicationException("Question '" + questionId + "' is missing 'type'");
    }
    String type = rawType.toString().strip().toLowerCase(java.util.Locale.ROOT);
    Object rawInstructions = question.get("instructions");
    if (rawInstructions == null || rawInstructions.toString().isBlank()) {
      throw new ApplicationException("Question '" + questionId + "' is missing 'instructions'");
    }
    String instructions = rawInstructions.toString().strip();

    try {
      return switch (type) {
        case "choice" -> parseChoice(questionId, instructions, question.get("criteria"));
        case "noul" -> parseNoul(instructions, question.get("criteria"));
        case "score" -> parseScore(questionId, instructions, question.get("criteria"));
        default ->
            throw new ApplicationException(
                "Question '" + questionId + "' has unsupported type '" + type + "'");
      };
    } catch (IllegalArgumentException invalid) {
      throw new ApplicationException(
          "Question '" + questionId + "' is invalid: " + invalid.getMessage(), invalid);
    }
  }

  private RoutingResult classifyWithEvaluator(
      String state, Map<String, ParsedQuestion> parsedQuestions, String effectiveModel)
      throws ApplicationException {
    Builder answers = new Builder();
    for (Map.Entry<String, ParsedQuestion> entry : parsedQuestions.entrySet()) {
      String questionId = entry.getKey();
      ParsedQuestion parsed = entry.getValue();
      Verdict verdict;
      try {
        verdict =
            Objects.requireNonNull(
                evaluator.apply(state, parsed.space()), "evaluator returned null verdict");
      } catch (RuntimeException failure) {
        throw new ApplicationException(
            "Decision evaluation failed for question '" + questionId + "': " + failure.getMessage(),
            failure);
      }
      answers.put(questionId, formatAnswer(parsed.type(), verdict));
    }
    Builder usage = new Builder();
    usage.put("input_tokens", estimateTokens(state));
    usage.put("output_tokens", 0);
    return new RoutingResult(answers, effectiveModel, usage);
  }

  private RoutingResult classifyWithBackend(
      String state, Map<String, ParsedQuestion> parsedQuestions, String effectiveModel)
      throws ApplicationException {
    synchronized (backendMonitor) {
      try {
        int[] stateTokens = backend.tokenizer().encode(state);
        if (stateTokens.length == 0) {
          throw new ApplicationException("Encoded state produced no tokens");
        }
        Builder answers = new Builder();
        int totalInputTokens = stateTokens.length;
        int statePrefills = 0;
        boolean sharedPrefixProven = false;

        if (parsedQuestions.size() > 1 && backend.supportsSharedPrefixes()) {
          SharedInferencePrefix prefix;
          try (InferenceSession source = backend.openSession()) {
            backend.prefill(source, stateTokens, 0);
            statePrefills = 1;
            prefix = backend.freezePrefix(source);
          }
          boolean allShared = true;
          try (InferenceSession witness = backend.fork(prefix)) {
            for (Map.Entry<String, ParsedQuestion> entry : parsedQuestions.entrySet()) {
              ParsedQuestion parsed = entry.getValue();
              int[] suffixTokens = backend.tokenizer().encode(renderQuestionSuffix(parsed.space()));
              totalInputTokens += suffixTokens.length;
              try (InferenceSession branch = backend.fork(prefix)) {
                allShared &= backend.sharesPrefixStorage(witness, branch);
                float[] logits =
                    prefillSuffixLogits(branch, stateTokens, suffixTokens, stateTokens.length);
                Verdict verdict = scoreVerdict(parsed.space(), logits);
                answers.put(entry.getKey(), formatAnswer(parsed.type(), verdict));
              }
            }
          }
          sharedPrefixProven = allShared;
        } else {
          for (Map.Entry<String, ParsedQuestion> entry : parsedQuestions.entrySet()) {
            ParsedQuestion parsed = entry.getValue();
            String fullPrompt = state + renderQuestionSuffix(parsed.space());
            int[] promptTokens = backend.tokenizer().encode(fullPrompt);
            totalInputTokens += Math.max(0, promptTokens.length - stateTokens.length);
            try (InferenceSession session = backend.openSession()) {
              float[] logits = backend.prefill(session, promptTokens, 0);
              statePrefills++;
              Verdict verdict = scoreVerdict(parsed.space(), logits);
              answers.put(entry.getKey(), formatAnswer(parsed.type(), verdict));
            }
          }
        }

        Builder usage = new Builder();
        usage.put("input_tokens", totalInputTokens);
        usage.put("output_tokens", 0);
        usage.put("state_prefills", statePrefills);
        usage.put("shared_prefix_proven", sharedPrefixProven);
        return new RoutingResult(answers, effectiveModel, usage);
      } catch (ApplicationException expected) {
        throw expected;
      } catch (RuntimeException failure) {
        throw new ApplicationException(
            "In-process TypeSafe classification failed: " + failure.getMessage(), failure);
      }
    }
  }

  private float[] prefillSuffixLogits(
      InferenceSession branch, int[] stateTokens, int[] suffixTokens, int startPosition) {
    if (suffixTokens.length == 0) {
      return backend.forward(
          branch, stateTokens[stateTokens.length - 1], Math.max(0, stateTokens.length - 1));
    }
    return backend.prefill(branch, suffixTokens, startPosition);
  }

  private Verdict scoreVerdict(AnswerSpace space, float[] logits) {
    List<String> labels = space.labels();
    int[] letterTokens = new int[labels.size()];
    for (int slot = 0; slot < labels.size(); slot++) {
      int[] encoded = backend.tokenizer().encode(" " + (char) ('A' + slot));
      if (encoded.length == 0) {
        throw new IllegalStateException("Tokenizer produced no token for option letter " + slot);
      }
      letterTokens[slot] = encoded[encoded.length - 1];
    }
    return letterScorer.score(space, logits, letterTokens);
  }

  private static String renderQuestionSuffix(AnswerSpace space) {
    List<String> labels = space.labels();
    String criteriaBlock = LetterLogitScorer.renderCriteria(labels, space.criteria());
    StringBuilder suffix = new StringBuilder();
    if (!criteriaBlock.isEmpty()) {
      suffix.append('\n').append(criteriaBlock);
    }
    suffix
        .append('\n')
        .append(space.question())
        .append('\n')
        .append(LetterLogitScorer.renderOptions(labels));
    return suffix.toString();
  }

  private static Builder formatAnswer(String type, Verdict verdict) {
    Builder answer = new Builder();
    if ("noul".equals(type)) {
      answer.put("noul", verdict.probabilityOfTrue());
      return answer;
    }
    answer.put("choice", verdict.winner());
    answer.put("confidence", verdict.confidence());
    Builder probabilities = new Builder();
    List<String> labels = verdict.space().labels();
    double[] values = verdict.probabilities();
    for (int index = 0; index < labels.size(); index++) {
      probabilities.put(labels.get(index), values[index]);
    }
    answer.put("probabilities", probabilities);
    return answer;
  }

  private static Map<String, ParsedQuestion> parseQuestions(Builder questions)
      throws ApplicationException {
    Map<String, ParsedQuestion> parsed = new LinkedHashMap<>();
    for (Map.Entry<String, Object> entry : questions.entrySet()) {
      String questionId = entry.getKey();
      if (!(entry.getValue() instanceof Builder descriptor)) {
        throw new ApplicationException(
            "Question '" + questionId + "' must be a Builder object, got: " + entry.getValue());
      }
      AnswerSpace space = toAnswerSpace(questionId, descriptor);
      String type = descriptor.get("type").toString().strip().toLowerCase(java.util.Locale.ROOT);
      parsed.put(questionId, new ParsedQuestion(type, space));
    }
    return parsed;
  }

  private static Choice parseChoice(String questionId, String instructions, Object rawCriteria)
      throws ApplicationException {
    if (!(rawCriteria instanceof Map<?, ?> criteriaMap) || criteriaMap.isEmpty()) {
      throw new ApplicationException(
          "Choice question '" + questionId + "' requires a non-empty 'criteria' map");
    }
    List<String> options = new ArrayList<>(criteriaMap.size());
    Map<String, String> rubric = new LinkedHashMap<>();
    for (Map.Entry<?, ?> entry : criteriaMap.entrySet()) {
      if (entry.getKey() == null) {
        throw new ApplicationException(
            "Choice question '" + questionId + "' contains a null option key");
      }
      String option = entry.getKey().toString().strip();
      options.add(option);
      if (entry.getValue() != null) {
        String description = entry.getValue().toString().strip();
        if (!description.isEmpty() && !description.equals(option)) {
          rubric.put(option, description);
        }
      }
    }
    return new Choice(instructions, options, rubric);
  }

  private static Noul parseNoul(String instructions, Object rawCriteria) {
    if (!(rawCriteria instanceof Map<?, ?> criteriaMap) || criteriaMap.isEmpty()) {
      return new Noul(instructions);
    }
    Map<String, String> rubric = new LinkedHashMap<>();
    for (Map.Entry<?, ?> entry : criteriaMap.entrySet()) {
      if (entry.getKey() != null && entry.getValue() != null) {
        String key = entry.getKey().toString().strip();
        String description = entry.getValue().toString().strip();
        if ((key.equals("false") || key.equals("true"))
            && !description.isEmpty()
            && !description.equals(key)) {
          rubric.put(key, description);
        }
      }
    }
    return new Noul(instructions, rubric);
  }

  private static Score parseScore(String questionId, String instructions, Object rawCriteria)
      throws ApplicationException {
    if (!(rawCriteria instanceof Map<?, ?> criteriaMap) || criteriaMap.isEmpty()) {
      throw new ApplicationException(
          "Score question '" + questionId + "' requires a non-empty 'criteria' map");
    }
    List<String> levels = new ArrayList<>(criteriaMap.size());
    Map<String, String> rubric = new LinkedHashMap<>();
    for (Map.Entry<?, ?> entry : criteriaMap.entrySet()) {
      if (entry.getKey() == null) {
        throw new ApplicationException(
            "Score question '" + questionId + "' contains a null level key");
      }
      String level = entry.getKey().toString().strip();
      levels.add(level);
      if (entry.getValue() != null) {
        String description = entry.getValue().toString().strip();
        if (!description.isEmpty() && !description.equals(level)) {
          rubric.put(level, description);
        }
      }
    }
    return new Score(instructions, levels, rubric);
  }

  private static String extractStateText(Object state) throws ApplicationException {
    if (state == null) {
      throw new ApplicationException("TypeSafe routing request state must not be null");
    }
    String text = state instanceof Builder builder ? builder.toString() : state.toString();
    if (text.isBlank()) {
      throw new ApplicationException("TypeSafe routing request state must not be blank");
    }
    return text;
  }

  private static String defaultModelName(SharedPrefixInferenceBackend backend) {
    Objects.requireNonNull(backend, "backend");
    String name = backend.name();
    return name != null && !name.isBlank() ? name : DEFAULT_MODEL_VERSION;
  }

  private static String requireText(String value, String name) {
    Objects.requireNonNull(value, name);
    if (value.isBlank()) {
      throw new IllegalArgumentException(name + " must not be blank");
    }
    return value;
  }

  private static int estimateTokens(String text) {
    return Math.max(1, (text.length() + 3) / 4);
  }

  /** Releases the underlying backend if one is held. */
  @Override
  public void close() {
    if (backend == null) {
      return;
    }
    synchronized (backendMonitor) {
      backend.close();
    }
  }

  private record ParsedQuestion(String type, AnswerSpace space) {}
}
