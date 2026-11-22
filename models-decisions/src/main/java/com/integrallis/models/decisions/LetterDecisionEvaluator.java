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
package com.integrallis.models.decisions;

import com.integrallis.models.api.InferenceSession;
import com.integrallis.models.api.SharedInferencePrefix;
import com.integrallis.models.api.SharedPrefixInferenceBackend;
import com.integrallis.models.api.Tokenizer;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Answers typed questions about one state by showing the model each question's options as a
 * lettered list and reading the distribution over the answer letters.
 *
 * <p>This is the path for questions that are not known in advance: the option text is read from the
 * prompt, so any set of options is handled, which a head fitted over a hidden state is not (see
 * {@link LetterLogitScorer}). Nothing is generated; each question costs one read of its own suffix.
 *
 * <p>With more than one question and a backend that supports it, the state is read once, frozen,
 * and every question forks that prefix. Otherwise the state is read once per question, and the
 * result says which happened rather than leaving it to be inferred from timing.
 *
 * <p>The prompt is the state followed by {@link #questionSuffix(AnswerSpace)}. That layout is the
 * arrangement {@link LetterLogitScorer#renderCriteria} records as measured best, and it is one
 * definition on purpose: every caller that scores letters builds its prompt here, so the same
 * question cannot be worded two ways through two entry points.
 *
 * <p>Not thread-safe. The backend is stateful; a caller that shares it must serialise calls.
 */
public final class LetterDecisionEvaluator {

  private final SharedPrefixInferenceBackend backend;
  private final LetterLogitScorer scorer;

  /**
   * Creates an evaluator over a loaded backend.
   *
   * @param backend the backend whose sessions the questions are asked in
   * @param temperature positive, finite calibration temperature applied across the letters
   */
  public LetterDecisionEvaluator(SharedPrefixInferenceBackend backend, double temperature) {
    this.backend = Objects.requireNonNull(backend, "backend");
    this.scorer = new LetterLogitScorer(temperature);
  }

  /**
   * Answers every question about {@code state}.
   *
   * @param state the text all questions are asked about
   * @param spaces the questions, each with its answer space
   * @return one verdict per space, in order, with the cost and sharing evidence
   * @throws IllegalArgumentException if the state encodes to no tokens or there are no questions
   */
  public LetterDecisionBatch evaluate(String state, List<? extends AnswerSpace> spaces) {
    Objects.requireNonNull(state, "state");
    Objects.requireNonNull(spaces, "spaces");
    if (spaces.isEmpty()) {
      throw new IllegalArgumentException("at least one question is required");
    }
    Tokenizer tokenizer = backend.tokenizer();
    int[] stateTokens = tokenizer.encode(state);
    if (stateTokens.length == 0) {
      throw new IllegalArgumentException("the state encoded to no tokens");
    }
    List<Verdict> verdicts = new ArrayList<>(spaces.size());
    int inputTokens = stateTokens.length;
    int statePrefills = 0;
    boolean sharedPrefixProven = false;

    if (spaces.size() > 1 && backend.supportsSharedPrefixes()) {
      SharedInferencePrefix prefix;
      try (InferenceSession source = backend.openSession()) {
        backend.prefill(source, stateTokens, 0);
        statePrefills = 1;
        prefix = backend.freezePrefix(source);
      }
      boolean allShared = true;
      try (InferenceSession witness = backend.fork(prefix)) {
        for (AnswerSpace space : spaces) {
          int[] suffixTokens = tokenizer.encode(questionSuffix(space));
          inputTokens += suffixTokens.length;
          try (InferenceSession branch = backend.fork(prefix)) {
            allShared &= backend.sharesPrefixStorage(witness, branch);
            float[] logits = backend.prefill(branch, suffixTokens, stateTokens.length);
            verdicts.add(score(tokenizer, space, logits));
          }
        }
      }
      sharedPrefixProven = allShared;
    } else {
      for (AnswerSpace space : spaces) {
        int[] promptTokens = tokenizer.encode(state + questionSuffix(space));
        inputTokens += Math.max(0, promptTokens.length - stateTokens.length);
        try (InferenceSession session = backend.openSession()) {
          float[] logits = backend.prefill(session, promptTokens, 0);
          statePrefills++;
          verdicts.add(score(tokenizer, space, logits));
        }
      }
    }
    return new LetterDecisionBatch(verdicts, inputTokens, statePrefills, sharedPrefixProven);
  }

  /**
   * The text appended to the state to ask one question: the rubric if there is one, then the
   * question, then the lettered options and the answer cue.
   *
   * @param space the question and its answer space
   * @return the suffix, starting with a newline
   */
  public static String questionSuffix(AnswerSpace space) {
    Objects.requireNonNull(space, "space");
    List<String> labels = space.labels();
    String criteriaBlock = LetterLogitScorer.renderCriteria(labels, space.criteria());
    StringBuilder suffix = new StringBuilder();
    if (!criteriaBlock.isEmpty()) {
      suffix.append('\n').append(criteriaBlock);
    }
    return suffix
        .append('\n')
        .append(space.question())
        .append('\n')
        .append(LetterLogitScorer.renderOptions(labels))
        .toString();
  }

  /**
   * The token id of each answer letter, {@code A} first: the last token of the letter preceded by a
   * space, which is how it appears after the answer cue.
   *
   * @param tokenizer the loaded model's tokenizer
   * @param count how many letters
   * @return one token id per letter
   * @throws IllegalStateException if the tokenizer produces no token for a letter
   */
  public static int[] letterTokens(Tokenizer tokenizer, int count) {
    Objects.requireNonNull(tokenizer, "tokenizer");
    int[] letterTokens = new int[count];
    for (int slot = 0; slot < count; slot++) {
      int[] encoded = tokenizer.encode(" " + (char) ('A' + slot));
      if (encoded.length == 0) {
        throw new IllegalStateException("Tokenizer produced no token for option letter " + slot);
      }
      letterTokens[slot] = encoded[encoded.length - 1];
    }
    return letterTokens;
  }

  private Verdict score(Tokenizer tokenizer, AnswerSpace space, float[] logits) {
    return scorer.score(space, logits, letterTokens(tokenizer, space.size()));
  }
}
