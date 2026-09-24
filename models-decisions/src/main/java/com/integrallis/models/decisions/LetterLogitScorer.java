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

import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Reads a typed decision from one position's logits, over the answer letters alone.
 *
 * <p>The options are written into the prompt as a lettered list and the answer is the distribution
 * over {@code A}, {@code B}, {@code C} at the position where the letter would go. One forward pass
 * answers the whole question however many options it has, and the option text is read from the
 * prompt rather than learned, so a set of options the model has never seen is handled on the same
 * footing as a familiar one.
 *
 * <p>That is the part this project got wrong for a long time. A head fitted over a frozen hidden
 * state learns the labels it was trained on and scores chance on any other set; a candidate scorer
 * spends a forward pass per option. Both were measured on 2026-09-21 and both are beaten by simply
 * asking the model which letter comes next.
 *
 * <p>The renormalisation over the letters is the whole calibration story: the model's mass on
 * everything that is not a letter is discarded, so the returned distribution is conditional on
 * answering the question at all.
 */
public final class LetterLogitScorer {

  /** A, B, C, ... The cap is the alphabet; a Choice may declare up to 255 options. */
  private static final int MAX_LETTERS = 26;

  private final double temperature;

  /**
   * Creates a scorer.
   *
   * @param temperature the calibration temperature applied across the letters
   */
  public LetterLogitScorer(double temperature) {
    if (!Double.isFinite(temperature) || temperature <= 0.0) {
      throw new IllegalArgumentException("temperature must be positive and finite: " + temperature);
    }
    this.temperature = temperature;
  }

  /**
   * Renders the options as the lettered list the prompt must end with.
   *
   * <p>Letters and answer cue only. Any rubric goes in its own block before the criterion; see
   * {@link #renderCriteria(List, Map)} for why.
   */
  public static String renderOptions(List<String> options) {
    Objects.requireNonNull(options, "options");
    requireRenderable(options.size());
    StringBuilder text = new StringBuilder();
    for (int index = 0; index < options.size(); index++) {
      text.append((char) ('A' + index)).append(": ").append(options.get(index)).append('\n');
    }
    return text.append("Answer:").toString();
  }

  /**
   * Renders the rubric alone, with no letters and no answer cue.
   *
   * <p>Separate from {@link #renderOptions(List)} because the two belong in different parts of the
   * prompt, and that is measured rather than assumed. MEASURED 2026-09-24 over 120 JevBench items
   * on the shipped Qwen3.5-4B, four arrangements of the same declarations:
   *
   * <ul>
   *   <li>no rubric at all: accuracy 0.7500, Intelligence 72.2
   *   <li>rubric and letters both after the criterion: 0.8750, 86.1
   *   <li>rubric and letters both before it: 0.8000, 79.6
   *   <li><b>rubric before the criterion, letters after it: 0.9000, 88.9</b>
   * </ul>
   *
   * <p>Against a noise floor of about one point. The last is the best measured and it is also the
   * cheapest, because the rubric is most of the added tokens and everything before the criterion is
   * shared across every question about one piece of evidence. Moving the letters as well undid it
   * -- {@code fact} fell from 1.0000 to 0.3333 -- so what a model needs after the question is the
   * letter-to-label mapping, and what it is happy to have read beforehand is what the labels mean.
   *
   * @param options the outcomes, in declaration order
   * @param criteria what each outcome covers, keyed by label; any subset, possibly empty
   * @return the rubric block with no trailing newline, or an empty string when nothing is declared
   */
  public static String renderCriteria(List<String> options, Map<String, String> criteria) {
    Objects.requireNonNull(options, "options");
    Objects.requireNonNull(criteria, "criteria");
    requireRenderable(options.size());
    if (criteria.isEmpty()) {
      return "";
    }
    StringBuilder text = new StringBuilder("Options:");
    for (String label : options) {
      // An option the rubric does not cover repeats its own label, so every option is listed and
      // none is silently absent from the block that is supposed to explain them.
      text.append("\n- ").append(label).append(": ").append(criteria.getOrDefault(label, label));
    }
    return text.toString();
  }

  /**
   * Turns one position's logits into a verdict over the space's labels.
   *
   * @param space the answer space, whose labels are in the same order as the rendered letters
   * @param logits the full vocabulary logits at the answer position
   * @param letterTokens the token id of each answer letter, in the same order
   */
  public Verdict score(AnswerSpace space, float[] logits, int[] letterTokens) {
    Objects.requireNonNull(space, "space");
    Objects.requireNonNull(logits, "logits");
    Objects.requireNonNull(letterTokens, "letterTokens");
    int outcomes = space.size();
    requireRenderable(outcomes);
    if (letterTokens.length != outcomes) {
      throw new IllegalArgumentException(
          "need one letter token per label: " + outcomes + ", got " + letterTokens.length);
    }

    double[] selected = new double[outcomes];
    for (int index = 0; index < outcomes; index++) {
      int token = letterTokens[index];
      if (token < 0 || token >= logits.length) {
        throw new IllegalArgumentException(
            "letter token " + token + " is outside the vocabulary of " + logits.length);
      }
      selected[index] = logits[token];
    }
    // Softmax over the letters only. Everything the model would rather say is discarded, so the
    // result is a distribution over answers rather than over the vocabulary.
    return new Verdict(space, Calibration.softmax(selected, temperature));
  }

  private static void requireRenderable(int outcomes) {
    if (outcomes < 2) {
      throw new IllegalArgumentException("a decision needs at least two options: " + outcomes);
    }
    if (outcomes > MAX_LETTERS) {
      throw new IllegalArgumentException(
          "letter scoring covers " + MAX_LETTERS + " options, asked for " + outcomes);
    }
  }
}
