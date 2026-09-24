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

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

/**
 * A LABEL IS A TOKEN, NOT AN EXPLANATION.
 *
 * <p>MEASURED 2026-09-24 over 120 JevBench items on the shipped Qwen3.5-4B, same kernel, the only
 * difference being whether a per-label rubric reaches the prompt: accuracy 0.8917 with it against
 * 0.7500 without, Intelligence 88.0 against 72.2. Ordinal questions fell from 0.9167 to 0.2500,
 * which is what "A: 3" means to a reader who was never told what 3 is. For scale, swapping an
 * entire matrix kernel moves one item in that same cohort.
 *
 * <p>The benchmark had always handed every system that rubric. This API had nowhere to put one, so
 * the published score belonged to a prompt the shipped code could not build. These assert the
 * rubric survives from declaration to prompt, which is the whole of the fix.
 */
final class AnswerSpaceCriteriaTest {

  @Test
  void aRubricReachesTheRenderedOptions() {
    Noul space =
        new Noul(
            "Does the cap apply to data breaches?",
            "The cap is stated and no carve-out removes breaches from it",
            "There is no cap, or a carve-out puts breaches outside it");

    String rendered = LetterLogitScorer.renderOptions(space.labels(), space.criteria());

    assertThat(rendered)
        .isEqualTo(
            """
            Options:
            - false: There is no cap, or a carve-out puts breaches outside it
            - true: The cap is stated and no carve-out removes breaches from it
            A: false
            B: true
            Answer:""");
  }

  @Test
  void aSpaceWithoutARubricRendersExactlyAsItAlwaysDid() {
    List<String> options = List.of("billing", "technical");

    assertThat(LetterLogitScorer.renderOptions(options, Map.of()))
        .isEqualTo(LetterLogitScorer.renderOptions(options))
        .isEqualTo("A: billing\nB: technical\nAnswer:");
  }

  @Test
  void aPartialRubricAnnotatesOnlyWhatItCovers() {
    Choice space =
        new Choice(
            "Which queue?",
            List.of("billing", "technical"),
            Map.of("billing", "Asks about a charge, invoice or payment"));

    // The option the rubric does not cover repeats its own label rather than vanishing from the
    // list, so the model still sees every option it is allowed to pick.
    assertThat(LetterLogitScorer.renderOptions(space.labels(), space.criteria()))
        .isEqualTo(
            """
            Options:
            - billing: Asks about a charge, invoice or payment
            - technical: technical
            A: billing
            B: technical
            Answer:""");
  }

  /**
   * A key that is not a label is rejected rather than dropped.
   *
   * <p>Silently ignoring it produces a prompt that still forms and an answer that still looks well
   * shaped, with the rule that was supposed to govern the option simply missing. That is the
   * failure mode this whole class exists to close, so it must not be reintroduced by a typo.
   */
  @Test
  void aRubricForAnOptionThatDoesNotExistIsRejected() {
    assertThatThrownBy(
            () ->
                new Choice(
                    "Which queue?",
                    List.of("billing", "technical"),
                    Map.of("blling", "Asks about a charge")))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("blling");

    assertThatThrownBy(() -> new Score("How severe?", List.of("low", "high"), Map.of("low", "  ")))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("blank");
  }

  /** The map handed back must reject mutation, since nothing copies it on every read. */
  @Test
  void theRubricHandedBackCannotBeMutated() {
    Map<String, String> mutable = new java.util.LinkedHashMap<>();
    mutable.put("low", "Recoverable in a day");
    Score space = new Score("How severe?", List.of("low", "high"), mutable);

    mutable.put("high", "Not recoverable");

    assertThat(space.criteria()).containsOnlyKeys("low");
    assertThatThrownBy(() -> space.criteria().put("high", "Not recoverable"))
        .isInstanceOf(UnsupportedOperationException.class);
  }

  @Test
  void theRubricIsPartOfWhatWasAsked() {
    Score withRubric =
        new Score("How severe?", List.of("low", "high"), Map.of("low", "Recoverable in a day"));
    Score without = new Score("How severe?", List.of("low", "high"));

    // Same question, same levels, different decision. An artifact that recorded one as the other
    // would not say what was actually asked.
    assertThat(withRubric).isNotEqualTo(without);
    assertThat(withRubric.criteria()).containsExactly(Map.entry("low", "Recoverable in a day"));
    assertThat(without.criteria()).isEmpty();
  }
}
