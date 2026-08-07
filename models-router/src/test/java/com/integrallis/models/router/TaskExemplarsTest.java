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
package com.integrallis.models.router;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.io.IOException;
import java.io.Reader;
import java.io.StringReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;

class TaskExemplarsTest {

  /** The farmed corpus lives beside the module rather than on the classpath. */
  private static final Path CORPUS = Path.of("corpus", "benchmark-prompts.tsv");

  static boolean corpusPresent() {
    return Files.isReadable(CORPUS);
  }

  private static TaskExemplars corpus() throws IOException {
    try (Reader reader = Files.newBufferedReader(CORPUS, StandardCharsets.UTF_8)) {
      return TaskExemplars.parse(reader);
    }
  }

  @Test
  void parsesTheFourColumnFormat() {
    TaskExemplars exemplars =
        TaskExemplars.parse(
            new StringReader(
                "# a comment\n\ntrain\tcode\tmbpp\twrite a parser\neval\tcode\tmbpp\tsort a list\n"));

    assertThat(exemplars.taskNames()).containsExactly("code");
    assertThat(exemplars.trainingPrompts().get("code"))
        .containsExactly(new TaskExemplars.Labelled("code", "mbpp", "write a parser"));
    assertThat(exemplars.evaluationPrompts())
        .containsExactly(new TaskExemplars.Labelled("code", "mbpp", "sort a list"));
  }

  @Test
  void rejectsAMalformedLine() {
    assertThatThrownBy(() -> TaskExemplars.parse(new StringReader("train\tcode\tmbpp\n")))
        .isInstanceOf(IllegalStateException.class)
        .hasMessageContaining("malformed exemplar on line 1");
  }

  @Test
  void rejectsAnUnknownSplit() {
    assertThatThrownBy(
            () -> TaskExemplars.parse(new StringReader("holdout\tcode\tmbpp\tparse this\n")))
        .isInstanceOf(IllegalStateException.class)
        .hasMessageContaining("unknown split 'holdout'");
  }

  @Test
  void rejectsACorpusWithNoTrainingPrompts() {
    assertThatThrownBy(() -> TaskExemplars.parse(new StringReader("eval\tcode\tmbpp\tsort a list\n")))
        .isInstanceOf(IllegalStateException.class)
        .hasMessageContaining("no training prompts");
  }

  @Test
  @EnabledIf("corpusPresent")
  void everyTaskCarriesEnoughTrainingPromptsToBeSeparable() throws IOException {
    // A task represented by a handful of prompts describes a phrasing, not a task; the classifier
    // would then match wording rather than intent.
    corpus()
        .trainingPrompts()
        .forEach((task, prompts) -> assertThat(prompts).as(task).hasSizeGreaterThanOrEqualTo(100));
  }

  @Test
  @EnabledIf("corpusPresent")
  void everyTaskIsRepresentedInBothSplits() throws IOException {
    TaskExemplars exemplars = corpus();
    Set<String> evaluated = new HashSet<>();
    exemplars.evaluationPrompts().forEach(labelled -> evaluated.add(labelled.task()));

    assertThat(evaluated).isEqualTo(exemplars.taskNames());
  }

  @Test
  @EnabledIf("corpusPresent")
  void heldOutPromptsNeverAppearInTraining() throws IOException {
    // An eval prompt that also trained the index measures memorisation, not accuracy.
    TaskExemplars exemplars = corpus();
    Set<String> training = new HashSet<>();
    exemplars
        .trainingPrompts()
        .values()
        .forEach(prompts -> prompts.forEach(p -> training.add(p.prompt())));

    List<String> leaked =
        exemplars.evaluationPrompts().stream()
            .map(TaskExemplars.Labelled::prompt)
            .filter(training::contains)
            .toList();

    assertThat(leaked).isEmpty();
  }

  @Test
  @EnabledIf("corpusPresent")
  void noPromptIsLabelledWithTwoDifferentTasks() throws IOException {
    // A prompt appearing under two tasks teaches the classifier to contradict itself.
    TaskExemplars exemplars = corpus();
    Set<String> seen = new HashSet<>();
    Set<String> duplicated = new HashSet<>();
    exemplars
        .trainingPrompts()
        .values()
        .forEach(
            prompts ->
                prompts.forEach(
                    p -> {
                      if (!seen.add(p.prompt())) {
                        duplicated.add(p.prompt());
                      }
                    }));

    assertThat(duplicated).isEmpty();
  }
}
