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

import java.io.BufferedReader;
import java.io.IOException;
import java.io.Reader;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * A labelled corpus of prompts, read from the tab-separated format under {@code models-router/
 * corpus}.
 *
 * <p>The corpus is not shipped in the jar. What ships is the index built from it — prompt text is
 * only needed to produce embeddings, and embeddings are what the classifier searches. Keeping the
 * text in the repository instead means the labels stay reviewable and the index stays rebuildable.
 *
 * <p>Prompts carry a split: {@code train} builds the index, {@code eval} is held out and read only
 * by the accuracy gate. Each also records the dataset it came from, so a task that turns out to be
 * badly classified can be traced back to its source.
 *
 * <p>Callers wanting to classify against their own tasks can parse their own corpus in the same
 * format rather than using the one this project ships.
 */
public final class TaskExemplars {

  private static final String TRAIN = "train";
  private static final String EVAL = "eval";
  private static final int FIELDS = 4;

  private final Map<String, List<Labelled>> train;
  private final List<Labelled> evaluation;

  private TaskExemplars(Map<String, List<Labelled>> train, List<Labelled> evaluation) {
    this.train = train;
    this.evaluation = evaluation;
  }

  /**
   * Parses a corpus.
   *
   * <p>Format is {@code split<TAB>task<TAB>source<TAB>prompt}; blank lines and lines beginning with
   * {@code #} are ignored.
   *
   * @param source the corpus text
   * @return the parsed exemplars
   * @throws IllegalStateException if a line is malformed, a split name is unknown, or the corpus
   *     holds no training prompts
   */
  public static TaskExemplars parse(Reader source) {
    Objects.requireNonNull(source, "source");
    Map<String, List<Labelled>> train = new LinkedHashMap<>();
    List<Labelled> evaluation = new ArrayList<>();
    try (BufferedReader reader = new BufferedReader(source)) {
      String line;
      int number = 0;
      while ((line = reader.readLine()) != null) {
        number++;
        if (line.isBlank() || line.charAt(0) == '#') {
          continue;
        }
        String[] fields = line.split("\t", -1);
        if (fields.length != FIELDS || fields[1].isBlank() || fields[3].isBlank()) {
          throw new IllegalStateException("malformed exemplar on line " + number + ": " + line);
        }
        Labelled labelled = new Labelled(fields[1], fields[2], fields[3]);
        if (TRAIN.equals(fields[0])) {
          train.computeIfAbsent(labelled.task(), key -> new ArrayList<>()).add(labelled);
        } else if (EVAL.equals(fields[0])) {
          evaluation.add(labelled);
        } else {
          throw new IllegalStateException("unknown split '" + fields[0] + "' on line " + number);
        }
      }
    } catch (IOException e) {
      throw new UncheckedIOException("cannot read exemplars", e);
    }
    if (train.isEmpty()) {
      throw new IllegalStateException("corpus contains no training prompts");
    }
    Map<String, List<Labelled>> frozen = new LinkedHashMap<>();
    train.forEach((task, prompts) -> frozen.put(task, List.copyOf(prompts)));
    return new TaskExemplars(Map.copyOf(frozen), List.copyOf(evaluation));
  }

  /**
   * Training prompts by task name.
   *
   * @return an immutable map from task name to its prompts
   */
  public Map<String, List<Labelled>> trainingPrompts() {
    // Already immutable; copyOf on an immutable map returns the same instance, so this is a
    // statement of intent rather than a copy.
    return Map.copyOf(train);
  }

  /**
   * Held-out prompts, read by the accuracy gate rather than used to build the index.
   *
   * @return an immutable list of labelled prompts
   */
  public List<Labelled> evaluationPrompts() {
    return List.copyOf(evaluation);
  }

  /**
   * Task names present in the training split.
   *
   * @return an immutable set of task names
   */
  public Set<String> taskNames() {
    return train.keySet();
  }

  /**
   * One labelled prompt.
   *
   * @param task the task name this prompt belongs to
   * @param source the dataset it was taken from
   * @param prompt the prompt text
   */
  public record Labelled(String task, String source, String prompt) {

    /** Validates every field. */
    public Labelled {
      Objects.requireNonNull(task, "task");
      Objects.requireNonNull(source, "source");
      Objects.requireNonNull(prompt, "prompt");
    }
  }
}
