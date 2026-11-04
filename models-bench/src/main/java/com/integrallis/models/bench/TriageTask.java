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
package com.integrallis.models.bench;

import com.integrallis.models.decisions.Choice;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.TreeMap;

/**
 * A "route this message to a team" decision task, read from data rather than compiled in.
 *
 * <p>The outcomes, their rules and the messages all come from files, because the task this exists
 * to answer is somebody else's: external work reports category accuracy on public customer-support
 * corpora, and the only way to compare is to run <i>their</i> labels, not a set written here. A
 * corpus we wrote would score our own assumptions back to us.
 *
 * <p>Every outcome must carry a rule. That is a refusal, not a default: measured over 120 JevBench
 * items on the shipped model, the same declarations scored 0.7500 with no rubric and 0.9000 with
 * the rubric before the criterion and the letters after it (see {@code
 * LetterLogitScorer#renderCriteria}). A run that silently dropped the rules would measure the
 * prompt and be read as a measurement of the model, so a label with no rule fails the load.
 */
final class TriageTask {

  /** One labelled message: the text, and the outcome a human assigned it. */
  record Message(String text, String expected) {

    Message {
      text = Objects.requireNonNull(text, "text").strip();
      expected = normalise(expected);
      if (text.isEmpty()) {
        throw new IllegalArgumentException("message text must not be blank");
      }
    }
  }

  private final String question;
  private final List<String> outcomes;
  private final Map<String, String> criteria;
  private final List<Message> messages;

  private TriageTask(
      String question,
      List<String> outcomes,
      Map<String, String> criteria,
      List<Message> messages) {
    this.question = question;
    this.outcomes = List.copyOf(outcomes);
    this.criteria = Map.copyOf(criteria);
    this.messages = List.copyOf(messages);
  }

  /**
   * Loads a task from a messages file and a rules file.
   *
   * <p>Both are two-column tab-separated: {@code outcome<TAB>text} for messages, {@code
   * outcome<TAB>rule} for rules. Blank lines and lines starting with {@code #} are skipped. The
   * outcome order is the rules file's order, which fixes the letter each outcome is rendered as, so
   * reordering that file is a different prompt and a different measurement.
   *
   * @param messagesFile labelled messages
   * @param rulesFile one rule per outcome
   * @param question the criterion put to the model
   * @param limit at most this many messages, or 0 for all
   */
  static TriageTask load(Path messagesFile, Path rulesFile, String question, int limit) {
    Map<String, String> rules = readPairs(rulesFile, "rules");
    if (rules.isEmpty()) {
      throw new IllegalArgumentException("no rules declared in " + rulesFile);
    }
    List<String> outcomes = new ArrayList<>(rules.keySet());

    List<Message> messages = new ArrayList<>();
    for (String[] row : readRows(messagesFile, "messages")) {
      messages.add(new Message(row[1], row[0]));
      if (limit > 0 && messages.size() >= limit) {
        break;
      }
    }
    if (messages.isEmpty()) {
      throw new IllegalArgumentException("no messages in " + messagesFile);
    }

    // A message whose label has no rule would be scored against a space that cannot contain it,
    // which reads as a wrong answer rather than as a broken task. Name every offender at once:
    // finding them one run at a time is how a long corpus eats an afternoon.
    Set<String> unknown = new LinkedHashSet<>();
    for (Message message : messages) {
      if (!rules.containsKey(message.expected())) {
        unknown.add(message.expected());
      }
    }
    if (!unknown.isEmpty()) {
      throw new IllegalArgumentException(
          "these labels appear in "
              + messagesFile
              + " but have no rule in "
              + rulesFile
              + ": "
              + String.join(", ", unknown));
    }
    return new TriageTask(requireQuestion(question), outcomes, rules, messages);
  }

  String question() {
    return question;
  }

  List<String> outcomes() {
    return outcomes;
  }

  Map<String, String> criteria() {
    return criteria;
  }

  List<Message> messages() {
    return messages;
  }

  /** The answer space this task puts to a model, rules included. */
  Choice space() {
    return new Choice(question, outcomes, criteria);
  }

  /**
   * Accuracy, and the coverage and accuracy a confidence gate would deliver.
   *
   * <p>Reported the way the comparable external report does -- how many cases a gate at a threshold
   * decides automatically, and how accurate those are -- because a bare accuracy figure says
   * nothing about whether the thing is usable behind a gate, which is the only reason to calibrate.
   *
   * @param chosen the outcome each message was assigned, in message order
   * @param confidence the probability attached to each choice, in message order
   * @param gate the confidence threshold to report coverage at
   */
  Scored score(List<String> chosen, List<Double> confidence, double gate) {
    if (chosen.size() != messages.size() || confidence.size() != messages.size()) {
      throw new IllegalArgumentException(
          "need one answer and one confidence per message: expected "
              + messages.size()
              + ", got "
              + chosen.size()
              + " and "
              + confidence.size());
    }
    int correct = 0;
    int gated = 0;
    int gatedCorrect = 0;
    Map<String, int[]> perOutcome = new TreeMap<>();
    Map<String, Integer> confusions = new LinkedHashMap<>();
    for (int index = 0; index < messages.size(); index++) {
      String expected = messages.get(index).expected();
      String actual = normalise(chosen.get(index));
      boolean hit = expected.equals(actual);
      int[] tally = perOutcome.computeIfAbsent(expected, key -> new int[2]);
      tally[1]++;
      if (hit) {
        correct++;
        tally[0]++;
      } else {
        confusions.merge(expected + " -> " + actual, 1, Integer::sum);
      }
      if (confidence.get(index) >= gate) {
        gated++;
        if (hit) {
          gatedCorrect++;
        }
      }
    }
    return new Scored(
        messages.size(),
        correct / (double) messages.size(),
        gate,
        gated / (double) messages.size(),
        gated == 0 ? Double.NaN : gatedCorrect / (double) gated,
        Map.copyOf(perOutcome),
        Map.copyOf(confusions));
  }

  /** What a run of this task produced. */
  record Scored(
      int cases,
      double accuracy,
      double gate,
      double coverageAtGate,
      double accuracyAtGate,
      Map<String, int[]> perOutcome,
      Map<String, Integer> confusions) {}

  private static String requireQuestion(String question) {
    String stripped = Objects.requireNonNull(question, "question").strip();
    if (stripped.isEmpty()) {
      throw new IllegalArgumentException("question must not be blank");
    }
    return stripped;
  }

  /** Upper-cases and trims a label so a corpus's casing cannot read as a different outcome. */
  private static String normalise(String label) {
    String stripped = Objects.requireNonNull(label, "label").strip().toUpperCase(Locale.ROOT);
    if (stripped.isEmpty()) {
      throw new IllegalArgumentException("label must not be blank");
    }
    return stripped;
  }

  private static Map<String, String> readPairs(Path file, String what) {
    Map<String, String> pairs = new LinkedHashMap<>();
    for (String[] row : readRows(file, what)) {
      if (pairs.put(normalise(row[0]), row[1]) != null) {
        throw new IllegalArgumentException("duplicate label " + row[0] + " in " + file);
      }
    }
    return pairs;
  }

  private static List<String[]> readRows(Path file, String what) {
    Objects.requireNonNull(file, what);
    List<String> lines;
    try {
      lines = Files.readAllLines(file, StandardCharsets.UTF_8);
    } catch (IOException failure) {
      throw new UncheckedIOException("cannot read " + what + " from " + file, failure);
    }
    List<String[]> rows = new ArrayList<>();
    for (int number = 1; number <= lines.size(); number++) {
      String line = lines.get(number - 1);
      if (line.isBlank() || line.startsWith("#")) {
        continue;
      }
      int tab = line.indexOf('\t');
      if (tab < 0) {
        throw new IllegalArgumentException(
            file + " line " + number + " is not two tab-separated columns: " + line);
      }
      String first = line.substring(0, tab).strip();
      String second = line.substring(tab + 1).strip();
      if (first.isEmpty() || second.isEmpty()) {
        throw new IllegalArgumentException(file + " line " + number + " has an empty column");
      }
      rows.add(new String[] {first, second});
    }
    return rows;
  }
}
