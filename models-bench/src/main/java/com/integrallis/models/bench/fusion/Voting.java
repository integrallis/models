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
package com.integrallis.models.bench.fusion;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.ToDoubleFunction;

/**
 * Answer-level aggregation rules.
 *
 * <p>Answers are grouped by the dataset's equivalence. The group with the largest total weight
 * wins. Ties go to the tied group containing the answer of the first member in {@code priority}
 * (the designated best member first, then the arm's member order); abstentions never win.
 */
public final class Voting {

  private Voting() {}

  private static final class Group {
    final String representative;
    final List<String> voters = new ArrayList<>();
    double weight;

    Group(String representative) {
      this.representative = representative;
    }
  }

  /** Plain majority vote. */
  public static String majority(
      List<Vote> votes, AnswerExtractor equivalence, List<String> priority) {
    return aggregate(votes, equivalence, priority, vote -> 1.0);
  }

  /** Majority over votes whose reasoning answer agrees with the stated answer. */
  public static String consistencyFiltered(
      List<Vote> votes, AnswerExtractor equivalence, List<String> priority) {
    List<Vote> kept = votes.stream().filter(v -> v.answer() != null && v.consistent()).toList();
    boolean anyAnswer = votes.stream().anyMatch(v -> v.answer() != null);
    return aggregate(
        kept.isEmpty() && anyAnswer ? votes : kept, equivalence, priority, vote -> 1.0);
  }

  /** Votes weighted by calibrated confidence {@code exp(c / τ_member)}. */
  public static String confidenceWeighted(
      List<Vote> votes,
      AnswerExtractor equivalence,
      List<String> priority,
      Map<String, Double> temperatures) {
    return aggregate(
        votes,
        equivalence,
        priority,
        vote -> {
          if (Double.isNaN(vote.confidence())) {
            return 0.0;
          }
          Double temperature = temperatures.get(vote.voter());
          if (temperature == null) {
            throw new IllegalStateException("no calibration temperature for " + vote.voter());
          }
          return ConfidenceCalibration.probability(vote.confidence(), temperature);
        });
  }

  /** Majority over repeated samples; ties go to the answer that appeared first. */
  public static String selfConsistency(List<String> answers, AnswerExtractor equivalence) {
    List<Vote> votes = new ArrayList<>();
    List<String> order = new ArrayList<>();
    for (int i = 0; i < answers.size(); i++) {
      votes.add(new Vote("s" + i, answers.get(i), true, Double.NaN));
      order.add("s" + i);
    }
    return aggregate(votes, equivalence, order, vote -> 1.0);
  }

  private static String aggregate(
      List<Vote> votes,
      AnswerExtractor equivalence,
      List<String> priority,
      ToDoubleFunction<Vote> weight) {
    List<Group> groups = new ArrayList<>();
    for (Vote vote : votes) {
      if (vote.answer() == null) {
        continue;
      }
      Group group = null;
      for (Group candidate : groups) {
        if (equivalence.equivalent(candidate.representative, vote.answer())) {
          group = candidate;
          break;
        }
      }
      if (group == null) {
        group = new Group(vote.answer());
        groups.add(group);
      }
      group.voters.add(vote.voter());
      group.weight += weight.applyAsDouble(vote);
    }
    if (groups.isEmpty()) {
      return null;
    }
    double best = groups.stream().mapToDouble(g -> g.weight).max().orElseThrow();
    List<Group> tied = groups.stream().filter(g -> Double.compare(g.weight, best) == 0).toList();
    if (tied.size() == 1) {
      return tied.getFirst().representative;
    }
    for (String voter : priority) {
      for (Group group : tied) {
        if (group.voters.contains(voter)) {
          return group.representative;
        }
      }
    }
    return tied.getFirst().representative;
  }
}
