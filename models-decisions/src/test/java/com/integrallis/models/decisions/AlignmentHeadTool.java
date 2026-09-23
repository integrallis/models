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

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Trains one alignment head that scores any candidate against any state.
 *
 * <p>A fixed-label head learns the labels themselves, so it cannot answer a question whose options
 * it has never seen. That is fatal for a benchmark: JevBench's choice items use three different
 * option sets, and a head trained on one of them has nothing to say about the others.
 *
 * <p>This head reads the <em>interaction</em> between a state and a candidate instead — their
 * elementwise product and absolute difference — and answers one binary question: does this
 * candidate fit this state. Nothing in those features names a label, so a label the head has never
 * encountered is scored on the same footing as one it has. {@link CandidateScorer} then normalises
 * the independent scores within a question.
 *
 * <p>Whether that transfer actually happens is the thing being measured here, not assumed. The tool
 * reports accuracy on the label set it trained on and on label sets held out entirely.
 */
public final class AlignmentHeadTool {

  private AlignmentHeadTool() {}

  /**
   * Arguments: state harvest, label harvest, training label-set name, comma-separated eval specs,
   * optional L2.
   *
   * <p>An eval spec is {@code labelSet} to score the training harvest's own sealed items, or {@code
   * labelSet@harvestPath} to score a different corpus. The second form is the one that can actually
   * test transfer: scoring a held-out label set against items whose true label is not in it asks an
   * incoherent question and can only fail.
   */
  public static void main(String[] args) throws IOException {
    Path stateHarvest = Path.of(args[0]);
    Path labelHarvest = Path.of(args[1]);
    String trainSet = args[2];
    List<String> evalSets = List.of(args[3].split(","));
    double l2 = args.length > 4 ? Double.parseDouble(args[4]) : 1.0;

    Map<String, List<float[]>> labelStates = new LinkedHashMap<>();
    Map<String, List<String>> labelNames = new LinkedHashMap<>();
    try (BufferedReader reader = Files.newBufferedReader(labelHarvest, StandardCharsets.UTF_8)) {
      String line;
      while ((line = reader.readLine()) != null) {
        if (line.isBlank()) {
          continue;
        }
        String id = between(line, "\"id\":\"", "\"");
        String[] parts = id.split("::");
        labelStates.computeIfAbsent(parts[0], key -> new ArrayList<>()).add(stateOf(line));
        labelNames.computeIfAbsent(parts[0], key -> new ArrayList<>()).add(parts[2]);
      }
    }
    System.out.println("  label sets encoded:");
    labelStates.forEach(
        (set, states) ->
            System.out.printf("    %-12s %d labels %s%n", set, states.size(), labelNames.get(set)));

    List<TypedRecord> items = new ArrayList<>();
    try (BufferedReader reader = Files.newBufferedReader(stateHarvest, StandardCharsets.UTF_8)) {
      String line;
      while ((line = reader.readLine()) != null) {
        if (line.isBlank()) {
          continue;
        }
        items.add(
            new TypedRecord(
                between(line, "\"id\":\"", "\""),
                between(line, "\"split\":\"", "\""),
                Integer.parseInt(between(line, "\"outcome\":", ",").trim()),
                line.contains("\"truncated\":true"),
                stateOf(line)));
      }
    }

    List<float[]> trainLabels = labelStates.get(trainSet);
    if (trainLabels == null) {
      throw new IllegalArgumentException("no label set named " + trainSet);
    }

    // One row per (item, candidate) pair: positive for the true label, negative for every other.
    List<float[]> features = new ArrayList<>();
    List<Integer> targets = new ArrayList<>();
    List<TypedRecord> sealedItems = new ArrayList<>();
    for (TypedRecord item : items) {
      if (item.split().equals("sealed")) {
        sealedItems.add(item);
        continue;
      }
      for (int candidate = 0; candidate < trainLabels.size(); candidate++) {
        features.add(interaction(item.state(), trainLabels.get(candidate)));
        targets.add(candidate == item.outcome() ? 1 : 0);
      }
    }

    float[][] x = features.toArray(new float[0][]);
    int[] y = new int[targets.size()];
    for (int index = 0; index < y.length; index++) {
      y[index] = targets.get(index);
    }
    FeatureStandardizer standardizer = FeatureStandardizer.fit(x);
    Noul fits = new Noul("this candidate fits this state");
    LinearDecisionHead head =
        new LogisticHeadTrainer(400, 0.1, l2).fit(fits, standardizer.applyAll(x), y);
    System.out.printf("%n  trained on %s: %d pairs of width %d%n", trainSet, x.length, x[0].length);

    CandidateScorer scorer =
        new CandidateScorer(
            (state, candidate) -> {
              double[] logits = head.logits(standardizer.apply(interaction(state, candidate)));
              return logits[1] - logits[0];
            },
            1.0);

    for (String spec : evalSets) {
      String evalSet = spec.contains("@") ? spec.substring(0, spec.indexOf('@')) : spec;
      List<TypedRecord> evalItems = sealedItems;
      if (spec.contains("@")) {
        evalItems = new ArrayList<>();
        Path other = Path.of(spec.substring(spec.indexOf('@') + 1));
        try (BufferedReader reader = Files.newBufferedReader(other, StandardCharsets.UTF_8)) {
          String row;
          while ((row = reader.readLine()) != null) {
            if (row.isBlank() || !row.contains("\"split\":\"sealed\"")) {
              continue;
            }
            evalItems.add(
                new TypedRecord(
                    between(row, "\"id\":\"", "\""),
                    "sealed",
                    Integer.parseInt(between(row, "\"outcome\":", ",").trim()),
                    row.contains("\"truncated\":true"),
                    stateOf(row)));
          }
        }
      }
      List<float[]> candidates = labelStates.get(evalSet);
      if (candidates == null) {
        System.out.printf("  %-12s  no such label set%n", evalSet);
        continue;
      }
      AnswerSpace space = new Choice("which label", labelNames.get(evalSet));
      int hits = 0;
      int[] perOutcome = new int[candidates.size()];
      for (TypedRecord item : evalItems) {
        if (item.outcome() >= candidates.size()) {
          continue; // this item's true label has no counterpart in the held-out set
        }
        Verdict verdict = scorer.score(space, item.state(), candidates);
        double[] probabilities = verdict.probabilities();
        int predicted = 0;
        for (int index = 1; index < probabilities.length; index++) {
          if (probabilities[index] > probabilities[predicted]) {
            predicted = index;
          }
        }
        hits += predicted == item.outcome() ? 1 : 0;
        perOutcome[item.outcome()]++;
      }
      int scored = 0;
      int majority = 0;
      for (int count : perOutcome) {
        scored += count;
        majority = Math.max(majority, count);
      }
      String note = evalSet.equals(trainSet) ? "trained" : "HELD OUT";
      String scoredOn = spec.contains("@") ? "own items" : "agnews items";
      System.out.printf(
          "  %-12s %-9s %-12s n=%-4d floor %.4f  accuracy %.4f%n",
          evalSet, note, scoredOn, scored, (double) majority / scored, (double) hits / scored);
    }
  }

  /**
   * Interaction features: elementwise product and absolute difference.
   *
   * <p>Neither names a label, which is the whole point — a head reading only these can score a
   * candidate it has never been trained on.
   */
  private static float[] interaction(float[] state, float[] candidate) {
    int width = Math.min(state.length, candidate.length);
    float[] features = new float[width * 2];
    for (int index = 0; index < width; index++) {
      features[index] = state[index] * candidate[index];
      features[width + index] = Math.abs(state[index] - candidate[index]);
    }
    return features;
  }

  private static float[] stateOf(String line) {
    String array = between(line, "\"state\":[", "]");
    String[] parts = array.split(",");
    float[] state = new float[parts.length];
    for (int index = 0; index < parts.length; index++) {
      state[index] = Float.intBitsToFloat(Integer.parseInt(parts[index].trim()));
    }
    return state;
  }

  private static String between(String line, String open, String close) {
    int a = line.indexOf(open);
    if (a < 0) {
      throw new IllegalArgumentException("missing " + open);
    }
    a += open.length();
    int b = line.indexOf(close, a);
    return line.substring(a, b);
  }
}
