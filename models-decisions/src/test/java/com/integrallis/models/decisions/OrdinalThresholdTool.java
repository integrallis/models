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
import java.util.List;

/**
 * Fits a Score as an ordered stack of thresholds rather than as unordered classes.
 *
 * <p>A multinomial head treats levels 0 and 4 as no more distant than 0 and 1, discarding the one
 * property that makes a Score a Score. This fits K-1 binary heads instead -- "is the level above
 * k?" -- so every training row informs every threshold below and above it, and the ordering is used
 * rather than thrown away. Level probabilities come from consecutive differences.
 *
 * <p>Measured against the multinomial baseline on the same harvest, because the question is whether
 * the ordering carries information a frozen state can express, not whether a different loss
 * produces a different number.
 */
public final class OrdinalThresholdTool {

  private OrdinalThresholdTool() {}

  /** Arguments: harvest JSONL, level count, L2. */
  public static void main(String[] args) throws IOException {
    Path harvest = Path.of(args[0]);
    int levels = Integer.parseInt(args[1]);
    double l2 = args.length > 2 ? Double.parseDouble(args[2]) : 1.0;

    List<TypedRecord> train = new ArrayList<>();
    List<TypedRecord> sealed = new ArrayList<>();
    try (BufferedReader reader = Files.newBufferedReader(harvest, StandardCharsets.UTF_8)) {
      String line;
      while ((line = reader.readLine()) != null) {
        if (line.isBlank()) {
          continue;
        }
        TypedRecord record =
            new TypedRecord(
                between(line, "\"id\":\"", "\""),
                between(line, "\"split\":\"", "\""),
                Integer.parseInt(between(line, "\"outcome\":", ",").trim()),
                line.contains("\"truncated\":true"),
                stateOf(line));
        if (record.split().equals("sealed")) {
          sealed.add(record);
        } else {
          train.add(record);
        }
      }
    }

    float[][] x = new float[train.size()][];
    for (int index = 0; index < train.size(); index++) {
      x[index] = train.get(index).state();
    }
    FeatureStandardizer standardizer = FeatureStandardizer.fit(x);
    float[][] standardized = standardizer.applyAll(x);

    // One head per cut point: head k answers "is this level greater than k?".
    Noul above = new Noul("the level is above this threshold");
    List<LinearDecisionHead> thresholds = new ArrayList<>();
    for (int cut = 0; cut < levels - 1; cut++) {
      int[] y = new int[train.size()];
      for (int index = 0; index < train.size(); index++) {
        y[index] = train.get(index).outcome() > cut ? 1 : 0;
      }
      thresholds.add(new LogisticHeadTrainer(400, 0.1, l2).fit(above, standardized, y));
    }

    int hits = 0;
    double ordinalTotal = 0.0;
    int[] perOutcome = new int[levels];
    for (TypedRecord record : sealed) {
      float[] state = standardizer.apply(record.state());

      // P(level > k) for each cut, forced non-increasing so the differences cannot go negative.
      double[] exceed = new double[levels - 1];
      for (int cut = 0; cut < levels - 1; cut++) {
        double[] probabilities = Calibration.softmax(thresholds.get(cut).logits(state), 1.0);
        exceed[cut] = probabilities[1];
        if (cut > 0) {
          exceed[cut] = Math.min(exceed[cut], exceed[cut - 1]);
        }
      }

      double[] level = new double[levels];
      level[0] = 1.0 - exceed[0];
      for (int index = 1; index < levels - 1; index++) {
        level[index] = exceed[index - 1] - exceed[index];
      }
      level[levels - 1] = exceed[levels - 2];

      int predicted = 0;
      for (int index = 1; index < levels; index++) {
        if (level[index] > level[predicted]) {
          predicted = index;
        }
      }
      hits += predicted == record.outcome() ? 1 : 0;
      double expected = 0.0;
      for (int index = 0; index < levels; index++) {
        expected += index * level[index];
      }
      ordinalTotal += Math.abs(expected - record.outcome());
      perOutcome[record.outcome()]++;
    }

    int majority = 0;
    for (int count : perOutcome) {
      majority = Math.max(majority, count);
    }
    double floor = (double) majority / sealed.size();
    double accuracy = (double) hits / sealed.size();
    System.out.printf("  levels %d, thresholds %d, l2 %.1f%n", levels, levels - 1, l2);
    System.out.printf("  sealed items          %d%n", sealed.size());
    System.out.printf("  majority-class floor  %.4f%n", floor);
    System.out.printf("  accuracy              %.4f%n", accuracy);
    System.out.printf("  beats floor by 0.05   %s%n", accuracy - floor >= 0.05 ? "yes" : "NO");
    System.out.printf("  ordinal MAE (levels)  %.4f%n", ordinalTotal / sealed.size());
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
    return line.substring(a, line.indexOf(close, a));
  }
}
