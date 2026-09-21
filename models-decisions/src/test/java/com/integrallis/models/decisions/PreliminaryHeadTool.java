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
 * A preliminary read on a harvested corpus that never touches the sealed split.
 *
 * <p>The train split is divided internally, the head is fitted on the larger part and scored on the
 * smaller, and the temperature is fitted on the fitting part alone. The sealed split is not opened,
 * not counted and not loaded, because it is read once and that reading belongs to the final result.
 *
 * <p>Numbers from here are a development signal. They are optimistic relative to a sealed reading
 * and must never be quoted as the result.
 */
public final class PreliminaryHeadTool {

  private PreliminaryHeadTool() {}

  /** Arguments: harvest JSONL, corpus name. */
  public static void main(String[] args) throws IOException {
    Path path = Path.of(args[0]);
    String corpus = args[1];

    List<HarvestRecord> train = new ArrayList<>();
    int skipped = 0;
    try (BufferedReader reader = Files.newBufferedReader(path, StandardCharsets.UTF_8)) {
      String line;
      while ((line = reader.readLine()) != null) {
        try {
          if (!line.contains("\"split\":\"train\"")) {
            continue; // sealed and calibration are not opened here
          }
          train.add(parse(line));
        } catch (RuntimeException incomplete) {
          skipped++; // partial trailing line during a flush
        }
      }
    }

    int cut = train.size() * 4 / 5;
    List<HarvestRecord> fit = train.subList(0, cut);
    List<HarvestRecord> dev = train.subList(cut, train.size());

    Noul space = new Noul("the state answers the question");
    // Statistics from the fitting rows alone; applying them to dev is not a leak, deriving them
    // from dev would be.
    FeatureStandardizer standardizer = FeatureStandardizer.fit(features(fit));
    float[][] fitX = standardizer.applyAll(features(fit));
    float[][] devX = standardizer.applyAll(features(dev));
    int[] fitY = labels(fit);

    System.out.printf("%n  %-10s %-9s %-9s %-9s %-9s%n", "L2", "devAcc", "agree", "ECE", "temp");
    LinearDecisionHead head = null;
    double temperature = 1.0;
    double bestAcc = -1.0;
    for (double l2 : new double[] {0.0, 0.01, 0.1, 1.0, 10.0, 100.0}) {
      LinearDecisionHead candidate = new LogisticHeadTrainer(400, 0.1, l2).fit(space, fitX, fitY);
      double[][] devLogits = new double[devX.length][];
      for (int i = 0; i < devX.length; i++) {
        devLogits[i] = candidate.logits(devX[i]);
      }
      int[] devY = labels(dev);
      double t = TemperatureFitter.fit(devLogits, devY);
      int hits = 0;
      int agreeN = 0;
      double[] conf = new double[devX.length];
      boolean[] ok = new boolean[devX.length];
      for (int i = 0; i < devX.length; i++) {
        Verdict v = candidate.decide(devX[i], t);
        boolean says = v.probabilityOfTrue() >= 0.5;
        hits += says == dev.get(i).label() ? 1 : 0;
        agreeN += says == dev.get(i).decode() ? 1 : 0;
        conf[i] = Math.max(v.probabilityOfTrue(), 1.0 - v.probabilityOfTrue());
        ok[i] = says == dev.get(i).label();
      }
      double acc = (double) hits / devX.length;
      System.out.printf(
          "  %-10.2f %-9.4f %-9.4f %-9.4f %-9.3f%n",
          l2,
          acc,
          (double) agreeN / devX.length,
          Calibration.expectedCalibrationError(conf, ok, 10),
          t);
      if (acc > bestAcc) {
        bestAcc = acc;
        head = candidate;
        temperature = t;
      }
    }
    System.out.println();

    int n = dev.size();
    double[] probability = new double[n];
    double[] confidence = new double[n];
    boolean[] correct = new boolean[n];
    boolean[] outcome = new boolean[n];
    int headHits = 0;
    int decodeHits = 0;
    int agree = 0;
    int positives = 0;
    for (int i = 0; i < n; i++) {
      HarvestRecord r = dev.get(i);
      Verdict v = head.decide(standardizer.apply(r.hidden()), temperature);
      boolean says = v.probabilityOfTrue() >= 0.5;
      probability[i] = v.probabilityOfTrue();
      confidence[i] = Math.max(probability[i], 1.0 - probability[i]);
      outcome[i] = r.label();
      correct[i] = says == r.label();
      headHits += correct[i] ? 1 : 0;
      decodeHits += r.decode() == r.label() ? 1 : 0;
      agree += says == r.decode() ? 1 : 0;
      positives += r.label() ? 1 : 0;
    }
    double share = (double) positives / n;
    double floor = Math.max(share, 1.0 - share);

    System.out.printf("=== %s PRELIMINARY (sealed split untouched) ===%n", corpus);
    System.out.printf(
        "  fit on %d, scored on %d held out of train; skipped %d partial lines%n",
        fit.size(), n, skipped);
    System.out.printf("  majority floor        %.4f%n", floor);
    System.out.printf(
        "  head accuracy         %.4f   (margin %+.4f)%n",
        (double) headHits / n, (double) headHits / n - floor);
    System.out.printf(
        "  decode accuracy       %.4f   (margin %+.4f)%n",
        (double) decodeHits / n, (double) decodeHits / n - floor);
    System.out.printf("  head/decode agreement %.4f%n", (double) agree / n);
    System.out.printf(
        "  ECE (15 bins)         %.4f%n",
        Calibration.expectedCalibrationError(confidence, correct, 15));
    System.out.printf(
        "  Brier                 %.4f%n", Calibration.brierScore(probability, outcome));
    System.out.printf("  temperature           %.4f%n", temperature);
  }

  private static HarvestRecord parse(String line) {
    String id = between(line, "\"id\":\"", "\"");
    boolean label = line.contains("\"label\":true");
    boolean decode = line.contains("\"decode\":true");
    boolean truncated = line.contains("\"truncated\":true");
    String arr = between(line, "\"hidden\":[", "]");
    String[] parts = arr.split(",");
    float[] hidden = new float[parts.length];
    for (int i = 0; i < parts.length; i++) {
      hidden[i] = Float.intBitsToFloat(Integer.parseInt(parts[i].trim()));
    }
    return new HarvestRecord(id, "train", label, decode, truncated, 0L, hidden);
  }

  private static String between(String s, String open, String close) {
    int a = s.indexOf(open);
    if (a < 0) {
      throw new IllegalArgumentException("incomplete line");
    }
    a += open.length();
    int b = s.indexOf(close, a);
    if (b < 0) {
      throw new IllegalArgumentException("incomplete line");
    }
    return s.substring(a, b);
  }

  private static float[][] features(List<HarvestRecord> rs) {
    float[][] f = new float[rs.size()][];
    for (int i = 0; i < rs.size(); i++) {
      f[i] = rs.get(i).hidden();
    }
    return f;
  }

  private static int[] labels(List<HarvestRecord> rs) {
    int[] l = new int[rs.size()];
    for (int i = 0; i < rs.size(); i++) {
      l[i] = rs.get(i).label() ? 1 : 0;
    }
    return l;
  }

  private static double[][] logitsOf(LinearDecisionHead head, List<HarvestRecord> rs) {
    double[][] g = new double[rs.size()][];
    for (int i = 0; i < rs.size(); i++) {
      g[i] = head.logits(rs.get(i).hidden());
    }
    return g;
  }
}
