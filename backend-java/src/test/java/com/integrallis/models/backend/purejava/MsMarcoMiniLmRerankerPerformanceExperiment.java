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
package com.integrallis.models.backend.purejava;

import com.integrallis.models.backend.purejava.fixture.ModelFixtureDescriptor;
import com.integrallis.models.backend.purejava.fixture.ModelFixtureRegistry;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;

/** Focused cold-load and warm-scoring experiment for the pinned MiniLM cross-encoder. */
public final class MsMarcoMiniLmRerankerPerformanceExperiment {
  private static final String FIXTURE_ID = "ms_marco_minilm_l6_v2_q4_k_imatrix_g7c_f7";
  private static final String QUERY = "How many people live in Berlin?";
  private static final List<String> DOCUMENTS =
      List.of(
          "Berlin has a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.",
          "Paris is the capital and most populous city of France.",
          "Berlin is well known for its museums and its metropolitan area of about six million people.",
          "Domestic cats sleep for a large part of the day.",
          "New York City had an estimated population of 8,804,190 in 2020.",
          "The Berlin Wall divided the city from 1961 until 1989.");

  private MsMarcoMiniLmRerankerPerformanceExperiment() {}

  public static void main(String[] args) {
    int warmups = nonNegativeArgument(args, 0, 3, "warmups");
    int pairIterations = positiveArgument(args, 1, 30, "pair iterations");
    int batchIterations = positiveArgument(args, 2, 10, "batch iterations");
    ModelFixtureDescriptor fixture =
        ModelFixtureRegistry.fromClasspath().descriptors().stream()
            .filter(candidate -> FIXTURE_ID.equals(candidate.id()))
            .findFirst()
            .orElseThrow();

    long loadStart = System.nanoTime();
    try (GgufRerankingModel model = GgufRerankingModel.load(fixture.localPath().orElseThrow())) {
      long loadNanos = System.nanoTime() - loadStart;
      for (int warmup = 0; warmup < warmups; warmup++) {
        model.scoreAll(QUERY, DOCUMENTS);
      }

      long[] pairNanos = new long[pairIterations];
      double scoreChecksum = 0.0;
      for (int iteration = 0; iteration < pairIterations; iteration++) {
        long start = System.nanoTime();
        double score = model.score(QUERY, DOCUMENTS.get(iteration % DOCUMENTS.size()));
        pairNanos[iteration] = System.nanoTime() - start;
        scoreChecksum += score;
      }

      long[] batchNanos = new long[batchIterations];
      List<Double> scores = List.of();
      for (int iteration = 0; iteration < batchIterations; iteration++) {
        long start = System.nanoTime();
        scores = model.scoreAll(QUERY, DOCUMENTS);
        batchNanos[iteration] = System.nanoTime() - start;
      }
      List<Integer> topTwo =
          model.rerank(QUERY, DOCUMENTS, 2).stream().map(result -> result.originalIndex()).toList();

      long pairP50 = percentile(pairNanos, 0.50);
      long pairP95 = percentile(pairNanos, 0.95);
      long batchP50 = percentile(batchNanos, 0.50);
      long batchP95 = percentile(batchNanos, 0.95);
      System.out.printf(
          Locale.ROOT,
          """
          {
            "model": "%s",
            "artifactSha256": "%s",
            "artifactSizeBytes": %d,
            "javaVersion": "%s",
            "javaVendor": "%s",
            "osName": "%s",
            "osVersion": "%s",
            "architecture": "%s",
            "processors": %d,
            "warmups": %d,
            "pairIterations": %d,
            "batchIterations": %d,
            "documentsPerBatch": %d,
            "coldLoadMillis": %.3f,
            "pairP50Millis": %.3f,
            "pairP95Millis": %.3f,
            "batchP50Millis": %.3f,
            "batchP95Millis": %.3f,
            "batchP50DocumentsPerSecond": %.3f,
            "scores": %s,
            "topTwoOriginalIndexes": %s,
            "scoreChecksum": %.9f
          }
          """,
          fixture.displayName(),
          fixture.sha256().orElseThrow(),
          fixture.sizeBytes().orElseThrow(),
          System.getProperty("java.version"),
          System.getProperty("java.vendor"),
          System.getProperty("os.name"),
          System.getProperty("os.version"),
          System.getProperty("os.arch"),
          Runtime.getRuntime().availableProcessors(),
          warmups,
          pairIterations,
          batchIterations,
          DOCUMENTS.size(),
          nanosToMillis(loadNanos),
          nanosToMillis(pairP50),
          nanosToMillis(pairP95),
          nanosToMillis(batchP50),
          nanosToMillis(batchP95),
          DOCUMENTS.size() * 1_000_000_000.0 / batchP50,
          scores,
          topTwo,
          scoreChecksum);
    }
  }

  private static long percentile(long[] measurements, double percentile) {
    long[] sorted = measurements.clone();
    Arrays.sort(sorted);
    int index = Math.max(0, (int) Math.ceil(percentile * sorted.length) - 1);
    return sorted[index];
  }

  private static double nanosToMillis(long nanos) {
    return nanos / 1_000_000.0;
  }

  private static int positiveArgument(String[] args, int index, int fallback, String name) {
    int value = nonNegativeArgument(args, index, fallback, name);
    if (value == 0) {
      throw new IllegalArgumentException(name + " must be greater than zero");
    }
    return value;
  }

  private static int nonNegativeArgument(String[] args, int index, int fallback, String name) {
    int value = args.length > index ? Integer.parseInt(args[index]) : fallback;
    if (value < 0) {
      throw new IllegalArgumentException(name + " must not be negative");
    }
    return value;
  }
}
