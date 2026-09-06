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

import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.Arrays;
import java.util.HexFormat;
import java.util.List;
import java.util.Locale;

/** Focused cold-load and warm-scoring experiment for the pinned mxbai DeBERTa reranker. */
public final class MxbaiDebertaRerankerPerformanceExperiment {

  private static final String QUERY = "What is the population of Berlin?";
  private static final List<String> DOCUMENTS =
      List.of(
          "Berlin has a population of about 3.7 million people.",
          "The Eiffel Tower is located in Paris.",
          "Berlin is well known for its museums and its metropolitan area of about six million people.",
          "Domestic cats sleep for a large part of the day.",
          "New York City had an estimated population of 8,804,190 in 2020.",
          "The Berlin Wall divided the city from 1961 until 1989.");

  private MxbaiDebertaRerankerPerformanceExperiment() {}

  public static void main(String[] args) throws Exception {
    Path directory =
        Path.of(requiredProperty("models.fixtures.mxbaiRerankerDirectory"))
            .toAbsolutePath()
            .normalize();
    Path weights = directory.resolve("model.safetensors");
    int warmups = argument(args, 0, 2);
    int pairIterations = argument(args, 1, 10);
    int batchIterations = argument(args, 2, 3);

    long heapBefore = usedHeapBytes();
    long loadStart = System.nanoTime();
    try (SafetensorsRerankingModel model = SafetensorsRerankingModel.load(directory)) {
      long loadNanos = System.nanoTime() - loadStart;
      long heapAfterLoad = usedHeapBytes();
      for (int warmup = 0; warmup < warmups; warmup++) {
        model.scoreAll(QUERY, DOCUMENTS);
      }

      long[] pairNanos = new long[pairIterations];
      for (int iteration = 0; iteration < pairIterations; iteration++) {
        long start = System.nanoTime();
        model.score(QUERY, DOCUMENTS.get(iteration % DOCUMENTS.size()));
        pairNanos[iteration] = System.nanoTime() - start;
      }

      long[] batchNanos = new long[batchIterations];
      List<Double> scores = List.of();
      for (int iteration = 0; iteration < batchIterations; iteration++) {
        long start = System.nanoTime();
        scores = model.scoreAll(QUERY, DOCUMENTS);
        batchNanos[iteration] = System.nanoTime() - start;
      }
      List<Integer> ranking =
          model.rerank(QUERY, DOCUMENTS, DOCUMENTS.size()).stream()
              .map(result -> result.originalIndex())
              .toList();

      System.out.printf(
          Locale.ROOT,
          """
          {
            "model": "mixedbread-ai/mxbai-rerank-xsmall-v1",
            "revision": "b5c6e9da73abc3711f593f705371cdbe9e0fe422",
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
            "heapIncreaseAfterLoadBytes": %d,
            "pairP50Millis": %.3f,
            "pairP95Millis": %.3f,
            "batchP50Millis": %.3f,
            "batchP95Millis": %.3f,
            "batchP50DocumentsPerSecond": %.3f,
            "scores": %s,
            "ranking": %s
          }
          """,
          sha256(weights),
          Files.size(weights),
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
          millis(loadNanos),
          heapAfterLoad - heapBefore,
          millis(percentile(pairNanos, 0.50)),
          millis(percentile(pairNanos, 0.95)),
          millis(percentile(batchNanos, 0.50)),
          millis(percentile(batchNanos, 0.95)),
          DOCUMENTS.size() * 1_000_000_000.0 / percentile(batchNanos, 0.50),
          scores,
          ranking);
    }
  }

  private static String requiredProperty(String name) {
    String value = System.getProperty(name, "");
    if (value.isBlank()) {
      throw new IllegalArgumentException("set -D" + name + "=/path/to/model-directory");
    }
    return value;
  }

  private static int argument(String[] args, int index, int fallback) {
    int value = args.length > index ? Integer.parseInt(args[index]) : fallback;
    if (value < 1) {
      throw new IllegalArgumentException("iteration counts must be positive");
    }
    return value;
  }

  private static long usedHeapBytes() {
    System.gc();
    return Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
  }

  private static long percentile(long[] values, double quantile) {
    long[] sorted = values.clone();
    Arrays.sort(sorted);
    return sorted[Math.max(0, (int) Math.ceil(quantile * sorted.length) - 1)];
  }

  private static double millis(long nanos) {
    return nanos / 1_000_000.0;
  }

  private static String sha256(Path path) throws Exception {
    MessageDigest digest = MessageDigest.getInstance("SHA-256");
    try (InputStream input = Files.newInputStream(path)) {
      byte[] buffer = new byte[64 * 1024];
      for (int read; (read = input.read(buffer)) != -1; ) {
        digest.update(buffer, 0, read);
      }
    }
    return HexFormat.of().formatHex(digest.digest());
  }
}
