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

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * The S3 record frozen before any test-set tuned run: tuned weights per member set and rule, the
 * development-split best member, and per-member confidence temperatures.
 *
 * <p>Test-set runs reference it by path and sha256; a digest mismatch refuses the run.
 */
public final class FrozenTuning {
  private static final ObjectMapper JSON = new ObjectMapper();

  private final String path;
  private final String sha256;
  private final Map<String, double[]> weights;
  private final String bestMember;
  private final Map<String, Double> temperatures;
  private final Set<String> devIds;
  private final String devIdsSha256;

  private FrozenTuning(
      String path,
      String sha256,
      Map<String, double[]> weights,
      String bestMember,
      Map<String, Double> temperatures,
      Set<String> devIds,
      String devIdsSha256) {
    this.path = path;
    this.sha256 = sha256;
    this.weights = weights;
    this.bestMember = bestMember;
    this.temperatures = temperatures;
    this.devIds = devIds;
    this.devIdsSha256 = devIdsSha256;
  }

  /** Loads the record after checking its sha256 against the value given on the command line. */
  public static FrozenTuning load(Path file, String expectedSha256) throws IOException {
    if (expectedSha256 == null || !expectedSha256.matches("[0-9a-f]{64}")) {
      throw new IllegalArgumentException("--frozen-sha256 must be a 64-character hex sha256");
    }
    byte[] bytes = Files.readAllBytes(file);
    String actual = Digests.sha256(bytes);
    if (!actual.equals(expectedSha256)) {
      throw new IllegalArgumentException(
          "frozen record digest mismatch: expected "
              + expectedSha256
              + " but "
              + file
              + " is "
              + actual);
    }
    JsonNode root = JSON.readTree(bytes);
    if (!"frozen".equals(root.path("kind").asText())) {
      throw new IllegalArgumentException(file + " is not a frozen tuning record");
    }
    Map<String, double[]> weights = new LinkedHashMap<>();
    for (JsonNode entry : root.path("weights")) {
      List<String> members = new ArrayList<>();
      entry.path("members").forEach(m -> members.add(m.asText()));
      double[] values = new double[entry.path("weights").size()];
      for (int i = 0; i < values.length; i++) {
        values[i] = entry.path("weights").get(i).asDouble();
      }
      FusionMath.validateWeights(values);
      weights.put(key(members, FusionRule.parse(entry.path("rule").asText())), values);
    }
    Map<String, Double> temperatures = new LinkedHashMap<>();
    root.path("calibration")
        .path("members")
        .properties()
        .forEach(e -> temperatures.put(e.getKey(), e.getValue().path("temperature").asDouble()));
    Set<String> devIds = new HashSet<>();
    root.path("devSplit").path("ids").forEach(id -> devIds.add(id.asText()));
    return new FrozenTuning(
        file.toAbsolutePath().toString(),
        actual,
        weights,
        root.path("bestMember").path("name").asText(null),
        temperatures,
        devIds,
        root.path("devSplit").path("idsSha256").asText(null));
  }

  static String key(List<String> members, FusionRule rule) {
    return String.join("+", members) + ":" + rule.id();
  }

  double[] weightsFor(List<String> members, FusionRule rule) {
    double[] values = weights.get(key(members, rule));
    if (values == null) {
      throw new IllegalStateException(
          "frozen record has no tuned weights for "
              + key(members, rule)
              + "; present: "
              + weights.keySet());
    }
    return values.clone();
  }

  String bestMember() {
    if (bestMember == null || bestMember.isBlank()) {
      throw new IllegalStateException("frozen record names no best member");
    }
    return bestMember;
  }

  Map<String, Double> temperatures() {
    if (temperatures.isEmpty()) {
      throw new IllegalStateException("frozen record has no confidence calibration");
    }
    return Map.copyOf(temperatures);
  }

  /** Refuses to score items that belong to the development split the record was tuned on. */
  void requireDisjoint(List<DatasetItem> items) {
    for (DatasetItem item : items) {
      if (devIds.contains(item.id())) {
        throw new IllegalArgumentException(
            "item " + item.id() + " belongs to the development split used for tuning");
      }
    }
  }

  String path() {
    return path;
  }

  String sha256() {
    return sha256;
  }

  String devIdsSha256() {
    return devIdsSha256;
  }
}
