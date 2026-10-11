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
package com.integrallis.models.backend.purejava.qwen35;

import com.integrallis.models.backend.purejava.gguf.GgufFile;
import com.integrallis.models.backend.purejava.gguf.GgufMetadata;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/** The signed, blockwise orthogonal transform declared by Prism's version-1 GGUF metadata. */
final class PrismHadamard {
  private static final String PREFIX = "prism.hadamard.";
  private final Map<String, Transform> projections;
  private final Transform embeddingInverse;

  private PrismHadamard(Map<String, Transform> projections, Transform embeddingInverse) {
    this.projections = Map.copyOf(projections);
    this.embeddingInverse = embeddingInverse;
  }

  static PrismHadamard load(GgufFile file, Qwen35Config config) {
    GgufMetadata metadata = file.metadata();
    if (metadata.get(PREFIX + "version").isEmpty()) return new PrismHadamard(Map.of(), null);
    if (metadata.getUint32(PREFIX + "version").orElse(0) != 1
        || metadata.getBool(PREFIX + "tied_output").orElse(false)) {
      throw new IllegalArgumentException(
          "Supported Prism Hadamard contract is version 1 with an untied output");
    }
    requireString(metadata, "transform", "normalized-sylvester-walsh-hadamard");
    requireString(metadata, "axis", "input-last-dimension");
    int block = metadata.getUint32(PREFIX + "block_size").orElse(0);
    if (block <= 0 || (block & (block - 1)) != 0)
      throw new IllegalArgumentException("Hadamard block must be a positive power of two");
    String signMode = metadata.getString(PREFIX + "sign_mode").orElse("");
    Map<Integer, int[]> signsByWidth = new HashMap<>();
    if (signMode.equals("explicit")) {
      List<Integer> widths = metadata.getInt32Array(PREFIX + "sign_widths").orElseThrow();
      List<Integer> signs = metadata.getInt32Array(PREFIX + "sign_values").orElseThrow();
      if (widths.isEmpty())
        throw new IllegalArgumentException("Explicit Hadamard signs must not be empty");
      int offset = 0;
      for (int width : widths) {
        if (width <= 0 || width % block != 0 || width > signs.size() - offset) {
          throw new IllegalArgumentException("Invalid Hadamard sign width " + width);
        }
        int[] values = new int[width];
        for (int i = 0; i < width; i++) {
          int sign = signs.get(offset++);
          if (sign != -1 && sign != 1)
            throw new IllegalArgumentException("Hadamard signs must be +/-1");
          values[i] = sign;
        }
        if (signsByWidth.put(width, values) != null)
          throw new IllegalArgumentException("Duplicate Hadamard sign width");
      }
      if (offset != signs.size())
        throw new IllegalArgumentException("Hadamard sign array length mismatch");
    } else if (!signMode.equals("identity")) {
      throw new IllegalArgumentException("Unknown Hadamard sign mode " + signMode);
    }
    boolean groupedValues = metadata.getBool(PREFIX + "gdn_v_grouped").orElse(false);
    List<String> names = metadata.getStringArray(PREFIX + "weight_names").orElseThrow();
    if (names.isEmpty())
      throw new IllegalArgumentException("Hadamard weight names must not be empty");
    Map<String, Transform> projections = new HashMap<>();
    for (String name : names) {
      var tensor = file.getTensor(name);
      if (tensor.shape().length != 2)
        throw new IllegalArgumentException("Hadamard requires a matrix: " + name);
      int width = Math.toIntExact(tensor.shape()[0]);
      int[] signs = signsByWidth.get(width);
      if (signMode.equals("explicit") && signs == null)
        throw new IllegalArgumentException("Missing signs for " + name);
      boolean permute = groupedValues && name.endsWith(".ssm_out.weight");
      var transform =
          new Transform(
              block,
              width,
              signs,
              permute ? config.gdnKeyHeads() : 1,
              permute ? config.gdnValueHeads() : 1);
      if (projections.put(name, transform) != null)
        throw new IllegalArgumentException("Duplicate Hadamard weight " + name);
    }
    Transform inverse = null;
    for (String name : metadata.getStringArray(PREFIX + "inverse_weight_names").orElse(List.of())) {
      if (!name.equals("token_embd.weight") || inverse != null || projections.containsKey(name)) {
        throw new IllegalArgumentException("Unsupported or duplicate Hadamard inverse " + name);
      }
      int width = config.embeddingDim();
      int[] signs = signsByWidth.get(width);
      if (signMode.equals("explicit") && signs == null)
        throw new IllegalArgumentException("Missing embedding signs");
      inverse = new Transform(block, width, signs, 1, 1);
      if (!file.hasTensor("output.weight"))
        throw new IllegalArgumentException(
            "Version-1 latent embeddings require an untied output head");
    }
    return new PrismHadamard(projections, inverse);
  }

  void requireRouted(Set<String> routedNames) {
    Set<String> missing = new HashSet<>(projections.keySet());
    missing.removeAll(routedNames);
    if (!missing.isEmpty())
      throw new IllegalArgumentException(
          "Hadamard weights lack a transformed projection path: " + missing);
  }

  Transform projection(String name) {
    return projections.get(name);
  }

  void restoreEmbedding(float[] row) {
    if (embeddingInverse != null) embeddingInverse.inverse(row);
  }

  private static void requireString(GgufMetadata metadata, String key, String expected) {
    if (!metadata.getString(PREFIX + key).orElse("").equals(expected)) {
      throw new IllegalArgumentException("Expected " + PREFIX + key + "=" + expected);
    }
  }

  static final class Transform {
    private final int block;
    private final int width;
    private final int[] signs;
    private final int keyHeads;
    private final int valueHeads;

    Transform(int block, int width, int[] signs, int keyHeads, int valueHeads) {
      if (block <= 0
          || (block & (block - 1)) != 0
          || width <= 0
          || width % block != 0
          || keyHeads <= 0
          || valueHeads <= 0
          || valueHeads % keyHeads != 0
          || width % valueHeads != 0) {
        throw new IllegalArgumentException("Invalid Hadamard block or head geometry");
      }
      if (signs != null) {
        if (signs.length != width)
          throw new IllegalArgumentException("Hadamard sign width mismatch");
        for (int sign : signs)
          if (sign != 1 && sign != -1)
            throw new IllegalArgumentException("Hadamard signs must be +/-1");
      }
      this.block = block;
      this.width = width;
      this.signs = signs == null ? null : signs.clone();
      this.keyHeads = keyHeads;
      this.valueHeads = valueHeads;
    }

    void forward(float[] input, float[] output) {
      int headDim = width / valueHeads;
      int repeats = valueHeads / keyHeads;
      for (int i = 0; i < width; i++) {
        // The GGUF recurrence emits tiled value heads. Prism folds ssm_out in grouped order.
        int head = i / headDim;
        int sourceHead = (head % repeats) * keyHeads + head / repeats;
        int source = sourceHead * headDim + i % headDim;
        output[i] = input[source] * (signs == null ? 1 : signs[i]);
      }
      walsh(output);
    }

    void inverse(float[] row) {
      walsh(row);
      if (signs != null) for (int i = 0; i < width; i++) row[i] *= signs[i];
    }

    private void walsh(float[] values) {
      for (int start = 0; start < width; start += block) {
        for (int stride = 1; stride < block; stride *= 2) {
          for (int group = start; group < start + block; group += 2 * stride) {
            for (int i = 0; i < stride; i++) {
              float left = values[group + i];
              float right = values[group + stride + i];
              values[group + i] = left + right;
              values[group + stride + i] = left - right;
            }
          }
        }
      }
      float scale = (float) (1.0 / Math.sqrt(block));
      for (int i = 0; i < width; i++) values[i] *= scale;
    }
  }
}
