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
package com.integrallis.models.audio;

import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.integrallis.models.backend.nativekernel.NativeOnnxKernels;
import com.integrallis.models.backend.onnx.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Per-node profile of the int8 Paradee graph on a fixed id sequence: time, operator, attributes and
 * input shapes, sorted by time. Decides which kernels to optimize; not a measurement of the gate.
 * Needs {@code MODELS_PARADEE_ONNX_DIR}.
 */
@Tag("integration")
class ParadeeKernelProfileIntegrationTest {

  @Test
  void profilesEveryNodeOfTheInt8Graph() throws Exception {
    String configured = System.getenv("MODELS_PARADEE_ONNX_DIR");
    assumeTrue(configured != null, "MODELS_PARADEE_ONNX_DIR not set");
    Path path = Path.of(configured, "onnx_paradee_int8.onnx");
    assumeTrue(Files.isRegularFile(path));
    OnnxModel model = OnnxModel.load(path);
    OnnxGraphExecutor executor =
        new OnnxGraphExecutor(
            model,
            NoiseSource.seeded(42),
            "pure-java".equals(System.getenv("MODELS_PARADEE_PROFILE_KERNELS"))
                ? null
                : NativeOnnxKernels.load(
                    Path.of(System.getProperty("models.native.kernels.library"))));
    // 94 ids: the card's first sentence in misaki phonemes, as the oracle dump recorded it.
    long[] ids = {
      0, 65, 156, 54, 86, 16, 4, 16, 83, 100, 16, 56, 156, 47, 45, 49, 156, 128, 49, 50, 16, 156,
      75, 61, 141, 57, 69, 16, 110, 59, 16, 57, 156, 47, 57, 6, 16, 45, 158, 154, 59, 16, 83, 100,
      16, 52, 156, 118, 48, 50, 16, 68, 156, 102, 61, 16, 57, 156, 158, 52, 58, 59, 16, 46, 156,
      118, 149, 16, 102, 69, 16, 50, 156, 56, 155, 100, 139, 100, 54, 50, 16, 45, 156, 102, 69, 50,
      16, 45, 47, 16, 156, 40, 59, 4, 0
    };
    Map<String, Tensor> feeds = new HashMap<>();
    feeds.put("input_ids", Tensor.ofLongs(new long[] {1, ids.length}, ids));
    feeds.put("speed", Tensor.ofFloats(new long[] {1}, new float[] {1f}));
    feeds.put("voice", Tensor.ofLongs(new long[] {1}, new long[] {0}));
    executor.run(feeds); // warm-up

    Map<String, OnnxNode> producers = new HashMap<>();
    for (OnnxNode node : model.graph().nodes()) {
      for (String output : node.outputs()) {
        producers.put(output, node);
      }
    }
    Map<String, Tensor> values = new HashMap<>(feeds);
    List<Object[]> rows = new ArrayList<>();
    long[] last = {System.nanoTime()};
    long start = System.nanoTime();
    executor.run(
        feeds,
        List.of("waveform"),
        (name, tensor) -> {
          long now = System.nanoTime();
          OnnxNode node = producers.get(name);
          StringBuilder shapes = new StringBuilder();
          for (String input : node.inputs()) {
            Tensor t = values.get(input);
            shapes.append(t == null ? "?" : t.type() + Arrays.toString(t.shape())).append(' ');
          }
          values.put(name, tensor);
          rows.add(
              new Object[] {
                now - last[0],
                node.opType(),
                node.attributes().values().toString(),
                shapes.toString().trim(),
                Arrays.toString(tensor.shape())
              });
          last[0] = now;
        });
    long total = System.nanoTime() - start;
    rows.sort((a, b) -> Long.compare((long) b[0], (long) a[0]));
    System.out.printf(
        "PARADEE_PROFILE total %d ms over %d dynamic nodes%n", total / 1_000_000L, rows.size());
    Map<String, long[]> byOp = new HashMap<>();
    for (Object[] r : rows) {
      byOp.computeIfAbsent((String) r[1], k -> new long[2]);
      byOp.get(r[1])[0] += (long) r[0];
      byOp.get(r[1])[1]++;
    }
    byOp.entrySet().stream()
        .sorted((a, b) -> Long.compare(b.getValue()[0], a.getValue()[0]))
        .limit(10)
        .forEach(
            e ->
                System.out.printf(
                    "PARADEE_PROFILE op %-20s %6d ms %4d nodes%n",
                    e.getKey(), e.getValue()[0] / 1_000_000L, e.getValue()[1]));
    for (int i = 0; i < Math.min(30, rows.size()); i++) {
      Object[] r = rows.get(i);
      System.out.printf(
          "PARADEE_PROFILE %6.1f ms %-18s in=%s out=%s %s%n",
          (long) r[0] / 1e6,
          r[1],
          r[3],
          r[4],
          ((String) r[2]).length() > 90 ? ((String) r[2]).substring(0, 90) : r[2]);
    }
  }
}
