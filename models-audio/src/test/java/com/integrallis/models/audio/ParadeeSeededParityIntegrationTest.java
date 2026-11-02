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

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.integrallis.models.backend.onnx.NoiseSource;
import com.integrallis.models.backend.onnx.OnnxGraphExecutor;
import com.integrallis.models.backend.onnx.OnnxModel;
import com.integrallis.models.backend.onnx.OnnxNode;
import com.integrallis.models.backend.onnx.Tensor;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Seeded parity against the publisher's runtime: the oracle run exposed the noise tensor each graph
 * drew, the executor consumes that tensor, and the waveforms are compared with the shipped speech
 * policy's thresholds (PCM cosine at least 0.995, signal-to-difference at least 20 dB).
 *
 * <p>Needs {@code MODELS_PARADEE_ONNX_DIR} and {@code MODELS_PARADEE_ORACLE_DIR} (the output of
 * {@code comparator/oracle_dump.py}). When a case falls short, the fp32 case that also carries
 * every intermediate is walked node by node and the first divergences are printed.
 */
@Tag("integration")
class ParadeeSeededParityIntegrationTest {

  /**
   * The kernels the parity run measures: the bundled Rust kernels, the product's default route,
   * unless MODELS_PARADEE_PARITY_KERNELS=pure-java asks for the Java kernels.
   */
  private static final com.integrallis.models.backend.onnx.OnnxKernels PARITY_KERNELS =
      parityKernels();

  private static com.integrallis.models.backend.onnx.OnnxKernels parityKernels() {
    if ("pure-java".equals(System.getenv("MODELS_PARADEE_PARITY_KERNELS"))) {
      return null;
    }
    String pinned = System.getProperty("models.native.kernels.library");
    if (pinned != null) {
      return com.integrallis.models.backend.nativekernel.NativeOnnxKernels.load(
          java.nio.file.Path.of(pinned));
    }
    try {
      return com.integrallis.models.backend.nativekernel.NativeOnnxKernels.load();
    } catch (java.io.IOException failure) {
      throw new IllegalStateException("bundled Rust ONNX kernels are not loadable", failure);
    }
  }

  private static final double MINIMUM_COSINE = 0.995;
  private static final double MINIMUM_SDR_DB = 20.0;

  @Test
  void matchesTheOracleWaveformsWhenFedTheOracleNoise() throws Exception {
    String onnxDir = System.getenv("MODELS_PARADEE_ONNX_DIR");
    String oracleDir = System.getenv("MODELS_PARADEE_ORACLE_DIR");
    assumeTrue(
        onnxDir != null && oracleDir != null,
        "MODELS_PARADEE_ONNX_DIR / MODELS_PARADEE_ORACLE_DIR not set");
    Path manifestPath = Path.of(oracleDir, "manifest.json");
    assumeTrue(Files.isRegularFile(manifestPath), manifestPath + " missing");
    List<Map<String, Object>> cases = cases(Files.readString(manifestPath, StandardCharsets.UTF_8));
    assertThat(cases).isNotEmpty();

    Map<String, OnnxGraphExecutorHolder> executors = new HashMap<>();
    List<String> failures = new ArrayList<>();
    double worstCosine = 1.0;
    double worstSdr = Double.POSITIVE_INFINITY;
    for (Map<String, Object> c : cases) {
      String tag = (String) c.get("tag");
      String graph = (String) c.get("graph");
      String file =
          graph.endsWith("paradee_int8.onnx") ? "onnx_paradee_int8.onnx" : "onnx_paradee.onnx";
      OnnxGraphExecutorHolder holder =
          executors.computeIfAbsent(file, f -> OnnxGraphExecutorHolder.load(Path.of(onnxDir, f)));
      Map<String, Tensor> dump = NpzArchive.read(Path.of(oracleDir, tag + ".npz"));
      Tensor noise = dump.get("noise");
      Map<String, Tensor> feeds = new LinkedHashMap<>();
      feeds.put("input_ids", dump.get("input_ids"));
      feeds.put("speed", dump.get("speed"));
      feeds.put("voice", dump.get("voice"));
      long start = System.nanoTime();
      Tensor ours =
          holder
              .executor
              .run(
                  feeds,
                  List.of("waveform"),
                  null,
                  NoiseSource.injected(Map.of(holder.noiseName, noise)))
              .get("waveform");
      long millis = (System.nanoTime() - start) / 1_000_000L;
      float[] expected = dump.get("waveform").floats();
      float[] actual = ours.floats();
      Metrics m = Metrics.of(expected, actual);
      worstCosine = Math.min(worstCosine, m.cosine);
      worstSdr = Math.min(worstSdr, m.sdrDb);
      boolean pass =
          actual.length == expected.length
              && m.cosine >= MINIMUM_COSINE
              && m.sdrDb >= MINIMUM_SDR_DB;
      System.out.printf(
          "PARADEE_PARITY %-32s samples ours=%d oracle=%d cosine=%.6f sdr=%.2fdB maxAbs=%.5f run=%dms %s%n",
          tag,
          actual.length,
          expected.length,
          m.cosine,
          m.sdrDb,
          m.maxAbs,
          millis,
          pass ? "PASS" : "FAIL");
      if (!pass) {
        failures.add(tag);
        if (c.containsKey("intermediates")) {
          explain(holder, feeds, noise, Path.of(oracleDir, (String) c.get("intermediates")));
        }
      }
    }
    System.out.printf(
        "PARADEE_PARITY worst cosine=%.6f worst sdr=%.2fdB over %d cases%n",
        worstCosine, worstSdr, cases.size());
    assertThat(failures).as("cases below the parity floor").isEmpty();
  }

  /**
   * Walks the fp32 graph against the oracle's per-node outputs and prints the first divergences.
   */
  private static void explain(
      OnnxGraphExecutorHolder holder, Map<String, Tensor> feeds, Tensor noise, Path intermediates)
      throws Exception {
    Map<String, Tensor> oracle = NpzArchive.read(intermediates);
    Map<String, String> ops = new HashMap<>();
    for (OnnxNode node : holder.model.graph().nodes()) {
      for (String output : node.outputs()) {
        ops.put(output, node.opType());
      }
    }
    List<String> report = new ArrayList<>();
    holder.executor.run(
        feeds,
        List.of("waveform"),
        (name, tensor) -> {
          Tensor reference = oracle.get(name);
          if (reference == null || report.size() >= 15) {
            return;
          }
          if (reference.size() != tensor.size()) {
            report.add(
                String.format(
                    "%-40s %-18s SHAPE ours=%s oracle=%s",
                    name,
                    ops.get(name),
                    java.util.Arrays.toString(tensor.shape()),
                    java.util.Arrays.toString(reference.shape())));
            return;
          }
          double maxAbs = 0;
          double maxRel = 0;
          for (int i = 0; i < tensor.size(); i++) {
            double a = tensor.doubleAt(i);
            double b = reference.doubleAt(i);
            double diff = Math.abs(a - b);
            maxAbs = Math.max(maxAbs, diff);
            maxRel = Math.max(maxRel, diff / (Math.abs(b) + 1e-6));
          }
          if (maxAbs > 1e-3 && maxRel > 1e-2) {
            report.add(
                String.format(
                    "%-40s %-18s maxAbs=%.5f maxRel=%.4f shape=%s",
                    name,
                    ops.get(name),
                    maxAbs,
                    maxRel,
                    java.util.Arrays.toString(tensor.shape())));
          }
        },
        NoiseSource.injected(Map.of(holder.noiseName, noise)));
    for (String line : report) {
      System.out.println("PARADEE_PARITY   divergence " + line);
    }
    if (report.isEmpty()) {
      System.out.println(
          "PARADEE_PARITY   no intermediate diverged beyond 1e-3 absolute and 1% relative");
    }
  }

  private static final class OnnxGraphExecutorHolder {
    final OnnxModel model;
    final OnnxGraphExecutor executor;
    final String noiseName;

    private OnnxGraphExecutorHolder(OnnxModel model, OnnxGraphExecutor executor, String noiseName) {
      this.model = model;
      this.executor = executor;
      this.noiseName = noiseName;
    }

    static OnnxGraphExecutorHolder load(Path path) {
      try {
        OnnxModel model = OnnxModel.load(path);
        String noiseName = null;
        for (OnnxNode node : model.graph().nodes()) {
          if (node.opType().equals("RandomNormalLike")) {
            noiseName = node.outputs().get(0);
          }
        }
        return new OnnxGraphExecutorHolder(
            model,
            new OnnxGraphExecutor(
                model, com.integrallis.models.backend.onnx.NoiseSource.seeded(0), PARITY_KERNELS),
            noiseName);
      } catch (java.io.IOException e) {
        throw new java.io.UncheckedIOException(e);
      }
    }
  }

  private record Metrics(double cosine, double sdrDb, double maxAbs) {
    static Metrics of(float[] reference, float[] actual) {
      int n = Math.min(reference.length, actual.length);
      double dot = 0;
      double refEnergy = 0;
      double actEnergy = 0;
      double diffEnergy = 0;
      double maxAbs = 0;
      for (int i = 0; i < n; i++) {
        dot += (double) reference[i] * actual[i];
        refEnergy += (double) reference[i] * reference[i];
        actEnergy += (double) actual[i] * actual[i];
        double d = reference[i] - actual[i];
        diffEnergy += d * d;
        maxAbs = Math.max(maxAbs, Math.abs(d));
      }
      for (int i = n; i < reference.length; i++) {
        refEnergy += (double) reference[i] * reference[i];
        diffEnergy += (double) reference[i] * reference[i];
      }
      double cosine = refEnergy == 0 || actEnergy == 0 ? 0 : dot / Math.sqrt(refEnergy * actEnergy);
      double sdr =
          diffEnergy == 0 ? Double.POSITIVE_INFINITY : 10 * Math.log10(refEnergy / diffEnergy);
      return new Metrics(cosine, sdr, maxAbs);
    }
  }

  /**
   * Minimal manifest reader: the cases array of flat objects, parsed with the audio package's JSON.
   */
  @SuppressWarnings("unchecked")
  private static List<Map<String, Object>> cases(String json) {
    Map<String, Object> root = com.integrallis.models.audio.phonemizer.JsonAccess.object(json);
    return (List<Map<String, Object>>) root.get("cases");
  }
}
