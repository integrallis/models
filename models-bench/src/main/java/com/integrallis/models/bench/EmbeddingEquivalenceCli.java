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
package com.integrallis.models.bench;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.integrallis.models.api.Pooling;
import com.integrallis.models.backend.purejava.GgufEmbeddingBackend;
import com.integrallis.models.backend.purejava.PureJavaBackend;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Gates a release on whether this runtime reproduces an embedding model.
 *
 * <p>Embeds a pinned probe set and compares the vectors against ones a pinned reference build
 * produced from the same model bytes. It is deliberately a gate and not a benchmark: eight probes
 * chosen to exercise the paths a runtime gets wrong — single token, long input, non-Latin script,
 * code, ordinary prose — and no corpus, no retrieval metric, no scoring. Runs in seconds because
 * the reference side is committed rather than recomputed.
 *
 * <p>Passing does not prove the embeddings are good; it proves they are the model's. Two
 * identically broken implementations would agree perfectly, which is why the equivalence claim
 * rests on the reference being an independent implementation.
 *
 * <pre>{@code
 * ./gradlew :models-bench:run --args="embedding-equivalence --model /path/to/model.gguf"
 * }</pre>
 */
public final class EmbeddingEquivalenceCli {

  private static final int PASS = 0;
  private static final int FAIL = 1;
  private static final int USAGE = 2;

  private EmbeddingEquivalenceCli() {}

  /**
   * Runs the gate.
   *
   * @param args {@code --model <path>} and optionally {@code --report <path>}
   * @return zero when the runtime reproduced the reference, non-zero otherwise
   * @throws IOException if the artifact or report path cannot be read or written
   */
  public static int run(String[] args) throws IOException {
    Path artifact = null;
    Path report = null;
    for (int index = 0; index < args.length; index++) {
      try {
        switch (args[index]) {
          case "--model" -> artifact = Path.of(requireValue(args, ++index, "--model"));
          case "--report" -> report = Path.of(requireValue(args, ++index, "--report"));
          default -> {
            return usage("unknown argument: " + args[index]);
          }
        }
      } catch (IllegalArgumentException malformed) {
        // A misspelled flag or a shell expansion that produced nothing is a usage problem, and
        // should read as one rather than as a stack trace.
        return usage(malformed.getMessage());
      }
    }
    if (artifact == null) {
      return usage("--model is required");
    }
    if (!Files.isRegularFile(artifact)) {
      return usage("artifact does not exist: " + artifact);
    }

    EmbeddingEquivalence.Reference reference = EmbeddingEquivalence.loadReference();
    List<String> probes = EmbeddingEquivalence.loadProbes();

    // Both digests must hold or the comparison is against vectors from different inputs. Checking
    // them is the difference between proving equivalence and proving nothing at all.
    String probeSetSha256 = EmbeddingEquivalence.sha256(EmbeddingEquivalence.loadProbesRaw());
    if (!probeSetSha256.equals(reference.probeSetSha256())) {
      return usage(
          "probe set has changed since the reference was generated; regenerate it (see"
              + " embedding-equivalence/README.md)");
    }
    String artifactSha256 = Hashing.sha256(artifact);
    if (!artifactSha256.equals(reference.artifactSha256())) {
      return usage(
          "artifact does not match the one the reference was generated from"
              + System.lineSeparator()
              + "  expected "
              + reference.artifactSha256()
              + System.lineSeparator()
              + "  actual   "
              + artifactSha256);
    }

    List<float[]> measured = new ArrayList<>(probes.size());
    List<Double> millis = new ArrayList<>(probes.size());
    long start = System.nanoTime();
    try (PureJavaBackend backend = PureJavaBackend.load(artifact);
        GgufEmbeddingBackend embedding =
            GgufEmbeddingBackend.builder(backend)
                .pooling(pooling(reference.pooling()))
                .normalize(reference.normalized())
                .build()) {
      if (embedding.dimension() != reference.embeddingDimension()) {
        System.err.printf(
            "dimension mismatch: reference is %d, backend reports %d%n",
            reference.embeddingDimension(), embedding.dimension());
        return FAIL;
      }
      for (String probe : probes) {
        long probeStart = System.nanoTime();
        measured.add(embedding.embed(probe));
        millis.add((System.nanoTime() - probeStart) / 1_000_000.0);
      }
    }
    double elapsedSeconds = (System.nanoTime() - start) / 1_000_000_000.0;

    EmbeddingEquivalence.Result result =
        EmbeddingEquivalence.compare(reference.vectors(), measured, reference.normalized());

    printSummary(reference, probes, result, elapsedSeconds);
    if (report != null) {
      write(report, buildReport(reference, probes, result, artifact, artifactSha256, millis));
      System.out.println("report: " + report.toAbsolutePath());
    }
    return result.passed() ? PASS : FAIL;
  }

  private static void printSummary(
      EmbeddingEquivalence.Reference reference,
      List<String> probes,
      EmbeddingEquivalence.Result result,
      double elapsedSeconds) {
    System.out.printf(
        "%s vs %s %s | %d probes | %s pooling, dim %d%n",
        reference.model(),
        reference.oracleBackend(),
        reference.oracleVersion(),
        probes.size(),
        reference.pooling(),
        reference.embeddingDimension());
    for (int probe = 0; probe < probes.size(); probe++) {
      String text = probes.get(probe);
      System.out.printf(
          "  %.7f  %s%n",
          result.cosines().get(probe), text.length() <= 56 ? text : text.substring(0, 53) + "...");
    }
    System.out.printf(
        "min %.7f  mean %.7f  max component delta %.6f  floor %.4f%n",
        result.minimumCosine(),
        result.meanCosine(),
        result.maxComponentDelta(),
        EmbeddingEquivalence.MINIMUM_COSINE);
    if (reference.normalized()) {
      System.out.printf(
          "unit length: %s (max deviation %.2e, floor %.0e)%n",
          result.normalizationHeld() ? "held" : "BROKEN",
          result.maxNormDeviation(),
          EmbeddingEquivalence.MAX_NORM_DEVIATION);
    }
    System.out.printf(
        "%s in %.1fs%n", result.passed() ? "REPRODUCED" : "NOT REPRODUCED", elapsedSeconds);
  }

  private static ObjectNode buildReport(
      EmbeddingEquivalence.Reference reference,
      List<String> probes,
      EmbeddingEquivalence.Result result,
      Path artifact,
      String artifactSha256,
      List<Double> millis)
      throws IOException {
    ObjectMapper mapper = new ObjectMapper();
    ObjectNode root = mapper.createObjectNode();
    root.put("workload", "oracle-equivalence-v1");
    root.put("model", reference.model());
    root.put("backend", "pure-java");
    root.put("artifactSha256", artifactSha256);
    root.put("artifactSizeBytes", Files.size(artifact));
    root.put("probeSetSha256", reference.probeSetSha256());
    root.put("probes", probes.size());
    root.put("embeddingDimension", reference.embeddingDimension());
    root.put("pooling", reference.pooling());
    root.put("normalized", reference.normalized());
    root.put("oracleBackend", reference.oracleBackend());
    root.put("oracleVersion", reference.oracleVersion());
    root.put("minimumOracleCosine", result.minimumCosine());
    root.put("meanOracleCosine", result.meanCosine());
    root.put("maxComponentDelta", result.maxComponentDelta());
    root.put("maxNormDeviation", result.maxNormDeviation());
    root.put("minimumOracleCosineFloor", EmbeddingEquivalence.MINIMUM_COSINE);
    root.put("maxNormDeviationFloor", EmbeddingEquivalence.MAX_NORM_DEVIATION);
    root.put("normalizationHeld", result.normalizationHeld());
    root.put("qualified", result.passed());

    // Raw per-probe timings, not percentiles. Eight samples cannot support a p95, and dressing
    // them up as one invites reading a throughput claim into evidence that carries none.
    ArrayNode perProbe = root.putArray("perProbe");
    for (int probe = 0; probe < probes.size(); probe++) {
      ObjectNode entry = perProbe.addObject();
      entry.put("probe", probes.get(probe));
      entry.put("cosine", result.cosines().get(probe));
      entry.put("embedMillis", millis.get(probe));
    }

    root.set("environment", mapper.valueToTree(BenchmarkEnvironment.capture()));
    return root;
  }

  private static void write(Path report, ObjectNode node) throws IOException {
    Path parent = report.toAbsolutePath().getParent();
    if (parent != null) {
      Files.createDirectories(parent);
    }
    ObjectMapper mapper = new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT);
    Files.writeString(report, mapper.writeValueAsString(node) + System.lineSeparator());
  }

  private static Pooling pooling(String value) {
    return switch (value) {
      case "last-token" -> Pooling.LAST_TOKEN;
      case "mean" -> Pooling.MEAN;
      default -> throw new IllegalArgumentException("unsupported pooling: " + value);
    };
  }

  private static int usage(String problem) {
    System.err.println(problem);
    System.err.println(
        "usage: embedding-equivalence --model <artifact.gguf> [--report <out.json>]");
    return USAGE;
  }

  private static String requireValue(String[] args, int index, String flag) {
    if (index >= args.length) {
      throw new IllegalArgumentException(flag + " requires a value");
    }
    return args[index];
  }

  /**
   * Entry point for running the gate directly.
   *
   * @param args see {@link #run(String[])}
   * @throws IOException if the artifact or report path cannot be read or written
   */
  public static void main(String[] args) throws IOException {
    System.exit(run(Arrays.copyOf(args, args.length)));
  }
}
