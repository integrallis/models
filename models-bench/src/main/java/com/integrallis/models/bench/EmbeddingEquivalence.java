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

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Compares embeddings produced here against vectors a reference implementation produced from the
 * same probe texts and the same model bytes.
 *
 * <p>Retrieval quality is a published property of the weights. What a runtime can get wrong is
 * pooling, rotary embeddings, dequantization, and normalization, so agreement with a reference is
 * what this measures.
 *
 * <p>The reference vectors are committed. They cannot drift unless the oracle build or the probe
 * set changes, and both are pinned by digest.
 */
public final class EmbeddingEquivalence {

  /**
   * Agreement below which the runtime is not considered to reproduce the model.
   *
   * <p>Mirrors {@code ModelEmbeddingQualification.MINIMUM_ORACLE_COSINE} in ModelJars. Duplicated
   * because ModelJars depends on this project, not the other way round; {@link
   * EmbeddingEquivalenceCli} writes the value into every report so the two cannot diverge silently.
   *
   * <p>Sits below the measured 0.99950 agreement and far above the 0.66156 a wrong pooling
   * produces.
   */
  public static final double MINIMUM_COSINE = 0.999;

  /**
   * How far a vector's length may sit from one before normalization is considered broken.
   *
   * <p>Cosine is scale-invariant, so a runtime that skips L2 normalization agrees with a normalized
   * reference at exactly 1.0. Length is therefore checked separately: callers that use a dot
   * product as a similarity shortcut depend on the vectors being unit length.
   *
   * <p>Loose enough for float32 accumulation, tight enough to catch an unnormalized vector.
   */
  public static final double MAX_NORM_DEVIATION = 1.0e-3;

  private static final String RESOURCE_ROOT = "/embedding-equivalence/";
  private static final String PROBES_RESOURCE = RESOURCE_ROOT + "probes.txt";
  private static final String REFERENCE_RESOURCE =
      RESOURCE_ROOT + "reference-qwen3-embedding-0.6b-q8_0.json";

  private EmbeddingEquivalence() {}

  /**
   * Vectors a pinned reference build produced from the pinned probe set.
   *
   * @param oracleBackend reference implementation that produced the vectors
   * @param oracleVersion exact pinned build of that implementation
   * @param model display name and variant of the model the vectors came from
   * @param artifactSha256 SHA-256 digest of the model artifact the vectors came from
   * @param probeSetSha256 SHA-256 digest of the probe texts the vectors came from
   * @param pooling how per-position states were reduced to one vector
   * @param normalized whether the vectors were scaled to unit length
   * @param embeddingDimension width of the vectors
   * @param vectors one reference vector per probe, in probe order
   */
  public record Reference(
      String oracleBackend,
      String oracleVersion,
      String model,
      String artifactSha256,
      String probeSetSha256,
      String pooling,
      boolean normalized,
      int embeddingDimension,
      List<float[]> vectors) {

    /**
     * Compact constructor deep-copying the reference vectors.
     *
     * <p>An immutable list of arrays is not immutable: {@code List.copyOf} freezes the list and
     * leaves every row writable. These vectors are the fixed point the gate measures against, so
     * nothing downstream may edit them.
     */
    public Reference {
      vectors = copy(vectors);
    }

    /**
     * Returns the reference vectors.
     *
     * @return a deep copy, so a caller cannot edit what later runs compare against
     */
    @Override
    public List<float[]> vectors() {
      return copy(vectors);
    }

    private static List<float[]> copy(List<float[]> source) {
      List<float[]> rows = new ArrayList<>(source.size());
      for (float[] row : source) {
        rows.add(row.clone());
      }
      return List.copyOf(rows);
    }
  }

  /**
   * Agreement measured across the probe set.
   *
   * @param cosines cosine agreement per probe, in probe order
   * @param minimumCosine lowest agreement observed, which is what the gate tests
   * @param meanCosine mean agreement, reported for context but never gated on
   * @param maxComponentDelta largest absolute per-component difference observed
   * @param maxNormDeviation largest distance from unit length among the produced vectors
   * @param unitNormExpected whether the reference was produced with normalization on
   */
  public record Result(
      List<Double> cosines,
      double minimumCosine,
      double meanCosine,
      double maxComponentDelta,
      double maxNormDeviation,
      boolean unitNormExpected) {

    /** Compact constructor taking a defensive copy of the per-probe agreements. */
    public Result {
      cosines = List.copyOf(cosines);
    }

    /**
     * Reports whether every probe reproduced the reference.
     *
     * @return whether the worst probe cleared both the agreement and the length checks
     */
    public boolean passed() {
      return minimumCosine >= MINIMUM_COSINE && normalizationHeld();
    }

    /**
     * Reports whether the produced vectors have the length the reference implies.
     *
     * @return true when normalization was not expected, or every vector is unit length
     */
    public boolean normalizationHeld() {
      return !unitNormExpected || maxNormDeviation <= MAX_NORM_DEVIATION;
    }
  }

  /**
   * Measures agreement between two aligned sets of vectors.
   *
   * <p>Gates on the worst probe: averaging lets one broken case hide behind seven good ones.
   *
   * @param reference vectors from the reference implementation, in probe order
   * @param actual vectors produced here, in the same order
   * @param expectUnitNorm whether the reference was produced with normalization on, in which case
   *     the produced vectors must also be unit length
   * @return the measured agreement
   * @throws IllegalArgumentException if the two sets are empty or do not align
   */
  public static Result compare(
      List<float[]> reference, List<float[]> actual, boolean expectUnitNorm) {
    Objects.requireNonNull(reference, "reference must not be null");
    Objects.requireNonNull(actual, "actual must not be null");
    if (reference.isEmpty()) {
      throw new IllegalArgumentException("probe count must be positive");
    }
    if (reference.size() != actual.size()) {
      throw new IllegalArgumentException(
          "probe count mismatch: reference has "
              + reference.size()
              + ", actual has "
              + actual.size());
    }

    List<Double> cosines = new ArrayList<>(reference.size());
    double minimum = Double.POSITIVE_INFINITY;
    double total = 0;
    double maxDelta = 0;
    double maxNormDeviation = 0;
    for (int probe = 0; probe < reference.size(); probe++) {
      float[] expected = reference.get(probe);
      float[] measured = actual.get(probe);
      if (expected.length != measured.length) {
        throw new IllegalArgumentException(
            "dimension mismatch at probe "
                + probe
                + ": reference is "
                + expected.length
                + ", actual is "
                + measured.length);
      }
      double cosine = cosine(expected, measured);
      cosines.add(cosine);
      minimum = Math.min(minimum, cosine);
      total += cosine;
      double squared = 0;
      for (int index = 0; index < expected.length; index++) {
        maxDelta = Math.max(maxDelta, Math.abs((double) expected[index] - measured[index]));
        squared += (double) measured[index] * measured[index];
      }
      maxNormDeviation = Math.max(maxNormDeviation, Math.abs(Math.sqrt(squared) - 1.0));
    }
    return new Result(
        cosines, minimum, total / reference.size(), maxDelta, maxNormDeviation, expectUnitNorm);
  }

  /**
   * Computes cosine similarity in double precision.
   *
   * <p>Does not assume the inputs are normalized, so a missing normalization shows up rather than
   * being silently corrected.
   *
   * @param left first vector
   * @param right second vector, of the same length
   * @return cosine similarity, or zero when either vector has no magnitude
   */
  static double cosine(float[] left, float[] right) {
    double dot = 0;
    double leftNorm = 0;
    double rightNorm = 0;
    for (int index = 0; index < left.length; index++) {
      dot += (double) left[index] * right[index];
      leftNorm += (double) left[index] * left[index];
      rightNorm += (double) right[index] * right[index];
    }
    double magnitude = Math.sqrt(leftNorm) * Math.sqrt(rightNorm);
    return magnitude == 0 ? 0 : dot / magnitude;
  }

  /**
   * Loads the pinned probe texts, one per line.
   *
   * @return the probe texts in the order the reference vectors were generated
   */
  public static List<String> loadProbes() {
    return loadProbesRaw().lines().filter(line -> !line.isEmpty()).toList();
  }

  /**
   * Loads the probe file exactly as committed.
   *
   * <p>Callers digest this rather than the parsed lines so the recorded {@code probeSetSha256}
   * covers whitespace and ordering too.
   *
   * @return the raw probe file contents
   */
  public static String loadProbesRaw() {
    return readResource(PROBES_RESOURCE);
  }

  /**
   * Loads the committed reference vectors.
   *
   * @return the pinned reference
   */
  public static Reference loadReference() {
    JsonNode root;
    try {
      root = new ObjectMapper().readTree(readResource(REFERENCE_RESOURCE));
    } catch (IOException failure) {
      throw new UncheckedIOException("reference vectors are not valid JSON", failure);
    }
    JsonNode rows = root.get("vectors");
    List<float[]> vectors = new ArrayList<>(rows.size());
    for (JsonNode row : rows) {
      float[] vector = new float[row.size()];
      for (int index = 0; index < vector.length; index++) {
        vector[index] = (float) row.get(index).asDouble();
      }
      vectors.add(vector);
    }
    return new Reference(
        root.get("oracleBackend").asText(),
        root.get("oracleVersion").asText(),
        root.get("model").asText(),
        root.get("artifactSha256").asText(),
        root.get("probeSetSha256").asText(),
        root.get("pooling").asText(),
        root.get("normalized").asBoolean(),
        root.get("embeddingDimension").asInt(),
        List.copyOf(vectors));
  }

  /**
   * Digests a value the way the recorded evidence does.
   *
   * @param value text to digest
   * @return lowercase hexadecimal SHA-256
   */
  public static String sha256(String value) {
    return Hashing.sha256(value);
  }

  private static String readResource(String name) {
    try (InputStream stream = EmbeddingEquivalence.class.getResourceAsStream(name)) {
      if (stream == null) {
        throw new IllegalStateException("missing packaged resource: " + name);
      }
      return new String(stream.readAllBytes(), StandardCharsets.UTF_8);
    } catch (IOException failure) {
      throw new UncheckedIOException("cannot read packaged resource: " + name, failure);
    }
  }
}
