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

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.EOFException;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * A released decision model: the standardiser, the head and the temperature that were measured
 * together, in one file that reproduces them exactly.
 *
 * <p>The three parts are stored together because they are only meaningful together. A head fitted
 * on standardised features scores raw features as noise, and a temperature fitted for one head
 * miscalibrates another. Shipping them separately invites a caller to assemble a combination that
 * was never measured, so this type is the unit of release and {@link #decide} is the whole path
 * from a hidden state to a calibrated verdict.
 *
 * <p>The format is deliberately plain: a magic number, a version, then big-endian IEEE-754 values
 * in a fixed order. Weights are written as {@code double} and restored bit for bit, because a
 * release is pinned by its digest and quoted by its benchmark, and both of those break under a
 * lossy round trip. Writing the same artifact twice produces the same bytes.
 *
 * <p>What this file does <em>not</em> contain is the base model. The artifact is the small learned
 * part — kilobytes, not gigabytes — and it names the base it was fitted against so a mismatch can
 * be caught rather than silently producing a plausible wrong answer.
 */
public final class DecisionArtifact {

  /** {@code IDSN} then the format version, so a wrong file fails immediately and by name. */
  private static final int MAGIC = 0x4944_534E;

  private static final int VERSION = 3;

  /**
   * The oldest layout this build still reads.
   *
   * <p>Version 3 added a per-label rubric after each answer space. Version 2 files simply stop
   * there, so they are read as an empty rubric rather than rejected -- an artifact is a record of a
   * decision that was already taken, and refusing to read one because a later build learned a new
   * field would destroy evidence to no purpose.
   */
  private static final int OLDEST_READABLE_VERSION = 2;

  private static final int KIND_NOUL = 0;
  private static final int KIND_CHOICE = 1;
  private static final int KIND_SCORE = 2;

  private final AnswerSpace space;
  private final FeatureStandardizer standardizer;
  private final LinearDecisionHead head;
  private final double temperature;
  private final String baseModel;
  private final String baseDigest;

  /**
   * Bundles a measured triple.
   *
   * @param space the answer space the head was fitted for
   * @param standardizer the statistics fitted on the training rows alone
   * @param head the fitted head, expecting standardised features
   * @param temperature the fitted temperature, {@code 1.0} for an uncalibrated release
   * @param baseModel the identifier of the base whose hidden states the head reads
   * @param baseDigest the base file's SHA-256, or empty if it was not recorded
   */
  public DecisionArtifact(
      AnswerSpace space,
      FeatureStandardizer standardizer,
      LinearDecisionHead head,
      double temperature,
      String baseModel,
      String baseDigest) {
    this.baseDigest = Objects.requireNonNull(baseDigest, "baseDigest");
    this.space = Objects.requireNonNull(space, "space");
    this.standardizer = Objects.requireNonNull(standardizer, "standardizer");
    this.head = Objects.requireNonNull(head, "head");
    this.baseModel = Objects.requireNonNull(baseModel, "baseModel");
    if (!(temperature > 0.0) || !Double.isFinite(temperature)) {
      throw new IllegalArgumentException("temperature must be finite and positive: " + temperature);
    }
    if (standardizer.width() != head.width()) {
      throw new IllegalArgumentException(
          "standardiser width "
              + standardizer.width()
              + " does not match head width "
              + head.width()
              + "; the head would read features it was never fitted on");
    }
    if (!head.space().equals(space)) {
      throw new IllegalArgumentException("the head was fitted for a different answer space");
    }
    this.temperature = temperature;
  }

  /** The answer space this artifact decides in. */
  public AnswerSpace space() {
    return space;
  }

  /** The fitted temperature. */
  public double temperature() {
    return temperature;
  }

  /** The base model the head reads hidden states from. */
  public String baseModel() {
    return baseModel;
  }

  /** The SHA-256 of the base file this head was fitted against, or empty if unrecorded. */
  public String baseDigest() {
    return baseDigest;
  }

  /**
   * Checks a base file is the one this head was fitted against.
   *
   * <p>A name is not an identity. Two files can carry the same model name and the same byte count
   * and hold different weights, and a head fed hidden states from the wrong weights returns
   * confident nonsense rather than an error. This is the check that turns that into a failure.
   *
   * @throws IllegalStateException if the digests disagree
   */
  public void requireBase(Path baseFile) throws IOException {
    if (baseDigest.isEmpty()) {
      return; // nothing was recorded, so there is nothing to contradict
    }
    String actual = digestOf(baseFile);
    if (!actual.equalsIgnoreCase(baseDigest)) {
      throw new IllegalStateException(
          "base mismatch: "
              + baseFile
              + " is sha256 "
              + actual
              + " but this head was fitted against "
              + baseDigest
              + "; the hidden states would come from different weights");
    }
  }

  /** SHA-256 of a file, as lowercase hex. */
  public static String digestOf(Path file) throws IOException {
    try {
      java.security.MessageDigest digest = java.security.MessageDigest.getInstance("SHA-256");
      try (java.io.InputStream in = new java.io.BufferedInputStream(Files.newInputStream(file))) {
        byte[] buffer = new byte[1 << 16];
        int read;
        while ((read = in.read(buffer)) != -1) {
          digest.update(buffer, 0, read);
        }
      }
      return java.util.HexFormat.of().formatHex(digest.digest());
    } catch (java.security.NoSuchAlgorithmException impossible) {
      throw new IllegalStateException(impossible);
    }
  }

  /** The hidden-state width this artifact requires. */
  public int width() {
    return head.width();
  }

  /**
   * Decides from a <em>raw</em> hidden state: standardise, score, calibrate.
   *
   * @param hiddenState the base model's hidden state, exactly as the base produced it
   */
  public Verdict decide(float[] hiddenState) {
    Objects.requireNonNull(hiddenState, "hiddenState");
    if (hiddenState.length != head.width()) {
      throw new IllegalArgumentException(
          "hidden state width "
              + hiddenState.length
              + " does not match the artifact's "
              + head.width());
    }
    double[] logits = head.logits(standardizer.apply(hiddenState));
    return new Verdict(space, Calibration.softmax(logits, temperature));
  }

  /** Writes this artifact, creating or truncating the file. */
  public void write(Path path) throws IOException {
    Objects.requireNonNull(path, "path");
    try (DataOutputStream out =
        new DataOutputStream(new java.io.BufferedOutputStream(Files.newOutputStream(path)))) {
      out.writeInt(MAGIC);
      out.writeInt(VERSION);
      writeString(out, baseModel);
      writeString(out, baseDigest);
      out.writeDouble(temperature);
      writeSpace(out, space);

      double[] mean = standardizer.meanInternal();
      double[] scale = standardizer.scaleInternal();
      out.writeInt(mean.length);
      for (double v : mean) {
        out.writeDouble(v);
      }
      for (double v : scale) {
        out.writeDouble(v);
      }

      double[][] weights = head.weightsInternal();
      double[] bias = head.biasInternal();
      out.writeInt(weights.length);
      out.writeInt(head.width());
      for (double[] row : weights) {
        for (double v : row) {
          out.writeDouble(v);
        }
      }
      for (double v : bias) {
        out.writeDouble(v);
      }
    }
  }

  /** Reads an artifact written by {@link #write}. */
  public static DecisionArtifact read(Path path) throws IOException {
    Objects.requireNonNull(path, "path");
    try (DataInputStream in =
        new DataInputStream(new java.io.BufferedInputStream(Files.newInputStream(path)))) {
      int magic = in.readInt();
      if (magic != MAGIC) {
        throw new IOException(
            String.format(
                "%s is not a decision artifact: magic 0x%08X, expected 0x%08X",
                path, magic, MAGIC));
      }
      int version = in.readInt();
      if (version < OLDEST_READABLE_VERSION || version > VERSION) {
        throw new IOException(
            "unsupported artifact version "
                + version
                + ", this build reads "
                + OLDEST_READABLE_VERSION
                + " to "
                + VERSION);
      }
      String baseModel = readString(in);
      String baseDigest = readString(in);
      double temperature = in.readDouble();
      AnswerSpace space = readSpace(in, version);

      int width = in.readInt();
      requireSane(width, "standardiser width");
      double[] mean = new double[width];
      double[] scale = new double[width];
      for (int i = 0; i < width; i++) {
        mean[i] = in.readDouble();
      }
      for (int i = 0; i < width; i++) {
        scale[i] = in.readDouble();
      }

      int rows = in.readInt();
      int cols = in.readInt();
      requireSane(rows, "head rows");
      requireSane(cols, "head width");
      double[][] weights = new double[rows][cols];
      for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
          weights[i][j] = in.readDouble();
        }
      }
      double[] bias = new double[rows];
      for (int i = 0; i < rows; i++) {
        bias[i] = in.readDouble();
      }
      if (in.read() != -1) {
        throw new IOException(path + " has trailing bytes after the artifact");
      }
      return new DecisionArtifact(
          space,
          FeatureStandardizer.of(mean, scale),
          new LinearDecisionHead(space, weights, bias),
          temperature,
          baseModel,
          baseDigest);
    } catch (EOFException truncated) {
      throw new IOException(
          path + " ends before the artifact does; the file is truncated", truncated);
    }
  }

  // A corrupt length must fail as a corrupt length, not as an OutOfMemoryError several frames away.
  private static void requireSane(int n, String what) throws IOException {
    if (n <= 0 || n > 1 << 24) {
      throw new IOException("implausible " + what + " in artifact: " + n);
    }
  }

  private static void writeSpace(DataOutputStream out, AnswerSpace space) throws IOException {
    switch (space) {
      case Noul noul -> {
        out.writeInt(KIND_NOUL);
        writeString(out, noul.proposition());
      }
      case Choice choice -> {
        out.writeInt(KIND_CHOICE);
        writeString(out, choice.question());
        writeLabels(out, choice.options());
      }
      case Score score -> {
        out.writeInt(KIND_SCORE);
        writeString(out, score.question());
        writeLabels(out, score.levels());
      }
    }
    // Version 3. The rubric is part of the question: the same labels with a different rubric are a
    // different decision, MEASURED 2026-09-24 at up to 0.67 of accuracy on one family. An artifact
    // that recorded the labels and dropped the rubric would not say what was actually asked.
    writeCriteria(out, space.criteria());
  }

  private static AnswerSpace readSpace(DataInputStream in, int version) throws IOException {
    int kind = in.readInt();
    return switch (kind) {
      case KIND_NOUL -> {
        String proposition = readString(in);
        yield new Noul(proposition, readCriteria(in, version));
      }
      case KIND_CHOICE -> {
        String question = readString(in);
        List<String> options = readLabels(in);
        yield new Choice(question, options, readCriteria(in, version));
      }
      case KIND_SCORE -> {
        String question = readString(in);
        List<String> levels = readLabels(in);
        yield new Score(question, levels, readCriteria(in, version));
      }
      default -> throw new IOException("unknown answer space kind " + kind);
    };
  }

  private static void writeCriteria(DataOutputStream out, Map<String, String> criteria)
      throws IOException {
    out.writeInt(criteria.size());
    for (Map.Entry<String, String> entry : criteria.entrySet()) {
      writeString(out, entry.getKey());
      writeString(out, entry.getValue());
    }
  }

  private static Map<String, String> readCriteria(DataInputStream in, int version)
      throws IOException {
    if (version < 3) {
      // Nothing to read: the field did not exist, and a version 2 artifact recorded a decision
      // taken without a rubric.
      return Map.of();
    }
    int count = in.readInt();
    if (count == 0) {
      // No rubric is the common case and a legitimate one, so it is checked before the sanity
      // bound, which exists to reject a corrupt length and treats zero as implausible.
      return Map.of();
    }
    requireSane(count, "criteria count");
    Map<String, String> criteria = new LinkedHashMap<>();
    for (int index = 0; index < count; index++) {
      String key = readString(in);
      criteria.put(key, readString(in));
    }
    return criteria;
  }

  private static void writeLabels(DataOutputStream out, List<String> labels) throws IOException {
    out.writeInt(labels.size());
    for (String label : labels) {
      writeString(out, label);
    }
  }

  private static List<String> readLabels(DataInputStream in) throws IOException {
    int n = in.readInt();
    requireSane(n, "label count");
    String[] labels = new String[n];
    for (int i = 0; i < n; i++) {
      labels[i] = readString(in);
    }
    return List.of(labels);
  }

  // Length-prefixed UTF-8 rather than DataOutput's modified UTF-8, which caps at 65535 bytes and
  // encodes NUL and astral characters differently from every other reader of this file.
  private static void writeString(DataOutputStream out, String value) throws IOException {
    byte[] bytes = value.getBytes(StandardCharsets.UTF_8);
    out.writeInt(bytes.length);
    out.write(bytes);
  }

  private static String readString(DataInputStream in) throws IOException {
    int n = in.readInt();
    if (n < 0 || n > 1 << 20) {
      throw new IOException("implausible string length in artifact: " + n);
    }
    byte[] bytes = new byte[n];
    in.readFully(bytes);
    return new String(bytes, StandardCharsets.UTF_8);
  }
}
