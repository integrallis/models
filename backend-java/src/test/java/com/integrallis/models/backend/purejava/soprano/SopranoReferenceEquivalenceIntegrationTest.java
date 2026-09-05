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
package com.integrallis.models.backend.purejava.soprano;

import static java.nio.ByteOrder.LITTLE_ENDIAN;
import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.HexFormat;
import java.util.Set;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("integration")
class SopranoReferenceEquivalenceIntegrationTest {

  private static final String PROMPT = "The JVM can speak for itself.";
  private static final Set<String> QUALIFICATION_ARTIFACT_SHA256 =
      Set.of(
          "4758ad908395dc73a1b973d9a29ce96941f4328594d1c6c1223e7b7710a6a131",
          "46c60cdf5b8e5a3b26bfd185c0870826c05b533979ea7c1d399a9ffb76a50a54");

  @Test
  void matchesTheOfficialF32LanguageModelAtTheFinalPromptPosition() throws Exception {
    Path artifact = configuredArtifact();

    try (SopranoBackend backend = SopranoBackend.load(artifact)) {
      int[] expectedTokens = readInts("prompt-tokens.i32le");
      assertThat(backend.encodePrompt(PROMPT)).containsExactly(expectedTokens);

      SopranoBackend.Step actual = backend.begin(PROMPT);
      float[] expectedLogits = readFloats("prompt-logits.f32le");
      float[] expectedHidden = readFloats("prompt-hidden.f32le");
      Comparison logits = compare(expectedLogits, actual.logits());
      Comparison hidden = compare(expectedHidden, actual.hiddenState());

      System.out.printf(
          "Soprano LM oracle: logits cosine=%.9f maxAbs=%.6f; hidden cosine=%.9f maxAbs=%.6f%n",
          logits.cosine(), logits.maxAbsoluteError(), hidden.cosine(), hidden.maxAbsoluteError());
      assertThat(logits.cosine()).isGreaterThan(0.999);
      assertThat(logits.maxAbsoluteError()).isLessThan(0.15);
      assertThat(hidden.cosine()).isGreaterThan(0.999);
      assertThat(hidden.maxAbsoluteError()).isLessThan(0.1);
    }
  }

  @Test
  void matchesTheOfficialF32VocoderForDeterministicFeatures() throws Exception {
    Path artifact = configuredArtifact();

    try (SopranoBackend backend = SopranoBackend.load(artifact)) {
      float[] features = readFloats("vocoder-features.f32le");
      float[] expectedPcm = readFloats("vocoder-pcm.f32le");
      float[] actualPcm = backend.decode(features, 4);
      Comparison pcm = compare(expectedPcm, actualPcm);
      double signalToDifferenceDb = signalToDifferenceDb(expectedPcm, actualPcm);

      System.out.printf(
          "Soprano vocoder oracle: cosine=%.9f maxAbs=%.6f SDR=%.3f dB%n",
          pcm.cosine(), pcm.maxAbsoluteError(), signalToDifferenceDb);
      assertThat(pcm.cosine()).isGreaterThan(0.995);
      assertThat(pcm.maxAbsoluteError()).isLessThan(0.02);
      assertThat(signalToDifferenceDb).isGreaterThan(20.0);
    }
  }

  private static Path configuredArtifact() throws Exception {
    String configured = System.getenv("MODELS_SOPRANO_GGUF");
    assumeTrue(configured != null && Files.isRegularFile(Path.of(configured)));
    Path artifact = Path.of(configured);
    assertThat(QUALIFICATION_ARTIFACT_SHA256).contains(sha256(artifact));
    return artifact;
  }

  private static String sha256(Path path) throws Exception {
    MessageDigest digest = MessageDigest.getInstance("SHA-256");
    try (InputStream stream = Files.newInputStream(path)) {
      byte[] buffer = new byte[1024 * 1024];
      int read;
      while ((read = stream.read(buffer)) >= 0) {
        digest.update(buffer, 0, read);
      }
    }
    return HexFormat.of().formatHex(digest.digest());
  }

  private static int[] readInts(String name) throws IOException {
    ByteBuffer buffer = read(name).order(LITTLE_ENDIAN);
    int[] values = new int[buffer.remaining() / Integer.BYTES];
    buffer.asIntBuffer().get(values);
    return values;
  }

  private static float[] readFloats(String name) throws IOException {
    ByteBuffer buffer = read(name).order(LITTLE_ENDIAN);
    float[] values = new float[buffer.remaining() / Float.BYTES];
    buffer.asFloatBuffer().get(values);
    return values;
  }

  private static ByteBuffer read(String name) throws IOException {
    String resource = "/soprano/oracle/" + name;
    try (InputStream stream =
        SopranoReferenceEquivalenceIntegrationTest.class.getResourceAsStream(resource)) {
      if (stream == null) {
        throw new IOException("Missing Soprano oracle resource " + resource);
      }
      return ByteBuffer.wrap(stream.readAllBytes());
    }
  }

  private static Comparison compare(float[] expected, float[] actual) {
    assertThat(actual).hasSameSizeAs(expected);
    double dot = 0.0;
    double expectedNorm = 0.0;
    double actualNorm = 0.0;
    double maxAbsoluteError = 0.0;
    for (int index = 0; index < expected.length; index++) {
      dot += (double) expected[index] * actual[index];
      expectedNorm += (double) expected[index] * expected[index];
      actualNorm += (double) actual[index] * actual[index];
      maxAbsoluteError = Math.max(maxAbsoluteError, Math.abs(expected[index] - actual[index]));
    }
    return new Comparison(dot / Math.sqrt(expectedNorm * actualNorm), maxAbsoluteError);
  }

  private static double signalToDifferenceDb(float[] expected, float[] actual) {
    double signal = 0.0;
    double difference = 0.0;
    for (int index = 0; index < expected.length; index++) {
      signal += (double) expected[index] * expected[index];
      double error = expected[index] - actual[index];
      difference += error * error;
    }
    return 10.0 * Math.log10(signal / difference);
  }

  private record Comparison(double cosine, double maxAbsoluteError) {}
}
