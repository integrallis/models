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

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * A released artifact must reproduce the decisions it was measured making.
 *
 * <p>Every assertion here is about exactness rather than closeness. A head that is nearly restored
 * is a different model from the one whose numbers were published, and the difference would surface
 * as an unreproducible benchmark long after the release.
 */
final class DecisionArtifactTest {

  @TempDir Path dir;

  private static LinearDecisionHead head(AnswerSpace space, int width) {
    int k = space.labels().size();
    double[][] weights = new double[k][width];
    double[] bias = new double[k];
    for (int i = 0; i < k; i++) {
      bias[i] = (i + 1) * 0.125 - 0.5;
      for (int j = 0; j < width; j++) {
        // Values chosen to need every mantissa bit, so a lossy format cannot pass.
        weights[i][j] = Math.sin((i + 1) * 7919.0 + j * 104_729.0) * 1e-3;
      }
    }
    return new LinearDecisionHead(space, weights, bias);
  }

  private static FeatureStandardizer standardizer(int width) {
    float[][] rows = new float[16][width];
    for (int i = 0; i < rows.length; i++) {
      for (int j = 0; j < width; j++) {
        rows[i][j] = (float) Math.cos(i * 31.0 + j * 17.0);
      }
    }
    return FeatureStandardizer.fit(rows);
  }

  @Test
  void roundTripsANoulBitForBit() throws IOException {
    int width = 24;
    Noul space = new Noul("the state answers the question");
    DecisionArtifact written =
        new DecisionArtifact(
            space, standardizer(width), head(space, width), 3.2794, "granite-4.1-3b", "abc123");
    Path file = dir.resolve("noul.idsn");
    written.write(file);

    DecisionArtifact read = DecisionArtifact.read(file);

    assertThat(read.space()).isEqualTo(space);
    assertThat(read.temperature()).isEqualTo(3.2794);
    assertThat(read.baseModel()).isEqualTo("granite-4.1-3b");
    assertThat(read.baseDigest()).isEqualTo("abc123");

    // The decision itself, not just the stored numbers, must come back identical.
    float[] state = new float[width];
    for (int j = 0; j < width; j++) {
      state[j] = (float) Math.tan(j * 0.37);
    }
    assertThat(read.decide(state).probabilities()).isEqualTo(written.decide(state).probabilities());
  }

  @Test
  void refusesABaseWhoseDigestDisagrees() throws IOException {
    // Two files can share a model name and a byte count and hold different weights. A head fed the
    // wrong weights returns confident nonsense, so the digest has to be the thing that is checked.
    int width = 8;
    Noul space = new Noul("p");
    Path base = dir.resolve("base.gguf");
    Files.write(base, new byte[] {1, 2, 3, 4});
    Path other = dir.resolve("other.gguf");
    Files.write(other, new byte[] {4, 3, 2, 1});

    DecisionArtifact artifact =
        new DecisionArtifact(
            space,
            standardizer(width),
            head(space, width),
            1.0,
            "base",
            DecisionArtifact.digestOf(base));

    artifact.requireBase(base); // the right file passes
    assertThatThrownBy(() -> artifact.requireBase(other))
        .isInstanceOf(IllegalStateException.class)
        .hasMessageContaining("base mismatch");
  }

  @Test
  void acceptsAnyBaseWhenNoDigestWasRecorded() throws IOException {
    int width = 8;
    Noul space = new Noul("p");
    Path any = dir.resolve("any.gguf");
    Files.write(any, new byte[] {9});
    // An older artifact records nothing, and silence must not be read as a contradiction.
    new DecisionArtifact(space, standardizer(width), head(space, width), 1.0, "base", "")
        .requireBase(any);
  }

  @Test
  void roundTripsAChoiceAndAScore() throws IOException {
    int width = 12;
    Choice choice = new Choice("which tool", List.of("search", "calculator", "none"));
    Score score = new Score("how severe", List.of("1", "2", "3", "4"));
    for (AnswerSpace space : List.of(choice, score)) {
      Path file = dir.resolve(space.getClass().getSimpleName() + ".idsn");
      DecisionArtifact written =
          new DecisionArtifact(space, standardizer(width), head(space, width), 1.5, "base", "");
      written.write(file);
      assertThat(DecisionArtifact.read(file).space()).isEqualTo(space);
    }
  }

  @Test
  void rejectsAFileThatIsNotAnArtifact() throws IOException {
    Path file = dir.resolve("rubbish.idsn");
    Files.write(file, "not a decision artifact".getBytes(java.nio.charset.StandardCharsets.UTF_8));
    assertThatThrownBy(() -> DecisionArtifact.read(file))
        .isInstanceOf(IOException.class)
        .hasMessageContaining("magic");
  }

  @Test
  void rejectsATruncatedFile() throws IOException {
    int width = 8;
    Noul space = new Noul("p");
    Path file = dir.resolve("short.idsn");
    new DecisionArtifact(space, standardizer(width), head(space, width), 1.0, "base", "")
        .write(file);
    byte[] all = Files.readAllBytes(file);
    Files.write(file, java.util.Arrays.copyOf(all, all.length - 9));

    assertThatThrownBy(() -> DecisionArtifact.read(file)).isInstanceOf(IOException.class);
  }

  @Test
  void refusesAStandardizerThatDoesNotMatchTheHead() {
    Noul space = new Noul("p");
    assertThatThrownBy(
            () -> new DecisionArtifact(space, standardizer(8), head(space, 16), 1.0, "base", ""))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("width");
  }

  @Test
  void writingTheSameArtifactTwiceProducesTheSameBytes() throws IOException {
    int width = 10;
    Noul space = new Noul("p");
    DecisionArtifact a =
        new DecisionArtifact(space, standardizer(width), head(space, width), 2.0, "base", "");
    Path one = dir.resolve("one.idsn");
    Path two = dir.resolve("two.idsn");
    a.write(one);
    a.write(two);
    // A reproducible artifact is a hashable artifact; a release is pinned by its digest.
    assertThat(Files.readAllBytes(one)).isEqualTo(Files.readAllBytes(two));
  }

  @Test
  void appliesStandardisationBeforeTheHead() throws IOException {
    // The artifact owns the whole path. A caller that had to remember to standardise first would
    // eventually forget, and the failure would look like a merely disappointing model.
    int width = 6;
    Noul space = new Noul("p");
    FeatureStandardizer std = standardizer(width);
    LinearDecisionHead h = head(space, width);
    DecisionArtifact artifact = new DecisionArtifact(space, std, h, 1.0, "base", "");

    float[] raw = new float[width];
    for (int j = 0; j < width; j++) {
      raw[j] = 3.5f + j;
    }
    double[] expected = Calibration.softmax(h.logits(std.apply(raw)), 1.0);
    assertThat(artifact.decide(raw).probabilities()).isEqualTo(expected);
  }
}
