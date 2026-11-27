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

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.integrallis.models.api.InferenceSession;
import com.integrallis.models.backend.purejava.LlavaBackend.LlavaPrefix;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import javax.imageio.ImageIO;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * #425 on real weights: a LLaVA 1.5 7B prefix holding text and an image, frozen on one backend,
 * moved as bytes to an independently loaded backend, continues exactly as the donor does and as the
 * receiver recomputing it does; the same bytes are refused for a different image behind the same
 * placeholders. Skipped unless the pinned files are present ({@code -Dmodels.fixtures.llavaText},
 * {@code -Dmodels.fixtures.llavaProjector}).
 */
@Tag("model-fixture")
class LlavaRealModelPortablePrefixIntegrationTest {

  private static Path fixture(String property) {
    String value = System.getProperty(property);
    assumeTrue(value != null, property + " is not set");
    Path path = Path.of(value);
    assumeTrue(Files.isRegularFile(path), path + " is not a file");
    return path;
  }

  /** A 336x336 PNG whose pixels are a pure function of {@code seed}. */
  private static byte[] image(int seed) throws IOException {
    BufferedImage image = new BufferedImage(336, 336, BufferedImage.TYPE_INT_RGB);
    for (int y = 0; y < 336; y++) {
      for (int x = 0; x < 336; x++) {
        int r = (x * 3 + seed * 41) & 0xff;
        int g = (y * 5 + seed * 17) & 0xff;
        int b = ((x ^ y) + seed * 97) & 0xff;
        image.setRGB(x, y, (r << 16) | (g << 8) | b);
      }
    }
    var out = new ByteArrayOutputStream();
    ImageIO.write(image, "png", out);
    return out.toByteArray();
  }

  @Test
  void aTextAndImagePrefixMovesExactlyBetweenIndependentBackends() throws Exception {
    Path text = fixture("models.fixtures.llavaText");
    Path projector = fixture("models.fixtures.llavaProjector");
    byte[] picture = image(1);
    byte[] otherPicture = image(2);
    try (var donor = LlavaBackend.load(text, projector);
        var receiver = LlavaBackend.load(text, projector)) {
      assumeTrue(donor.portablePrefixCapabilities().supported(), "prefixes not portable here");
      int[] before = donor.tokenizer().encode("USER: ");
      int[] after = donor.tokenizer().encode("\nWhat is in the image? ASSISTANT:");
      int rows = donor.imageTokenCount();
      int positions = before.length + rows + after.length;
      MultimodalContent content =
          MultimodalContent.builder().tokens(before).image(picture, rows).tokens(after).build();

      long t0 = System.nanoTime();
      InferenceSession session = donor.openSession();
      donor.prefill(session, before, 0);
      donor.prefillImage(session, picture, before.length);
      donor.prefill(session, after, before.length + rows);
      LlavaPrefix prefix = donor.freezePrefix(session);
      long donorMs = (System.nanoTime() - t0) / 1_000_000;

      var buffer = new ByteArrayOutputStream();
      donor.writePrefix(prefix, new DataOutputStream(buffer));
      byte[] bytes = buffer.toByteArray();
      long t1 = System.nanoTime();
      LlavaPrefix imported =
          receiver.readPrefix(
              new DataInputStream(new ByteArrayInputStream(bytes)), bytes.length, content);
      long importMs = (System.nanoTime() - t1) / 1_000_000;

      long t2 = System.nanoTime();
      InferenceSession recomputed = receiver.openSession();
      receiver.prefill(recomputed, before, 0);
      receiver.prefillImage(recomputed, picture, before.length);
      receiver.prefill(recomputed, after, before.length + rows);
      long recomputeMs = (System.nanoTime() - t2) / 1_000_000;

      InferenceSession fromDonor = donor.fork(prefix);
      InferenceSession fromImport = receiver.fork(imported);
      int next = donor.tokenizer().encode(" The")[0];
      float[] reference = donor.forward(fromDonor, next, positions);
      float[] transferred = receiver.forward(fromImport, next, positions);
      float[] recomputedLogits = receiver.forward(recomputed, next, positions);
      System.out.printf(
          "LLAVA-PORTABLE positions=%d imageRows=%d bytes=%d donorPrefillMs=%d importMs=%d"
              + " recomputeMs=%d argmax=%d identicalToDonor=%s identicalToRecompute=%s%n",
          positions,
          rows,
          bytes.length,
          donorMs,
          importMs,
          recomputeMs,
          argmax(reference),
          Arrays.equals(reference, transferred),
          Arrays.equals(recomputedLogits, transferred));
      assertThat(transferred).containsExactly(reference);
      assertThat(recomputedLogits).containsExactly(reference);

      MultimodalContent otherImage =
          MultimodalContent.builder()
              .tokens(before)
              .image(otherPicture, rows)
              .tokens(after)
              .build();
      assertThatThrownBy(
              () ->
                  receiver.readPrefix(
                      new DataInputStream(new ByteArrayInputStream(bytes)),
                      bytes.length,
                      otherImage))
          .isInstanceOf(IOException.class)
          .hasMessageContaining("segment 1");
      fromDonor.close();
      fromImport.close();
      recomputed.close();
    }
  }

  private static int argmax(float[] values) {
    int best = 0;
    for (int i = 1; i < values.length; i++) if (values[i] > values[best]) best = i;
    return best;
  }
}
