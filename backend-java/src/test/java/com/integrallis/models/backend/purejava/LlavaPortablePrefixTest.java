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

import com.integrallis.models.api.InferenceSession;
import com.integrallis.models.backend.purejava.LlavaBackend.LlavaPrefix;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Random;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * #425: multimodal prefixes move between independently loaded LLaVA backends exactly, and only when
 * the receiver's own content (tokens and images) and identity (text model, projector,
 * preprocessing, layout) match.
 */
@Tag("unit")
class LlavaPortablePrefixTest {

  @TempDir static Path dir;
  static Path text;
  static Path projector;
  static Path otherProjector;
  static byte[] imageA;
  static byte[] imageB;

  @BeforeAll
  static void fixtures() throws Exception {
    text = PureJavaBackendTest.buildNanoModelFile(dir, new Random(42));
    projector = SyntheticLlava.writeProjector(dir.resolve("mmproj.gguf"), 32, 7);
    otherProjector = SyntheticLlava.writeProjector(dir.resolve("mmproj-other.gguf"), 32, 8);
    imageA = SyntheticLlava.image(1);
    imageB = SyntheticLlava.image(2);
  }

  private static final int[] HEAD = {2, 3};
  private static final int[] MIDDLE = {4, 5, 6};
  private static final int[] TAIL = {7};
  private static final int NEXT = 9;

  /** Reads text, image, text, image, text into a session; returns it. */
  private static InferenceSession readMixed(LlavaBackend llava, byte[] first, byte[] second) {
    InferenceSession session = llava.openSession();
    int at = 0;
    llava.prefill(session, HEAD, at);
    at += HEAD.length;
    llava.prefillImage(session, first, at);
    at += llava.imageTokenCount();
    llava.prefill(session, MIDDLE, at);
    at += MIDDLE.length;
    llava.prefillImage(session, second, at);
    at += llava.imageTokenCount();
    llava.prefill(session, TAIL, at);
    return session;
  }

  private static MultimodalContent expected(byte[] first, byte[] second) {
    return MultimodalContent.builder()
        .tokens(HEAD)
        .image(first, SyntheticLlava.IMAGE_ROWS)
        .tokens(MIDDLE)
        .image(second, SyntheticLlava.IMAGE_ROWS)
        .tokens(TAIL)
        .build();
  }

  private static byte[] export(LlavaBackend llava, LlavaPrefix prefix) throws IOException {
    var buffer = new ByteArrayOutputStream();
    llava.writePrefix(prefix, new DataOutputStream(buffer));
    return buffer.toByteArray();
  }

  private static LlavaPrefix importInto(
      LlavaBackend llava, byte[] bytes, MultimodalContent expected) throws IOException {
    return llava.readPrefix(
        new DataInputStream(new ByteArrayInputStream(bytes)), bytes.length, expected);
  }

  @Test
  void mixedImageAndTextPrefixContinuesExactlyOnAnIndependentReceiverAndMatchesRecomputation()
      throws Exception {
    try (var donor = LlavaBackend.load(text, projector);
        var receiver = LlavaBackend.load(text, projector)) {
      assertThat(donor.portablePrefixCapabilities().supported()).isTrue();
      assertThat(donor.imageTokenCount()).isEqualTo(SyntheticLlava.IMAGE_ROWS);
      LlavaPrefix prefix = donor.freezePrefix(readMixed(donor, imageA, imageB));
      int positions = HEAD.length + MIDDLE.length + TAIL.length + 2 * SyntheticLlava.IMAGE_ROWS;
      assertThat(prefix.checkpoint()).isEqualTo(positions);
      assertThat(prefix.content()).isEqualTo(expected(imageA, imageB));

      byte[] bytes = export(donor, prefix);
      assertThat((long) bytes.length).isEqualTo(donor.portablePrefixBytes(prefix));
      LlavaPrefix imported = importInto(receiver, bytes, expected(imageA, imageB));

      InferenceSession recomputed = readMixed(receiver, imageA, imageB);
      InferenceSession fromDonor = donor.fork(prefix);
      InferenceSession fromImport = receiver.fork(imported);
      float[] reference = donor.forward(fromDonor, NEXT, positions);
      assertThat(receiver.forward(fromImport, NEXT, positions)).containsExactly(reference);
      assertThat(receiver.forward(recomputed, NEXT, positions)).containsExactly(reference);
      fromDonor.close();
      fromImport.close();
      recomputed.close();
    }
  }

  @Test
  void theImagesMatterSoTheExactnessAboveIsNotVacuous() throws Exception {
    try (var llava = LlavaBackend.load(text, projector)) {
      int positions = HEAD.length + MIDDLE.length + TAIL.length + 2 * SyntheticLlava.IMAGE_ROWS;
      InferenceSession withA = readMixed(llava, imageA, imageA);
      InferenceSession withB = readMixed(llava, imageB, imageA);
      float[] a = llava.forward(withA, NEXT, positions);
      float[] b = llava.forward(withB, NEXT, positions);
      assertThat(Arrays.equals(a, b)).isFalse();
      withA.close();
      withB.close();
    }
  }

  @Test
  void aDifferentImageBehindIdenticalPlaceholdersIsRefusedAndTheReceiverStaysUsable()
      throws Exception {
    try (var donor = LlavaBackend.load(text, projector);
        var receiver = LlavaBackend.load(text, projector)) {
      byte[] bytes = export(donor, donor.freezePrefix(readMixed(donor, imageA, imageB)));
      assertThatThrownBy(() -> importInto(receiver, bytes, expected(imageA, imageA)))
          .isInstanceOf(IOException.class)
          .hasMessageContaining("segment 3");
      assertThatThrownBy(() -> importInto(receiver, bytes, expected(imageB, imageB)))
          .isInstanceOf(IOException.class)
          .hasMessageContaining("segment 1");
      // Different tokens, same images.
      MultimodalContent otherTokens =
          MultimodalContent.builder()
              .tokens(HEAD)
              .image(imageA, SyntheticLlava.IMAGE_ROWS)
              .tokens(4, 5, 8)
              .image(imageB, SyntheticLlava.IMAGE_ROWS)
              .tokens(TAIL)
              .build();
      assertThatThrownBy(() -> importInto(receiver, bytes, otherTokens))
          .isInstanceOf(IOException.class)
          .hasMessageContaining("segment 2");
      assertThat(importInto(receiver, bytes, expected(imageA, imageB)).content())
          .isEqualTo(expected(imageA, imageB));
    }
  }

  @Test
  void aDifferentProjectorIsADifferentIdentity() throws Exception {
    try (var donor = LlavaBackend.load(text, projector);
        var receiver = LlavaBackend.load(text, otherProjector)) {
      assertThat(donor.portablePrefixFingerprint())
          .isNotEqualTo(receiver.portablePrefixFingerprint());
      byte[] bytes = export(donor, donor.freezePrefix(readMixed(donor, imageA, imageB)));
      assertThatThrownBy(() -> importInto(receiver, bytes, expected(imageA, imageB)))
          .isInstanceOf(IOException.class)
          .hasMessageContaining("incompatible");
    }
  }

  @Test
  void aTamperedImageDigestABadFormatAndAnOverBudgetPayloadAreRefused() throws Exception {
    try (var donor = LlavaBackend.load(text, projector);
        var receiver = LlavaBackend.load(text, projector)) {
      byte[] bytes = export(donor, donor.freezePrefix(readMixed(donor, imageA, imageB)));
      // The first image digest sits after magic(4) + identity(32) + count(4) + the head run
      // (kind 1 + length 4 + 2 tokens x 4) + the image's kind and length (5).
      byte[] tampered = bytes.clone();
      tampered[4 + 32 + 4 + (1 + 4 + 8) + 5] ^= 1;
      assertThatThrownBy(() -> importInto(receiver, tampered, expected(imageA, imageB)))
          .isInstanceOf(IOException.class)
          .hasMessageContaining("segment 1");
      byte[] wrongFormat = bytes.clone();
      wrongFormat[0] ^= 1;
      assertThatThrownBy(() -> importInto(receiver, wrongFormat, expected(imageA, imageB)))
          .isInstanceOf(IOException.class)
          .hasMessageContaining("not a LLaVA prefix");
      assertThatThrownBy(
              () ->
                  receiver.readPrefix(
                      new DataInputStream(new ByteArrayInputStream(bytes)),
                      60,
                      expected(imageA, imageB)))
          .isInstanceOf(IOException.class)
          .hasMessageContaining("budget");
    }
  }

  @Test
  void ownershipAndLifetimes() throws Exception {
    try (var first = LlavaBackend.load(text, projector);
        var second = LlavaBackend.load(text, projector)) {
      InferenceSession session = readMixed(first, imageA, imageB);
      LlavaPrefix prefix = first.freezePrefix(session);
      assertThat(session.isClosed()).as("freezing consumes the session").isTrue();
      assertThatThrownBy(() -> second.fork(prefix))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("another backend");
      InferenceSession one = first.fork(prefix);
      InferenceSession two = first.fork(prefix);
      int positions = prefix.checkpoint();
      float[] reference = first.forward(one, NEXT, positions);
      one.close();
      assertThat(first.forward(two, NEXT, positions))
          .as("closing one fork leaves another intact")
          .containsExactly(reference);
      two.close();
      assertThatThrownBy(() -> first.prefill(one, TAIL, positions))
          .isInstanceOf(IllegalStateException.class);
    }
    var closed = LlavaBackend.load(text, projector);
    closed.close();
    assertThatThrownBy(closed::openSession).isInstanceOf(IllegalStateException.class);
  }

  @Test
  void contentIsCanonicalAndAPrefixMayNotEndInsideAnImage() {
    MultimodalContent split =
        MultimodalContent.builder().tokens(1, 2).tokens(3).image(new byte[] {1}, 4).build();
    MultimodalContent merged =
        MultimodalContent.builder().tokens(1, 2, 3).image(new byte[] {1}, 4).build();
    assertThat(split).isEqualTo(merged);
    assertThat(split.firstDifference(merged)).isNull();
    assertThat(merged.positions()).isEqualTo(7);
    assertThat(merged.prefix(2).segments()).hasSize(1);
    assertThat(merged.prefix(7)).isEqualTo(merged);
    assertThatThrownBy(() -> merged.prefix(5))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("inside an image");
  }
}
