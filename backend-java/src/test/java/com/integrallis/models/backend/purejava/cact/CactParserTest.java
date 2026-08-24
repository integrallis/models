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
package com.integrallis.models.backend.purejava.cact;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.integrallis.models.backend.purejava.fixture.ModelFixtureRegistry;
import java.io.IOException;
import java.io.InputStream;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

@Tag("unit")
class CactParserTest {

  private static final int HEADER_BYTES = 120;
  private static final int CODEBOOK_BYTES = 28 * Float.BYTES;
  private static final int DIRECTORY_OFFSET = HEADER_BYTES + CODEBOOK_BYTES;
  private static final int RECORD_BYTES = 44;
  private static final String NEEDLE2_SHA256 =
      "b43aabfcaf1a6db6acf488076eab71d823c08697c7af4521fc1d174b60ede5ba";

  @Test
  void parsesHeaderDirectoryAndTensorSlices() {
    byte[] data =
        new SyntheticCactBuilder()
            .addCq(3, 133, 128, 2)
            .addFp16(8)
            .addRaw((byte) 1, (byte) 2, (byte) 3)
            .build();

    CactFile file = CactParser.parseSegment(MemorySegment.ofArray(data));

    assertThat(file.header().vocabularySize()).isEqualTo(32);
    assertThat(file.header().modelWidth()).isEqualTo(8);
    assertThat(file.header().engramOrders()).containsExactly(2, 3);
    assertThat(file.header().engramSites()).containsExactly(0);
    assertThat(file.codebook()).hasSize(28);
    assertThat(file.tensorInfos()).hasSize(3);

    CactTensorInfo cq = file.tensorInfos().getFirst();
    assertThat(cq.index()).isZero();
    assertThat(cq.type()).isEqualTo(CactTensorType.CQ);
    assertThat(cq.shape()).containsExactly(3, 133);
    assertThat(cq.groupSize()).isEqualTo(128);
    assertThat(cq.recordBits()).isEqualTo(2);
    assertThat(cq.packedCodeBytes()).isEqualTo(192);
    assertThat(cq.normBytes()).isEqualTo(12);
    assertThat(file.tensor(0).data().byteSize()).isEqualTo(204);
    assertThat(file.fileSegment().isReadOnly()).isTrue();
    assertThat(file.tensor(0).data().isReadOnly()).isTrue();
    assertThat(file.tensor(2).data().toArray(java.lang.foreign.ValueLayout.JAVA_BYTE))
        .containsExactly(1, 2, 3);
  }

  @Test
  void parsesMappedFile(@TempDir Path directory) throws IOException {
    byte[] data = new SyntheticCactBuilder().addFp16(4).build();
    Path path = directory.resolve("tiny.cact");
    Files.write(path, data);

    try (Arena arena = Arena.ofConfined()) {
      CactFile file = CactParser.parse(path, arena);
      assertThat(file.tensorInfos())
          .singleElement()
          .satisfies(tensor -> assertThat(tensor.shape()).containsExactly(4));
    }
  }

  @Test
  void fixtureRegistryPreservesTheCactFormat() {
    var descriptor =
        ModelFixtureRegistry.fromClasspath().descriptors().stream()
            .filter(candidate -> candidate.id().equals("needle2_cq2"))
            .findFirst()
            .orElseThrow();

    assertThat(descriptor.format()).isEqualTo("cact");
    assertThat(descriptor.architecture()).isEqualTo("needle2");
    assertThat(descriptor.backend()).isEqualTo("parser");
    assertThat(descriptor.capabilities()).contains("tool-calling", "structured-output");
  }

  @Test
  void rejectsBadMagicAndTruncatedStructures() {
    byte[] badMagic = new SyntheticCactBuilder().withMagic(0).addFp16(4).build();
    byte[] valid = new SyntheticCactBuilder().addFp16(4).build();

    assertMalformed(badMagic, "magic");
    assertMalformed(new byte[HEADER_BYTES - 1], "header");
    assertMalformed(
        java.util.Arrays.copyOf(valid, DIRECTORY_OFFSET + RECORD_BYTES - 1), "directory");
  }

  @Test
  void rejectsMisalignedOverlappingAndOutOfRangeTensors() {
    byte[] misaligned = new SyntheticCactBuilder().addFp16(4).build();
    putLong(misaligned, DIRECTORY_OFFSET + 20, getLong(misaligned, DIRECTORY_OFFSET + 20) + 1);

    byte[] overlap = new SyntheticCactBuilder().addFp16(4).addFp16(4).build();
    putLong(overlap, DIRECTORY_OFFSET + RECORD_BYTES + 20, getLong(overlap, DIRECTORY_OFFSET + 20));

    byte[] outOfRange = new SyntheticCactBuilder().addFp16(4).build();
    putLong(outOfRange, DIRECTORY_OFFSET + 28, Long.MAX_VALUE);

    assertMalformed(misaligned, "aligned");
    assertMalformed(overlap, "overlap");
    assertMalformed(outOfRange, "range");
  }

  @Test
  void rejectsInvalidTensorMetadataAndCqLayout() {
    byte[] unsupportedType = new SyntheticCactBuilder().addFp16(4).build();
    unsupportedType[DIRECTORY_OFFSET] = 99;

    byte[] nonZeroTail = new SyntheticCactBuilder().addFp16(4).build();
    putInt(nonZeroTail, DIRECTORY_OFFSET + 8, 7);

    byte[] wrongCqBytes = new SyntheticCactBuilder().addCq(3, 133, 128, 4).build();
    putLong(wrongCqBytes, DIRECTORY_OFFSET + 28, 1);

    byte[] badCqBits = new SyntheticCactBuilder().addCq(3, 133, 128, 4).build();
    putInt(badCqBits, DIRECTORY_OFFSET + 40, 6);

    assertMalformed(unsupportedType, "type");
    assertMalformed(nonZeroTail, "unused shape");
    assertMalformed(wrongCqBytes, "CQ byte count");
    assertMalformed(badCqBits, "record bits");
  }

  @Test
  void acceptsAttentionProjectionWidthDifferentFromTheResidualWidth() {
    byte[] data = new SyntheticCactBuilder().addFp16(4).build();
    putInt(data, 6 * Integer.BYTES, 12);
    putInt(data, 12 * Integer.BYTES, 16);

    CactHeader header = CactParser.parseSegment(MemorySegment.ofArray(data)).header();

    assertThat(header.modelWidth()).isEqualTo(12);
    assertThat(header.queryHeadCount() * header.headWidth()).isEqualTo(8);
  }

  @Test
  void rejectsAZeroActiveEngramOrder() {
    byte[] data = new SyntheticCactBuilder().addFp16(4).build();
    putInt(data, 20 * Integer.BYTES, 0);

    assertMalformed(data, "engram order");
  }

  @Test
  void parsesPinnedOfficialNeedle2ArtifactWhenProvided() throws IOException {
    String configured = System.getProperty("models.fixtures.needle2Cact", "");
    assumeTrue(!configured.isBlank(), "set -Dmodels.fixtures.needle2Cact=<needle2.cact>");
    Path path = Path.of(configured);
    assumeTrue(Files.isRegularFile(path), "Needle 2 fixture is not installed");
    assertThat(sha256(path)).isEqualTo(NEEDLE2_SHA256);

    try (Arena arena = Arena.ofConfined()) {
      CactFile file = CactParser.parse(path, arena);
      CactHeader header = file.header();
      assertThat(file.fileSegment().byteSize()).isEqualTo(13_737_807);
      assertThat(header.tensorCount()).isEqualTo(405);
      assertThat(header.vocabularySize()).isEqualTo(8192);
      assertThat(header.modelWidth()).isEqualTo(512);
      assertThat(header.layerCount()).isEqualTo(27);
      assertThat(header.queryHeadCount()).isEqualTo(8);
      assertThat(header.kvHeadCount()).isEqualTo(4);
      assertThat(header.headWidth()).isEqualTo(64);
      assertThat(header.maximumSequenceLength()).isEqualTo(2048);
      assertThat(header.kvWindow()).isEqualTo(256);
      assertThat(header.kvBits()).isEqualTo(8);
      assertThat(header.hadamardSize()).isEqualTo(512);
      assertThat(header.mhcLanes()).isEqualTo(4);
      assertThat(header.engramOrders()).containsExactly(2, 3);
      assertThat(header.engramSites()).containsExactly(2, 15);
      assertThat(header.ropeTheta()).isEqualTo(100_000.0f);
      assertThat(file.codebook()).hasSize(28);
      assertThat(file.tensorInfos()).hasSize(405);
      assertThat(file.tensorInfos().stream().filter(t -> t.type() == CactTensorType.FP16))
          .hasSize(259);
      assertThat(file.tensorInfos().stream().filter(t -> t.type() == CactTensorType.CQ))
          .hasSize(145);
      assertThat(file.tensorInfos().stream().filter(t -> t.type() == CactTensorType.RAW))
          .hasSize(1);
      assertThat(
              file.tensorInfos().stream()
                  .filter(t -> t.type() == CactTensorType.CQ && t.recordBits() == 2))
          .hasSize(141);
      assertThat(
              file.tensorInfos().stream()
                  .filter(t -> t.type() == CactTensorType.CQ && t.recordBits() == 4))
          .hasSize(4);

      CactTensorInfo embedding = file.tensorInfos().getFirst();
      assertThat(embedding.shape()).containsExactly(8192, 512);
      assertThat(embedding.byteSize()).isEqualTo(2_162_688);
      assertThat(embedding.groupSize()).isEqualTo(128);
      assertThat(embedding.recordBits()).isEqualTo(4);
      CactTensorInfo tokenizer = file.tensorInfos().getLast();
      assertThat(tokenizer.type()).isEqualTo(CactTensorType.RAW);
      assertThat(tokenizer.shape()).isEmpty();
      assertThat(tokenizer.offset()).isEqualTo(13_622_592);
      assertThat(tokenizer.byteSize()).isEqualTo(115_215);
    }
  }

  private static void assertMalformed(byte[] data, String message) {
    assertThatThrownBy(() -> CactParser.parseSegment(MemorySegment.ofArray(data)))
        .isInstanceOf(MalformedCactException.class)
        .hasMessageContaining(message);
  }

  private static long getLong(byte[] data, int offset) {
    return ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN).getLong(offset);
  }

  private static void putInt(byte[] data, int offset, int value) {
    ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN).putInt(offset, value);
  }

  private static void putLong(byte[] data, int offset, long value) {
    ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN).putLong(offset, value);
  }

  private static String sha256(Path path) throws IOException {
    try {
      MessageDigest digest = MessageDigest.getInstance("SHA-256");
      try (InputStream input = Files.newInputStream(path)) {
        byte[] buffer = new byte[1024 * 1024];
        for (int read; (read = input.read(buffer)) >= 0; ) {
          if (read > 0) {
            digest.update(buffer, 0, read);
          }
        }
      }
      return HexFormat.of().formatHex(digest.digest());
    } catch (NoSuchAlgorithmException exception) {
      throw new IllegalStateException("SHA-256 is unavailable", exception);
    }
  }
}
