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

import com.integrallis.models.api.ModelPrompt;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class CactTokenizerTest {

  @Test
  void matchesTheReferenceSentencePieceMergesAndByteFallback() {
    CactTokenizer tokenizer = CactTokenizer.parse(testBlob(), 12);

    assertThat(tokenizer.encode("")).containsExactly(2);
    assertThat(tokenizer.encode("hi")).containsExactly(2, 9);
    assertThat(tokenizer.decode(new int[] {9})).isEqualTo("hi");
    assertThat(tokenizer.decode(9)).isEqualTo(" hi");
    assertThat(tokenizer.encode("é")).containsExactly(2, 5, 10, 11);
    assertThat(tokenizer.decode(new int[] {5, 10, 11})).isEqualTo("é");
  }

  @Test
  void recognizesUserDefinedPiecesOnlyInTrustedControlText() {
    CactTokenizer tokenizer = CactTokenizer.parse(testBlob(), 12);
    ModelPrompt prompt = ModelPrompt.builder().control("<|m|>").text("hi").build();

    assertThat(tokenizer.encodeControl("<|m|>")).containsExactly(2, 5, 4);
    assertThat(tokenizer.encode("<|m|>")).doesNotContain(4);
    assertThat(tokenizer.encode(prompt)).containsExactly(2, 5, 4, 8);
  }

  @Test
  void exposesTheSerializedTokenizerContract() {
    CactTokenizer tokenizer = CactTokenizer.parse(testBlob(), 12);

    assertThat(tokenizer.vocabSize()).isEqualTo(12);
    assertThat(tokenizer.padToken()).isZero();
    assertThat(tokenizer.eosToken()).isEqualTo(1);
    assertThat(tokenizer.bosToken()).isEqualTo(2);
    assertThat(tokenizer.unknownToken()).isEqualTo(3);
    assertThat(tokenizer.tokenId("<|m|>")).isEqualTo(4);
    assertThat(tokenizer.addDummyPrefix()).isTrue();
    assertThat(tokenizer.byteFallback()).isTrue();
    assertThat(tokenizer.isEndOfGeneration(1)).isTrue();
  }

  @Test
  void rejectsTruncatedInvalidAndMismatchedTokenizerBlobs() {
    byte[] valid = testBlob().toArray(java.lang.foreign.ValueLayout.JAVA_BYTE);
    byte[] invalidType = valid.clone();
    invalidType[24 + Float.BYTES] = 9;

    assertThatThrownBy(() -> CactTokenizer.parse(MemorySegment.ofArray(new byte[23]), 12))
        .isInstanceOf(MalformedCactException.class)
        .hasMessageContaining("tokenizer header");
    assertThatThrownBy(() -> CactTokenizer.parse(MemorySegment.ofArray(invalidType), 12))
        .isInstanceOf(MalformedCactException.class)
        .hasMessageContaining("piece type");
    assertThatThrownBy(() -> CactTokenizer.parse(testBlob(), 13))
        .isInstanceOf(MalformedCactException.class)
        .hasMessageContaining("vocabulary");
  }

  @Test
  void matchesPinnedOfficialNeedle2ReferenceValuesWhenProvided() throws IOException {
    String configured = System.getProperty("models.fixtures.needle2Cact", "");
    assumeTrue(!configured.isBlank(), "set -Dmodels.fixtures.needle2Cact=<needle2.cact>");
    Path path = Path.of(configured);
    assumeTrue(Files.isRegularFile(path), "Needle 2 fixture is not installed");

    try (Arena arena = Arena.ofConfined()) {
      CactTokenizer tokenizer = CactTokenizer.from(CactParser.parse(path, arena));
      assertThat(tokenizer.vocabSize()).isEqualTo(8192);
      assertThat(tokenizer.padToken()).isZero();
      assertThat(tokenizer.eosToken()).isEqualTo(1);
      assertThat(tokenizer.bosToken()).isEqualTo(2);
      assertThat(tokenizer.unknownToken()).isEqualTo(3);
      assertThat(tokenizer.tokenId("<|im_start|>")).isEqualTo(4);
      assertThat(tokenizer.tokenId("<|im_end|>")).isEqualTo(5);
      assertThat(tokenizer.encode("Name one JVM language."))
          .containsExactly(2, 449, 471, 1285, 815, 8120, 8085, 2083, 8063);
      assertThat(tokenizer.encode("weather in Lagos")).containsExactly(2, 5329, 301, 441, 493, 370);
      assertThat(tokenizer.encode("café 🚇"))
          .containsExactly(2, 280, 1344, 8118, 8042, 254, 173, 168, 149);

      String prompt =
          "<|im_start|>system\n"
              + "You call tools.<|im_end|>\n"
              + "<|im_start|>user\n"
              + "weather in Lagos<|im_end|>\n"
              + "<|im_start|>assistant\n";
      int[] expected = {
        2, 8042, 4, 2204, 24, 8130, 312, 1987, 3582, 8063, 5, 24, 4, 573, 24, 685, 892, 301, 441,
        493, 370, 5, 24, 4, 612, 24
      };
      assertThat(tokenizer.encodeControl(prompt)).containsExactly(expected);
      assertThat(tokenizer.decode(expected)).isEqualTo(prompt);
    }
  }

  @Test
  void matchesPinnedNeedleReferenceForRobotToolPromptWhenProvided() throws IOException {
    String configured = System.getProperty("models.fixtures.needle2Cact", "");
    assumeTrue(!configured.isBlank(), "set -Dmodels.fixtures.needle2Cact=<needle2.cact>");
    Path path = Path.of(configured);
    assumeTrue(Files.isRegularFile(path), "Needle 2 fixture is not installed");
    String prompt =
        "<|im_start|>user\n<tools>[{\"name\":\"move\",\"description\":\"Drive the robot in a direction.\","
            + "\"parameters\":{\"type\":\"object\",\"properties\":{\"direction\":{\"type\":\"string\","
            + "\"enum\":[\"forward\",\"backward\",\"left\",\"right\"]},\"distance_m\":{\"type\":\"number\","
            + "\"description\":\"Distance in meters.\"}},\"required\":[\"direction\",\"distance_m\"]}},"
            + "{\"name\":\"rotate\",\"description\":\"Rotate the robot in place.\",\"parameters\":{\"type\":"
            + "\"object\",\"properties\":{\"direction\":{\"type\":\"string\",\"enum\":[\"left\",\"right\"]},"
            + "\"degrees\":{\"type\":\"number\"}},\"required\":[\"direction\",\"degrees\"]}},{\"name\":"
            + "\"gripper\",\"description\":\"Open or close the gripper.\",\"parameters\":{\"type\":\"object\","
            + "\"properties\":{\"action\":{\"type\":\"string\",\"enum\":[\"open\",\"close\"]}},\"required\":"
            + "[\"action\"]}}]</tools>\nmove forward 2 meters, turn left 90 degrees, then close the gripper"
            + "<|im_end|>\n<|im_start|>assistant\n";
    int[] expected = {
      2, 8042, 4, 573, 24, 8, 1075, 598, 359, 2476, 352, 362, 473, 359, 8096, 320, 352, 306, 4930,
      390, 301, 286, 3132, 555, 801, 433, 491, 359, 1886, 362, 1213, 433, 8053, 2355, 433, 491, 359,
      683, 362, 2064, 1047, 4614, 362, 2629, 1178, 362, 299, 1481, 362, 2939, 3345, 6873, 8097,
      8055, 433, 491, 359, 1888, 362, 473, 359, 8096, 1798, 301, 7872, 1525, 656, 1047, 8053, 2355,
      362, 6873, 8097, 8055, 1875, 598, 359, 295, 8045, 378, 362, 473, 359, 8095, 390, 378, 306,
      4930, 390, 301, 2936, 555, 801, 433, 491, 359, 1886, 362, 1213, 433, 8053, 2355, 433, 491,
      359, 683, 362, 2064, 1047, 299, 1481, 362, 2939, 3345, 560, 8058, 5620, 433, 491, 359, 1888,
      4844, 656, 1047, 8053, 2355, 362, 560, 8058, 5620, 1875, 598, 359, 8058, 320, 6014, 362, 473,
      359, 8114, 6215, 461, 6392, 306, 356, 320, 6014, 555, 801, 433, 491, 359, 1886, 362, 1213,
      433, 1646, 433, 491, 359, 683, 362, 2064, 1047, 5692, 362, 918, 1101, 4770, 656, 1047, 1646,
      3007, 9, 24, 2476, 352, 5737, 466, 7872, 8066, 3392, 5228, 5745, 6249, 8066, 2237, 6392, 306,
      356, 320, 6014, 5, 24, 4, 612, 24
    };
    assertThat(prompt).hasSize(829);
    assertThat(prompt.hashCode()).isEqualTo(-849368063);

    try (Arena arena = Arena.ofConfined()) {
      CactTokenizer tokenizer = CactTokenizer.from(CactParser.parse(path, arena));
      assertThat(tokenizer.encodeControl(prompt)).containsExactly(expected);
    }
  }

  private static MemorySegment testBlob() {
    List<Piece> pieces =
        List.of(
            new Piece("<pad>", 0.0f, 2),
            new Piece("</s>", 0.0f, 2),
            new Piece("<s>", 0.0f, 2),
            new Piece("<unk>", 0.0f, 1),
            new Piece("<|m|>", 0.0f, 3),
            new Piece("▁", -8.0f, 0),
            new Piece("h", -9.0f, 0),
            new Piece("i", -10.0f, 0),
            new Piece("hi", -2.0f, 0),
            new Piece("▁hi", -1.0f, 0),
            new Piece("<0xC3>", 0.0f, 4),
            new Piece("<0xA9>", 0.0f, 4));
    ByteArrayOutputStream records = new ByteArrayOutputStream();
    for (Piece piece : pieces) {
      byte[] surface = piece.text().getBytes(StandardCharsets.UTF_8);
      ByteBuffer record = ByteBuffer.allocate(7).order(ByteOrder.LITTLE_ENDIAN);
      record.putFloat(piece.score()).put((byte) piece.type()).putShort((short) surface.length);
      records.writeBytes(record.array());
      records.writeBytes(surface);
    }
    ByteBuffer header = ByteBuffer.allocate(24).order(ByteOrder.LITTLE_ENDIAN);
    header.putInt(pieces.size());
    header.putInt(0);
    header.putInt(1);
    header.putInt(2);
    header.putInt(3);
    header.put((byte) 1);
    header.put((byte) 1);
    header.putShort((short) 0);
    byte[] blob = new byte[header.capacity() + records.size()];
    System.arraycopy(header.array(), 0, blob, 0, header.capacity());
    System.arraycopy(records.toByteArray(), 0, blob, header.capacity(), records.size());
    return MemorySegment.ofArray(blob);
  }

  private record Piece(String text, float score, int type) {}
}
