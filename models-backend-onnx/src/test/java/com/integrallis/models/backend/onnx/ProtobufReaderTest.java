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
package com.integrallis.models.backend.onnx;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class ProtobufReaderTest {

  @Test
  void readsVarintsIncludingTenByteNegatives() {
    byte[] bytes = new ProtobufWriter().varint(1, 300).varint(2, -1).varint(3, 0).toByteArray();
    ProtobufReader reader = new ProtobufReader(bytes);
    assertThat(ProtobufReader.fieldNumber(reader.readTag())).isEqualTo(1);
    assertThat(reader.readVarint()).isEqualTo(300);
    assertThat(ProtobufReader.fieldNumber(reader.readTag())).isEqualTo(2);
    assertThat(reader.readVarint()).isEqualTo(-1);
    assertThat(ProtobufReader.fieldNumber(reader.readTag())).isEqualTo(3);
    assertThat(reader.readVarint()).isZero();
    assertThat(reader.hasMore()).isFalse();
  }

  @Test
  void readsFixedWidthAndLengthDelimitedFields() {
    byte[] bytes =
        new ProtobufWriter().fixed32(1, 1.5f).fixed64(2, -2.25).string(3, "héllo").toByteArray();
    ProtobufReader reader = new ProtobufReader(bytes);
    reader.readTag();
    assertThat(reader.readFloat()).isEqualTo(1.5f);
    reader.readTag();
    assertThat(reader.readDouble()).isEqualTo(-2.25);
    reader.readTag();
    assertThat(reader.readString()).isEqualTo("héllo");
  }

  @Test
  void skipsEveryWireTypeItUnderstandsAndRejectsGroups() {
    byte[] bytes =
        new ProtobufWriter()
            .varint(1, 7)
            .fixed64(2, 1.0)
            .bytes(3, new byte[] {1, 2, 3})
            .fixed32(4, 2.0f)
            .varint(5, 9)
            .toByteArray();
    ProtobufReader reader = new ProtobufReader(bytes);
    for (int field = 1; field <= 4; field++) {
      int tag = reader.readTag();
      assertThat(ProtobufReader.fieldNumber(tag)).isEqualTo(field);
      reader.skip(ProtobufReader.wireType(tag));
    }
    reader.readTag();
    assertThat(reader.readVarint()).isEqualTo(9);
    assertThatThrownBy(() -> new ProtobufReader(bytes).skip(3))
        .isInstanceOf(IllegalStateException.class);
  }

  @Test
  void decodesPackedAndUnpackedRepeatedScalars() {
    byte[] packed =
        new ProtobufWriter().packedVarints(1, 1, 2, 3).packedFloats(2, 0.5f).toByteArray();
    ProtobufReader reader = new ProtobufReader(packed);
    int tag = reader.readTag();
    assertThat(ProtobufReader.appendVarints(new long[0], reader, ProtobufReader.wireType(tag)))
        .containsExactly(1, 2, 3);
    tag = reader.readTag();
    assertThat(ProtobufReader.appendFloats(new float[] {9f}, reader, ProtobufReader.wireType(tag)))
        .containsExactly(9f, 0.5f);

    byte[] unpacked =
        new ProtobufWriter().varint(1, 4).fixed32(2, 1f).fixed64(3, 2.0).toByteArray();
    reader = new ProtobufReader(unpacked);
    tag = reader.readTag();
    assertThat(ProtobufReader.appendVarints(new long[] {1}, reader, ProtobufReader.wireType(tag)))
        .containsExactly(1, 4);
    tag = reader.readTag();
    assertThat(ProtobufReader.appendFloats(new float[0], reader, ProtobufReader.wireType(tag)))
        .containsExactly(1f);
    tag = reader.readTag();
    assertThat(ProtobufReader.appendDoubles(new double[0], reader, ProtobufReader.wireType(tag)))
        .containsExactly(2.0);
  }

  @Test
  void refusesTruncatedInput() {
    byte[] bytes = new ProtobufWriter().bytes(1, new byte[] {1, 2, 3}).toByteArray();
    byte[] truncated = java.util.Arrays.copyOf(bytes, bytes.length - 1);
    ProtobufReader reader = new ProtobufReader(truncated);
    reader.readTag();
    assertThatThrownBy(reader::readBytes).isInstanceOf(IllegalStateException.class);
    assertThatThrownBy(() -> new ProtobufReader(new byte[] {(byte) 0x80}).readVarint())
        .isInstanceOf(IllegalStateException.class);
    assertThatThrownBy(() -> new ProtobufReader(new byte[2], 1, 5))
        .isInstanceOf(IllegalArgumentException.class);
  }
}
