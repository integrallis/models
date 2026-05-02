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
package com.integrallis.models.backend.purejava.gguf;

import static org.assertj.core.api.Assertions.assertThat;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class GgufReaderTest {

  private MemorySegment segmentFrom(byte[] data) {
    var arena = Arena.ofConfined();
    MemorySegment segment = arena.allocate(data.length);
    MemorySegment.copy(data, 0, segment, ValueLayout.JAVA_BYTE, 0, data.length);
    return segment;
  }

  @Nested
  static class ReadU32 {

    @Test
    void readsLittleEndianUint32() {
      byte[] data = new byte[8];
      ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN).putInt(0, 42).putInt(4, 0xDEADBEEF);
      var segment = Arena.ofConfined().allocate(8);
      MemorySegment.copy(data, 0, segment, ValueLayout.JAVA_BYTE, 0, 8);

      var cursor = new GgufReader.Cursor(segment, 0);
      assertThat(cursor.readU32()).isEqualTo(42);
      assertThat(cursor.readU32()).isEqualTo(0xDEADBEEF);
    }

    @Test
    void readsAtOffset() {
      byte[] data = new byte[8];
      ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN).putInt(4, 123);
      var segment = Arena.ofConfined().allocate(8);
      MemorySegment.copy(data, 0, segment, ValueLayout.JAVA_BYTE, 0, 8);

      var cursor = new GgufReader.Cursor(segment, 4);
      assertThat(cursor.readU32()).isEqualTo(123);
    }
  }

  @Nested
  static class ReadI32 {

    @Test
    void readsNegativeValue() {
      byte[] data = new byte[4];
      ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN).putInt(0, -7);
      var segment = Arena.ofConfined().allocate(4);
      MemorySegment.copy(data, 0, segment, ValueLayout.JAVA_BYTE, 0, 4);

      var cursor = new GgufReader.Cursor(segment, 0);
      assertThat(cursor.readI32()).isEqualTo(-7);
    }
  }

  @Nested
  static class ReadU64 {

    @Test
    void readsLittleEndianUint64() {
      byte[] data = new byte[8];
      ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN).putLong(0, 1_000_000_000_000L);
      var segment = Arena.ofConfined().allocate(8);
      MemorySegment.copy(data, 0, segment, ValueLayout.JAVA_BYTE, 0, 8);

      var cursor = new GgufReader.Cursor(segment, 0);
      assertThat(cursor.readU64()).isEqualTo(1_000_000_000_000L);
    }
  }

  @Nested
  static class ReadF32 {

    @Test
    void readsFloat() {
      byte[] data = new byte[4];
      ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN).putFloat(0, 3.14f);
      var segment = Arena.ofConfined().allocate(4);
      MemorySegment.copy(data, 0, segment, ValueLayout.JAVA_BYTE, 0, 4);

      var cursor = new GgufReader.Cursor(segment, 0);
      assertThat(cursor.readF32()).isEqualTo(3.14f);
    }
  }

  @Nested
  static class ReadString {

    @Test
    void readsAsciiString() {
      String expected = "hello";
      byte[] strBytes = expected.getBytes(StandardCharsets.UTF_8);
      byte[] data = new byte[8 + strBytes.length];
      ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN).putLong(0, strBytes.length);
      System.arraycopy(strBytes, 0, data, 8, strBytes.length);

      var segment = Arena.ofConfined().allocate(data.length);
      MemorySegment.copy(data, 0, segment, ValueLayout.JAVA_BYTE, 0, data.length);

      var cursor = new GgufReader.Cursor(segment, 0);
      assertThat(cursor.readString()).isEqualTo("hello");
    }

    @Test
    void readsUtf8MultibytString() {
      String expected = "\u00e9\u00e8\u00ea"; // é è ê
      byte[] strBytes = expected.getBytes(StandardCharsets.UTF_8);
      byte[] data = new byte[8 + strBytes.length];
      ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN).putLong(0, strBytes.length);
      System.arraycopy(strBytes, 0, data, 8, strBytes.length);

      var segment = Arena.ofConfined().allocate(data.length);
      MemorySegment.copy(data, 0, segment, ValueLayout.JAVA_BYTE, 0, data.length);

      var cursor = new GgufReader.Cursor(segment, 0);
      assertThat(cursor.readString()).isEqualTo(expected);
    }

    @Test
    void readsEmptyString() {
      byte[] data = new byte[8];
      ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN).putLong(0, 0);

      var segment = Arena.ofConfined().allocate(8);
      MemorySegment.copy(data, 0, segment, ValueLayout.JAVA_BYTE, 0, 8);

      var cursor = new GgufReader.Cursor(segment, 0);
      assertThat(cursor.readString()).isEmpty();
    }
  }

  @Nested
  static class ReadBool {

    @Test
    void readsTrueFromNonZero() {
      byte[] data = new byte[1];
      data[0] = 1;
      var segment = Arena.ofConfined().allocate(1);
      MemorySegment.copy(data, 0, segment, ValueLayout.JAVA_BYTE, 0, 1);

      var cursor = new GgufReader.Cursor(segment, 0);
      assertThat(cursor.readBool()).isTrue();
    }

    @Test
    void readsFalseFromZero() {
      byte[] data = new byte[1];
      var segment = Arena.ofConfined().allocate(1);
      MemorySegment.copy(data, 0, segment, ValueLayout.JAVA_BYTE, 0, 1);

      var cursor = new GgufReader.Cursor(segment, 0);
      assertThat(cursor.readBool()).isFalse();
    }
  }

  @Nested
  static class SequentialReads {

    @Test
    void cursorAdvancesCorrectly() {
      // Layout: u32(42) + f32(1.5) + string("hi") = 4 + 4 + 8 + 2 = 18 bytes
      byte[] data = new byte[18];
      ByteBuffer buf = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
      buf.putInt(0, 42);
      buf.putFloat(4, 1.5f);
      buf.putLong(8, 2L);
      data[16] = 'h';
      data[17] = 'i';

      var segment = Arena.ofConfined().allocate(data.length);
      MemorySegment.copy(data, 0, segment, ValueLayout.JAVA_BYTE, 0, data.length);

      var cursor = new GgufReader.Cursor(segment, 0);
      assertThat(cursor.readU32()).isEqualTo(42);
      assertThat(cursor.readF32()).isEqualTo(1.5f);
      assertThat(cursor.readString()).isEqualTo("hi");
      assertThat(cursor.offset()).isEqualTo(18);
    }
  }
}
