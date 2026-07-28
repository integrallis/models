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

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;

/** Stateless reader for GGUF little-endian primitives from a {@link MemorySegment}. */
public final class GgufReader {

  private static final ValueLayout.OfInt LE_INT =
      ValueLayout.JAVA_INT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
  private static final ValueLayout.OfLong LE_LONG =
      ValueLayout.JAVA_LONG_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
  private static final ValueLayout.OfFloat LE_FLOAT =
      ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
  private static final ValueLayout.OfShort LE_SHORT =
      ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
  private static final ValueLayout.OfDouble LE_DOUBLE =
      ValueLayout.JAVA_DOUBLE_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

  private GgufReader() {}

  /** Mutable cursor that tracks position while reading from a {@link MemorySegment}. */
  public static final class Cursor {
    private final MemorySegment segment;
    private long offset;

    public Cursor(MemorySegment segment, long offset) {
      this.segment = segment;
      this.offset = offset;
    }

    /** Returns the current byte offset. */
    public long offset() {
      return offset;
    }

    /** Reads a little-endian unsigned 32-bit integer (returned as int, interpret as unsigned). */
    public int readU32() {
      int value = segment.get(LE_INT, offset);
      offset += 4;
      return value;
    }

    /** Reads a little-endian signed 32-bit integer. */
    public int readI32() {
      int value = segment.get(LE_INT, offset);
      offset += 4;
      return value;
    }

    /** Reads a little-endian unsigned 64-bit integer (returned as long). */
    public long readU64() {
      long value = segment.get(LE_LONG, offset);
      offset += 8;
      return value;
    }

    /** Reads a little-endian signed 64-bit integer. */
    public long readI64() {
      long value = segment.get(LE_LONG, offset);
      offset += 8;
      return value;
    }

    /** Reads a little-endian 32-bit float. */
    public float readF32() {
      float value = segment.get(LE_FLOAT, offset);
      offset += 4;
      return value;
    }

    /** Reads a little-endian 64-bit double. */
    public double readF64() {
      double value = segment.get(LE_DOUBLE, offset);
      offset += 8;
      return value;
    }

    /** Reads a little-endian unsigned 16-bit integer (returned as int). */
    public int readU16() {
      int value = Short.toUnsignedInt(segment.get(LE_SHORT, offset));
      offset += 2;
      return value;
    }

    /** Reads a little-endian signed 16-bit integer. */
    public short readI16() {
      short value = segment.get(LE_SHORT, offset);
      offset += 2;
      return value;
    }

    /** Reads a single unsigned byte (returned as int 0-255). */
    public int readU8() {
      int value = Byte.toUnsignedInt(segment.get(ValueLayout.JAVA_BYTE, offset));
      offset += 1;
      return value;
    }

    /** Reads a single signed byte. */
    public byte readI8() {
      byte value = segment.get(ValueLayout.JAVA_BYTE, offset);
      offset += 1;
      return value;
    }

    /** Reads a GGUF string: u64 length prefix followed by UTF-8 bytes. */
    public String readString() {
      long length = readU64();
      if (length == 0) {
        return "";
      }
      byte[] bytes = new byte[(int) length];
      MemorySegment.copy(segment, ValueLayout.JAVA_BYTE, offset, bytes, 0, (int) length);
      offset += length;
      return new String(bytes, StandardCharsets.UTF_8);
    }

    /** Reads a boolean (single byte, non-zero = true). */
    public boolean readBool() {
      byte value = segment.get(ValueLayout.JAVA_BYTE, offset);
      offset += 1;
      return value != 0;
    }
  }
}
