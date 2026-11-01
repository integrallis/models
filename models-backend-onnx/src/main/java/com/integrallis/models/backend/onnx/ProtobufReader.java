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

import java.nio.charset.StandardCharsets;
import java.util.Arrays;

/**
 * Minimal protobuf wire-format reader over a byte range.
 *
 * <p>It understands the five wire types the ONNX schema uses: varint (0), 64-bit (1),
 * length-delimited (2) and 32-bit (5); groups (3, 4) are rejected because ONNX never emits them.
 * Nested messages are read as sub-ranges of the same array, so decoding a 35 MB model copies only
 * the tensor payloads it keeps.
 */
final class ProtobufReader {

  static final int WIRE_VARINT = 0;
  static final int WIRE_FIXED64 = 1;
  static final int WIRE_LENGTH_DELIMITED = 2;
  static final int WIRE_FIXED32 = 5;

  private final byte[] data;
  private final int limit;
  private int position;

  ProtobufReader(byte[] data) {
    this(data, 0, data.length);
  }

  ProtobufReader(byte[] data, int offset, int length) {
    if (offset < 0 || length < 0 || offset + length > data.length) {
      throw new IllegalArgumentException(
          "range [" + offset + ", " + (offset + length) + ") exceeds " + data.length + " bytes");
    }
    this.data = data;
    this.position = offset;
    this.limit = offset + length;
  }

  boolean hasMore() {
    return position < limit;
  }

  int remaining() {
    return limit - position;
  }

  /** Reads a field tag and returns it raw; the field number is {@code tag >>> 3}. */
  int readTag() {
    long tag = readVarint();
    if (tag <= 0 || tag > Integer.MAX_VALUE) {
      throw new IllegalStateException("invalid protobuf tag " + tag + " at " + position);
    }
    return (int) tag;
  }

  static int fieldNumber(int tag) {
    return tag >>> 3;
  }

  static int wireType(int tag) {
    return tag & 0x7;
  }

  long readVarint() {
    long result = 0;
    int shift = 0;
    while (true) {
      if (position >= limit) {
        throw new IllegalStateException("truncated varint at " + position);
      }
      byte b = data[position++];
      result |= (long) (b & 0x7F) << shift;
      if ((b & 0x80) == 0) {
        return result;
      }
      shift += 7;
      if (shift > 63) {
        throw new IllegalStateException("varint longer than 10 bytes at " + position);
      }
    }
  }

  int readFixed32() {
    require(4);
    int value =
        (data[position] & 0xFF)
            | (data[position + 1] & 0xFF) << 8
            | (data[position + 2] & 0xFF) << 16
            | (data[position + 3] & 0xFF) << 24;
    position += 4;
    return value;
  }

  long readFixed64() {
    long low = readFixed32() & 0xFFFFFFFFL;
    long high = readFixed32() & 0xFFFFFFFFL;
    return low | high << 32;
  }

  float readFloat() {
    return Float.intBitsToFloat(readFixed32());
  }

  double readDouble() {
    return Double.longBitsToDouble(readFixed64());
  }

  /** Reads a length-delimited field as a sub-reader over the same backing array. */
  ProtobufReader readMessage() {
    int length = readLength();
    ProtobufReader nested = new ProtobufReader(data, position, length);
    position += length;
    return nested;
  }

  byte[] readBytes() {
    int length = readLength();
    byte[] bytes = Arrays.copyOfRange(data, position, position + length);
    position += length;
    return bytes;
  }

  String readString() {
    return new String(readBytes(), StandardCharsets.UTF_8);
  }

  /** Skips one field of the given wire type; unknown fields are legal and must be tolerated. */
  void skip(int wireType) {
    switch (wireType) {
      case WIRE_VARINT -> readVarint();
      case WIRE_FIXED64 -> {
        require(8);
        position += 8;
      }
      case WIRE_LENGTH_DELIMITED -> {
        int length = readLength();
        position += length;
      }
      case WIRE_FIXED32 -> {
        require(4);
        position += 4;
      }
      default -> throw new IllegalStateException("unsupported protobuf wire type " + wireType);
    }
  }

  /** Decodes a packed or single repeated varint field into the growing list. */
  static long[] appendVarints(long[] existing, ProtobufReader reader, int wireType) {
    if (wireType == WIRE_LENGTH_DELIMITED) {
      ProtobufReader packed = reader.readMessage();
      long[] values = existing;
      int count = existing.length;
      while (packed.hasMore()) {
        if (count == values.length) {
          values = Arrays.copyOf(values, Math.max(8, count * 2));
        }
        values[count++] = packed.readVarint();
      }
      return Arrays.copyOf(values, count);
    }
    long[] values = Arrays.copyOf(existing, existing.length + 1);
    values[existing.length] = reader.readVarint();
    return values;
  }

  /** Decodes a packed or single repeated float field into the growing list. */
  static float[] appendFloats(float[] existing, ProtobufReader reader, int wireType) {
    if (wireType == WIRE_LENGTH_DELIMITED) {
      ProtobufReader packed = reader.readMessage();
      int count = packed.remaining() / 4;
      float[] values = Arrays.copyOf(existing, existing.length + count);
      for (int i = 0; i < count; i++) {
        values[existing.length + i] = packed.readFloat();
      }
      return values;
    }
    float[] values = Arrays.copyOf(existing, existing.length + 1);
    values[existing.length] = reader.readFloat();
    return values;
  }

  /** Decodes a packed or single repeated double field into the growing list. */
  static double[] appendDoubles(double[] existing, ProtobufReader reader, int wireType) {
    if (wireType == WIRE_LENGTH_DELIMITED) {
      ProtobufReader packed = reader.readMessage();
      int count = packed.remaining() / 8;
      double[] values = Arrays.copyOf(existing, existing.length + count);
      for (int i = 0; i < count; i++) {
        values[existing.length + i] = packed.readDouble();
      }
      return values;
    }
    double[] values = Arrays.copyOf(existing, existing.length + 1);
    values[existing.length] = reader.readDouble();
    return values;
  }

  private int readLength() {
    long length = readVarint();
    if (length < 0 || length > remaining()) {
      throw new IllegalStateException(
          "length " + length + " exceeds the " + remaining() + " bytes left at " + position);
    }
    return (int) length;
  }

  private void require(int bytes) {
    if (remaining() < bytes) {
      throw new IllegalStateException("truncated field at " + position);
    }
  }
}
