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

import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;

/** Test-side protobuf encoder, enough to build ONNX fixtures byte by byte. */
final class ProtobufWriter {

  private final ByteArrayOutputStream out = new ByteArrayOutputStream();

  ProtobufWriter varint(int field, long value) {
    tag(field, ProtobufReader.WIRE_VARINT);
    rawVarint(value);
    return this;
  }

  ProtobufWriter fixed32(int field, float value) {
    tag(field, ProtobufReader.WIRE_FIXED32);
    rawFixed32(Float.floatToIntBits(value));
    return this;
  }

  ProtobufWriter fixed64(int field, double value) {
    tag(field, ProtobufReader.WIRE_FIXED64);
    long bits = Double.doubleToLongBits(value);
    rawFixed32((int) bits);
    rawFixed32((int) (bits >>> 32));
    return this;
  }

  ProtobufWriter bytes(int field, byte[] value) {
    tag(field, ProtobufReader.WIRE_LENGTH_DELIMITED);
    rawVarint(value.length);
    out.write(value, 0, value.length);
    return this;
  }

  ProtobufWriter string(int field, String value) {
    return bytes(field, value.getBytes(StandardCharsets.UTF_8));
  }

  ProtobufWriter message(int field, ProtobufWriter nested) {
    return bytes(field, nested.toByteArray());
  }

  /** Packed repeated varints, the proto3 default encoding. */
  ProtobufWriter packedVarints(int field, long... values) {
    ProtobufWriter packed = new ProtobufWriter();
    for (long value : values) {
      packed.rawVarint(value);
    }
    return bytes(field, packed.toByteArray());
  }

  /** Packed repeated floats. */
  ProtobufWriter packedFloats(int field, float... values) {
    ProtobufWriter packed = new ProtobufWriter();
    for (float value : values) {
      packed.rawFixed32(Float.floatToIntBits(value));
    }
    return bytes(field, packed.toByteArray());
  }

  byte[] toByteArray() {
    return out.toByteArray();
  }

  private void tag(int field, int wire) {
    rawVarint(((long) field << 3) | wire);
  }

  private void rawVarint(long value) {
    long remaining = value;
    while ((remaining & ~0x7FL) != 0) {
      out.write((int) ((remaining & 0x7F) | 0x80));
      remaining >>>= 7;
    }
    out.write((int) remaining);
  }

  private void rawFixed32(int bits) {
    out.write(bits);
    out.write(bits >>> 8);
    out.write(bits >>> 16);
    out.write(bits >>> 24);
  }
}
