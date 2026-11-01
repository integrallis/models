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

/** Element types of the ONNX {@code TensorProto.DataType} enumeration that this module reads. */
public enum OnnxDataType {
  /** Unset or unknown. */
  UNDEFINED(0),
  /** IEEE 754 single precision. */
  FLOAT(1),
  /** Unsigned 8-bit integer. */
  UINT8(2),
  /** Signed 8-bit integer. */
  INT8(3),
  /** Unsigned 16-bit integer. */
  UINT16(4),
  /** Signed 16-bit integer. */
  INT16(5),
  /** Signed 32-bit integer. */
  INT32(6),
  /** Signed 64-bit integer. */
  INT64(7),
  /** UTF-8 strings. */
  STRING(8),
  /** Boolean. */
  BOOL(9),
  /** IEEE 754 half precision. */
  FLOAT16(10),
  /** IEEE 754 double precision. */
  DOUBLE(11),
  /** Unsigned 32-bit integer. */
  UINT32(12),
  /** Unsigned 64-bit integer. */
  UINT64(13),
  /** Brain floating point, 16 bits. */
  BFLOAT16(16);

  private final int code;

  OnnxDataType(int code) {
    this.code = code;
  }

  /** The numeric code the protobuf carries. */
  public int code() {
    return code;
  }

  /** Resolves a protobuf code; unknown codes map to {@link #UNDEFINED}. */
  public static OnnxDataType fromCode(long code) {
    for (OnnxDataType type : values()) {
      if (type.code == code) {
        return type;
      }
    }
    return UNDEFINED;
  }
}
