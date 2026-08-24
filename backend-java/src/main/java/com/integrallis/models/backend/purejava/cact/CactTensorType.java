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

/** Serialized tensor storage types in a `.cact` artifact. */
public enum CactTensorType {
  /** Little-endian IEEE 754 binary16 values. */
  FP16(1),

  /** Little-endian IEEE 754 binary32 values. */
  FP32(2),

  /** Grouped, Hadamard-rotated codebook values. */
  CQ(3),

  /** Opaque bytes, currently used for the embedded tokenizer. */
  RAW(4);

  private final byte id;

  CactTensorType(int id) {
    this.id = (byte) id;
  }

  /** Returns the one-byte serialized identifier. */
  public byte id() {
    return id;
  }

  static CactTensorType fromId(int id) {
    for (CactTensorType type : values()) {
      if (Byte.toUnsignedInt(type.id) == id) {
        return type;
      }
    }
    throw new MalformedCactException("tensor type " + id + " is unsupported");
  }
}
