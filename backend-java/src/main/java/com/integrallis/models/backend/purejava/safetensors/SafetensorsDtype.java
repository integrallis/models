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
package com.integrallis.models.backend.purejava.safetensors;

/** Tensor storage types defined by the Safetensors reference implementation. */
public enum SafetensorsDtype {
  BOOL(8),
  F4(4),
  F6_E2M3(6),
  F6_E3M2(6),
  U8(8),
  I8(8),
  F8_E5M2(8),
  F8_E4M3(8),
  F8_E8M0(8),
  F8_E4M3FNUZ(8),
  F8_E5M2FNUZ(8),
  I16(16),
  U16(16),
  F16(16),
  BF16(16),
  I32(32),
  U32(32),
  F32(32),
  C64(64),
  F64(64),
  I64(64),
  U64(64);

  private final int bitWidth;

  SafetensorsDtype(int bitWidth) {
    this.bitWidth = bitWidth;
  }

  /** Header code used by the format. */
  public String code() {
    return name();
  }

  /** Number of stored bits per tensor element. */
  public int bitWidth() {
    return bitWidth;
  }

  static SafetensorsDtype fromCode(String code) {
    try {
      return valueOf(code);
    } catch (IllegalArgumentException unsupported) {
      throw new MalformedSafetensorsException("unsupported tensor dtype: " + code, unsupported);
    }
  }
}
