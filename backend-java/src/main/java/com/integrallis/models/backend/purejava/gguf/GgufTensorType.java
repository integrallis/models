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

import java.util.HashMap;
import java.util.Map;

/** GGUF tensor quantization types with their block sizes and byte sizes per block. */
public enum GgufTensorType {
  F32(0, 1, 4),
  F16(1, 1, 2),
  Q4_0(2, 32, 18),
  Q4_1(3, 32, 20),
  Q5_0(6, 32, 22),
  Q5_1(7, 32, 24),
  Q8_0(8, 32, 34),
  Q8_1(9, 32, 36),
  Q2_K(10, 256, 84),
  Q3_K(11, 256, 110),
  Q4_K(12, 256, 144),
  Q5_K(13, 256, 176),
  Q6_K(14, 256, 210),
  Q8_K(15, 256, 292),
  BF16(30, 1, 2),

  /**
   * Upstream ternary, 1.6875 bpw: 48 bytes of base-3 trits (5 per byte), 4 bytes of 4-per-byte
   * trits, one fp16 scale, per 256 weights.
   */
  TQ1_0(34, 256, 54),

  /** Upstream ternary, 2.0625 bpw: 64 bytes at 2 bits per element plus one fp16 scale, per 256. */
  TQ2_0(35, 256, 66),

  /**
   * PrismML-private 2-bit at group 128, 2.125 bpw. Not ternary despite living beside the ternary
   * types: the four codes map to -1, 0, +1 and <b>+2</b>. The fp16 scale is the <b>first</b> field,
   * where upstream's equivalent puts it last.
   */
  PQ2_0(142, 128, 34),

  /**
   * PrismML-private ternary at group 128, 1.75 bpw. The same base-3 trit packing as {@link #TQ1_0}
   * with one fp16 scale per 128 weights instead of per 256, which is what makes it lossless for
   * checkpoints already ternary at group 128 -- a 256-wide scale has to discard one of the two
   * group scales it straddles.
   */
  PTQ1_0(143, 128, 28);

  private static final Map<Integer, GgufTensorType> BY_ID = new HashMap<>();

  static {
    for (GgufTensorType type : values()) {
      BY_ID.put(type.id, type);
    }
  }

  private final int id;
  private final int blockSize;
  private final int typeSize;

  GgufTensorType(int id, int blockSize, int typeSize) {
    this.id = id;
    this.blockSize = blockSize;
    this.typeSize = typeSize;
  }

  /** Returns the numeric id for this tensor type in the GGUF spec. */
  public int id() {
    return id;
  }

  /** Returns the number of elements per quantization block. */
  public int blockSize() {
    return blockSize;
  }

  /** Returns the byte size per block (or per element for unquantized types). */
  public int typeSize() {
    return typeSize;
  }

  /** Looks up a tensor type by its numeric id. */
  public static GgufTensorType fromId(int id) {
    GgufTensorType type = BY_ID.get(id);
    if (type == null) {
      throw new IllegalArgumentException("Unknown GGUF tensor type id: " + id);
    }
    return type;
  }
}
