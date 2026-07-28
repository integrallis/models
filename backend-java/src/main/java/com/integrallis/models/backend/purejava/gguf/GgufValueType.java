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

/** GGUF metadata value types as defined in the GGUF specification. */
public enum GgufValueType {
  UINT8(0),
  INT8(1),
  UINT16(2),
  INT16(3),
  UINT32(4),
  INT32(5),
  FLOAT32(6),
  BOOL(7),
  STRING(8),
  ARRAY(9),
  UINT64(10),
  INT64(11),
  FLOAT64(12);

  private static final GgufValueType[] BY_ID;

  static {
    BY_ID = new GgufValueType[values().length];
    for (GgufValueType type : values()) {
      BY_ID[type.id] = type;
    }
  }

  private final int id;

  GgufValueType(int id) {
    this.id = id;
  }

  /** Returns the numeric id for this value type. */
  public int id() {
    return id;
  }

  /** Looks up a value type by its numeric id. */
  public static GgufValueType fromId(int id) {
    if (id < 0 || id >= BY_ID.length) {
      throw new IllegalArgumentException("Unknown GGUF value type id: " + id);
    }
    return BY_ID[id];
  }
}
