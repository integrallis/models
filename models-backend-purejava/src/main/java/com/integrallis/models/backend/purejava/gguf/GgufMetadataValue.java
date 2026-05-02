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

import java.util.List;
import java.util.Objects;

/** Sealed hierarchy representing GGUF metadata values. */
public sealed interface GgufMetadataValue
    permits GgufMetadataValue.Uint8Value,
        GgufMetadataValue.Int8Value,
        GgufMetadataValue.Uint16Value,
        GgufMetadataValue.Int16Value,
        GgufMetadataValue.Uint32Value,
        GgufMetadataValue.Int32Value,
        GgufMetadataValue.Float32Value,
        GgufMetadataValue.BoolValue,
        GgufMetadataValue.StringValue,
        GgufMetadataValue.ArrayValue,
        GgufMetadataValue.Uint64Value,
        GgufMetadataValue.Int64Value,
        GgufMetadataValue.Float64Value {

  record Uint8Value(int value) implements GgufMetadataValue {}

  record Int8Value(byte value) implements GgufMetadataValue {}

  record Uint16Value(int value) implements GgufMetadataValue {}

  record Int16Value(short value) implements GgufMetadataValue {}

  record Uint32Value(int value) implements GgufMetadataValue {}

  record Int32Value(int value) implements GgufMetadataValue {}

  record Float32Value(float value) implements GgufMetadataValue {}

  record BoolValue(boolean value) implements GgufMetadataValue {}

  record StringValue(String value) implements GgufMetadataValue {
    public StringValue {
      Objects.requireNonNull(value, "value");
    }
  }

  record ArrayValue(GgufValueType elementType, List<GgufMetadataValue> elements)
      implements GgufMetadataValue {
    public ArrayValue {
      Objects.requireNonNull(elementType, "elementType");
      elements = List.copyOf(elements);
    }
  }

  record Uint64Value(long value) implements GgufMetadataValue {}

  record Int64Value(long value) implements GgufMetadataValue {}

  record Float64Value(double value) implements GgufMetadataValue {}
}
