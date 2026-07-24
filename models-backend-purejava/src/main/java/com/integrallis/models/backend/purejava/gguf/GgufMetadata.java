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
import java.util.Map;
import java.util.Optional;

/** Typed wrapper around GGUF metadata key-value pairs. */
public record GgufMetadata(Map<String, GgufMetadataValue> entries) {

  public GgufMetadata {
    entries = Map.copyOf(entries);
  }

  /** Returns the raw metadata value for a key, or empty if not present. */
  public Optional<GgufMetadataValue> get(String key) {
    return Optional.ofNullable(entries.get(key));
  }

  /** Returns a string value for the given key. */
  public Optional<String> getString(String key) {
    return get(key)
        .filter(v -> v instanceof GgufMetadataValue.StringValue)
        .map(v -> ((GgufMetadataValue.StringValue) v).value());
  }

  /** Returns a uint32 value as int for the given key. */
  public Optional<Integer> getUint32(String key) {
    return get(key)
        .filter(v -> v instanceof GgufMetadataValue.Uint32Value)
        .map(v -> ((GgufMetadataValue.Uint32Value) v).value());
  }

  /** Returns an int32 value for the given key. */
  public Optional<Integer> getInt32(String key) {
    return get(key)
        .filter(v -> v instanceof GgufMetadataValue.Int32Value)
        .map(v -> ((GgufMetadataValue.Int32Value) v).value());
  }

  /** Returns a float32 value for the given key. */
  public Optional<Float> getFloat32(String key) {
    return get(key)
        .filter(v -> v instanceof GgufMetadataValue.Float32Value)
        .map(v -> ((GgufMetadataValue.Float32Value) v).value());
  }

  /** Returns a boolean value for the given key. */
  public Optional<Boolean> getBool(String key) {
    return get(key)
        .filter(v -> v instanceof GgufMetadataValue.BoolValue)
        .map(v -> ((GgufMetadataValue.BoolValue) v).value());
  }

  /** Returns a uint64 value as long for the given key. */
  public Optional<Long> getUint64(String key) {
    return get(key)
        .filter(v -> v instanceof GgufMetadataValue.Uint64Value)
        .map(v -> ((GgufMetadataValue.Uint64Value) v).value());
  }

  /** Returns the number of elements in an array value without copying its contents. */
  public Optional<Integer> getArraySize(String key) {
    return get(key)
        .filter(v -> v instanceof GgufMetadataValue.ArrayValue)
        .map(v -> ((GgufMetadataValue.ArrayValue) v).elements().size());
  }

  /** Returns a string array value for the given key. */
  public Optional<List<String>> getStringArray(String key) {
    return get(key)
        .filter(v -> v instanceof GgufMetadataValue.ArrayValue)
        .map(v -> (GgufMetadataValue.ArrayValue) v)
        .filter(a -> a.elementType() == GgufValueType.STRING)
        .map(
            a ->
                a.elements().stream()
                    .map(e -> ((GgufMetadataValue.StringValue) e).value())
                    .toList());
  }

  /** Returns a float array value for the given key. */
  public Optional<List<Float>> getFloat32Array(String key) {
    return get(key)
        .filter(v -> v instanceof GgufMetadataValue.ArrayValue)
        .map(v -> (GgufMetadataValue.ArrayValue) v)
        .filter(a -> a.elementType() == GgufValueType.FLOAT32)
        .map(
            a ->
                a.elements().stream()
                    .map(e -> ((GgufMetadataValue.Float32Value) e).value())
                    .toList());
  }

  /** Returns an int32 array value for the given key. */
  public Optional<List<Integer>> getInt32Array(String key) {
    return get(key)
        .filter(v -> v instanceof GgufMetadataValue.ArrayValue)
        .map(v -> (GgufMetadataValue.ArrayValue) v)
        .filter(a -> a.elementType() == GgufValueType.INT32)
        .map(
            a ->
                a.elements().stream()
                    .map(e -> ((GgufMetadataValue.Int32Value) e).value())
                    .toList());
  }
}
