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

import java.lang.foreign.MemorySegment;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/** Parsed Safetensors header backed by the original mapped or heap memory. */
public final class SafetensorsFile {

  private final Map<String, String> metadata;
  private final Map<String, SafetensorsTensorInfo> tensors;
  private final MemorySegment data;

  SafetensorsFile(
      Map<String, String> metadata,
      Map<String, SafetensorsTensorInfo> tensors,
      MemorySegment data) {
    this.metadata = Map.copyOf(metadata);
    this.tensors = java.util.Collections.unmodifiableMap(new LinkedHashMap<>(tensors));
    this.data = Objects.requireNonNull(data, "data").asReadOnly();
  }

  /** Free-form string metadata from the reserved {@code __metadata__} header entry. */
  public Map<String, String> metadata() {
    return metadata;
  }

  /** Tensor names in header order. */
  public List<String> tensorNames() {
    return List.copyOf(tensors.keySet());
  }

  /** Returns whether this shard contains the named tensor. */
  public boolean contains(String name) {
    return tensors.containsKey(name);
  }

  /** Returns a zero-copy tensor view. */
  public SafetensorsTensor tensor(String name) {
    SafetensorsTensorInfo info = tensors.get(name);
    if (info == null) {
      throw new IllegalArgumentException("Safetensors tensor not found: " + name);
    }
    return new SafetensorsTensor(info, data.asSlice(info.dataBegin(), info.byteCount()));
  }
}
