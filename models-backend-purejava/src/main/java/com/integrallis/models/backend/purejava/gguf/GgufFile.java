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

import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.Objects;

/** Represents a fully-parsed GGUF file with zero-copy access to tensor data. */
public record GgufFile(
    GgufHeader header,
    GgufMetadata metadata,
    List<GgufTensorInfo> tensorInfos,
    long tensorDataOffset,
    MemorySegment fileSegment) {

  public GgufFile {
    Objects.requireNonNull(header, "header");
    Objects.requireNonNull(metadata, "metadata");
    tensorInfos = List.copyOf(tensorInfos);
    Objects.requireNonNull(fileSegment, "fileSegment");
  }

  /** Returns the tensor data for a tensor by name. */
  public GgufTensorData getTensor(String name) {
    for (GgufTensorInfo info : tensorInfos) {
      if (info.name().equals(name)) {
        long start = tensorDataOffset + info.offset();
        long size = info.byteSize();
        MemorySegment slice = fileSegment.asSlice(start, size);
        return new GgufTensorData(info, slice);
      }
    }
    throw new IllegalArgumentException("Tensor not found: " + name);
  }
}
