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
import java.util.Objects;

/** Provides access to a tensor's raw data as a {@link MemorySegment} slice. */
public record GgufTensorData(GgufTensorInfo info, MemorySegment dataSegment) {

  public GgufTensorData {
    Objects.requireNonNull(info, "info");
    Objects.requireNonNull(dataSegment, "dataSegment");
  }

  /** Returns the tensor name. */
  public String name() {
    return info.name();
  }

  /** Returns the tensor quantization type. */
  public GgufTensorType type() {
    return info.type();
  }

  /** Returns the tensor shape. */
  public long[] shape() {
    return info.shape();
  }
}
