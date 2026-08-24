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
package com.integrallis.models.backend.purejava.tensor;

import java.util.Objects;

/** Format-qualified tensor type and its packed storage geometry. */
public record TensorStorage(String format, String type, int blockElements, int blockBytes) {

  public TensorStorage {
    if (Objects.requireNonNull(format, "format").isBlank()) {
      throw new IllegalArgumentException("format must not be blank");
    }
    if (Objects.requireNonNull(type, "type").isBlank()) {
      throw new IllegalArgumentException("type must not be blank");
    }
    if (blockElements <= 0 || blockBytes <= 0) {
      throw new IllegalArgumentException("tensor storage block geometry must be positive");
    }
  }
}
