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
package com.integrallis.models.backend.onnx;

import java.util.List;

/**
 * A graph input, output or annotated intermediate value, decoded from {@code ValueInfoProto}.
 *
 * <p>Symbolic dimensions keep their parameter name and a value of {@code -1}.
 *
 * @param name the value name
 * @param elementType the declared element type
 * @param dims the declared dimensions, {@code -1} where symbolic or unknown
 * @param dimParams the symbolic dimension names, empty strings where numeric
 */
public record OnnxValueInfo(
    String name, OnnxDataType elementType, long[] dims, List<String> dimParams) {

  /** Defensive copies. */
  public OnnxValueInfo {
    dims = dims.clone();
    dimParams = List.copyOf(dimParams);
  }

  /** The declared dimensions as a copy. */
  @Override
  public long[] dims() {
    return dims.clone();
  }
}
