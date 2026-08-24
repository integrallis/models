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

import com.integrallis.models.backend.purejava.gguf.GgufFile;
import com.integrallis.models.backend.purejava.gguf.GgufTensorData;
import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import java.util.List;
import java.util.Objects;

/** Adapts GGUF's fastest-dimension-first tensor descriptors to logical row-major views. */
public final class GgufTensorSource implements TensorSource {

  private static final String FORMAT = "gguf";
  private final GgufFile file;
  private final List<String> tensorNames;

  public GgufTensorSource(GgufFile file) {
    this.file = Objects.requireNonNull(file, "file");
    tensorNames = file.tensorInfos().stream().map(info -> info.name()).toList();
  }

  @Override
  public String format() {
    return FORMAT;
  }

  @Override
  public List<String> tensorNames() {
    return List.copyOf(tensorNames);
  }

  @Override
  public TensorView tensor(String name) {
    GgufTensorData tensor = file.getTensor(name);
    GgufTensorType type = tensor.type();
    return new TensorView(
        name,
        logicalShape(tensor.shape()),
        new TensorStorage(FORMAT, type.name(), type.blockSize(), type.typeSize()),
        tensor.dataSegment());
  }

  private static long[] logicalShape(long[] fastestDimensionFirst) {
    long[] logical = new long[fastestDimensionFirst.length];
    for (int index = 0; index < fastestDimensionFirst.length; index++) {
      logical[index] = fastestDimensionFirst[fastestDimensionFirst.length - index - 1];
    }
    return logical;
  }
}
