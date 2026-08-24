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

import com.integrallis.models.backend.purejava.safetensors.SafetensorsBundle;
import com.integrallis.models.backend.purejava.safetensors.SafetensorsDtype;
import com.integrallis.models.backend.purejava.safetensors.SafetensorsTensor;
import java.util.List;
import java.util.Objects;

/** Exposes a single-file or sharded Safetensors bundle through {@link TensorSource}. */
public final class SafetensorsTensorSource implements TensorSource {

  private static final String FORMAT = "safetensors";
  private final SafetensorsBundle bundle;

  public SafetensorsTensorSource(SafetensorsBundle bundle) {
    this.bundle = Objects.requireNonNull(bundle, "bundle");
  }

  @Override
  public String format() {
    return FORMAT;
  }

  @Override
  public List<String> tensorNames() {
    return bundle.tensorNames();
  }

  @Override
  public TensorView tensor(String name) {
    SafetensorsTensor tensor = bundle.tensor(name);
    SafetensorsDtype dtype = tensor.info().dtype();
    int divisor = greatestCommonDivisor(dtype.bitWidth(), Byte.SIZE);
    return new TensorView(
        name,
        tensor.info().shape(),
        new TensorStorage(FORMAT, dtype.code(), Byte.SIZE / divisor, dtype.bitWidth() / divisor),
        tensor.data());
  }

  private static int greatestCommonDivisor(int left, int right) {
    while (right != 0) {
      int remainder = left % right;
      left = right;
      right = remainder;
    }
    return left;
  }
}
