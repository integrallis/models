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
package com.integrallis.models.backend.purejava.gptoss;

import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.gguf.GgufTensorValues;
import com.integrallis.models.backend.purejava.ops.TensorOps;
import com.integrallis.vectors.core.BFloat16Matrix;
import java.lang.foreign.MemorySegment;
import java.util.Objects;

/**
 * One weight matrix, whatever format it is stored in.
 *
 * <p>GPT-OSS ships in two artifacts with different storage. The safetensors release stores its
 * attention matrices as BF16; the GGUF release stores them as K-quants, and the two also differ in
 * how the MXFP4 experts are laid out. The graph only ever multiplies a matrix by a vector and reads
 * a row of the embedding, so those two operations are all this abstracts -- the alternative was a
 * second copy of a 379-line forward pass, which would then have had to be kept in step with the
 * first.
 *
 * <p>{@link BFloat16Matrix} lives in the {@code vectors} project and cannot implement an interface
 * declared here, hence the wrapper rather than a direct implementation.
 */
interface GptOssProjection {

  /** Multiplies this matrix by {@code input}, writing {@code rows()} values into {@code output}. */
  void multiply(float[] input, float[] output);

  int rows();

  int columns();

  /** Reads one row, which is how the token embedding is used. */
  void row(int index, float[] destination);

  /** A BF16 matrix from the safetensors release. */
  static GptOssProjection ofBFloat16(BFloat16Matrix matrix) {
    Objects.requireNonNull(matrix, "matrix");
    return new GptOssProjection() {
      @Override
      public void multiply(float[] input, float[] output) {
        matrix.multiply(input, output);
      }

      @Override
      public int rows() {
        return matrix.rows();
      }

      @Override
      public int columns() {
        return matrix.columns();
      }

      @Override
      public void row(int index, float[] destination) {
        for (int column = 0; column < destination.length; column++) {
          destination[column] = matrix.value(index, column);
        }
      }
    };
  }

  /** A mapped GGUF tensor of any type {@code ggufMatmul} can run. */
  static GptOssProjection ofGguf(MemorySegment data, GgufTensorType type, int rows, int columns) {
    Objects.requireNonNull(data, "data");
    Objects.requireNonNull(type, "type");
    if (rows <= 0 || columns <= 0) {
      throw new IllegalArgumentException("matrix dimensions must be positive");
    }
    return new GptOssProjection() {
      @Override
      public void multiply(float[] input, float[] output) {
        TensorOps.ggufMatmul(output, input, data, type, rows, columns);
      }

      @Override
      public int rows() {
        return rows;
      }

      @Override
      public int columns() {
        return columns;
      }

      @Override
      public void row(int index, float[] destination) {
        GgufTensorValues.dequantizeRow(data, type, index, columns, destination);
      }
    };
  }
}
