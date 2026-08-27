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
package com.integrallis.models.backend.purejava.cact;

import com.integrallis.vectors.core.RotatedCodebookMatrix;
import java.lang.foreign.MemorySegment;
import java.util.Arrays;
import java.util.Objects;

/** Zero-copy executable view of one codebook-quantized `.cact` matrix. */
public final class CactCqMatrix {

  private static final float TERNARY_CENTROID = 1.2240064f;

  private final RotatedCodebookMatrix matrix;

  private CactCqMatrix(RotatedCodebookMatrix matrix) {
    this.matrix = matrix;
  }

  /** Creates an executable matrix from one CQ tensor and the artifact's shared codebook. */
  public static CactCqMatrix from(CactTensorData tensor, float[] sharedCodebook) {
    Objects.requireNonNull(tensor, "tensor");
    Objects.requireNonNull(sharedCodebook, "sharedCodebook");
    CactTensorInfo info = tensor.info();
    if (info.type() != CactTensorType.CQ || info.shape().length != 2) {
      throw new IllegalArgumentException("tensor must be a two-dimensional CACT CQ matrix");
    }
    int rows = Math.toIntExact(info.shape()[0]);
    int columns = Math.toIntExact(info.shape()[1]);
    Encoding encoding = encoding(info.recordBits(), info.groupSize(), sharedCodebook);
    long packedBytes = info.packedCodeBytes();
    MemorySegment packed = tensor.data().asSlice(0, packedBytes);
    MemorySegment norms = tensor.data().asSlice(packedBytes, info.normBytes());
    return new CactCqMatrix(
        RotatedCodebookMatrix.of(
            packed, norms, rows, columns, info.groupSize(), encoding.type(), encoding.codebook()));
  }

  /** Returns the logical output width. */
  public int rows() {
    return matrix.rows();
  }

  /** Returns the logical input width. */
  public int columns() {
    return matrix.columns();
  }

  /** Returns a zero-copy view over a contiguous range of output rows. */
  public CactCqMatrix rowSlice(int fromRow, int rowCount) {
    return new CactCqMatrix(matrix.rowSlice(fromRow, rowCount));
  }

  /** Reconstructs one logical matrix row into caller-owned storage. */
  public void decodeRow(int row, float[] output) {
    matrix.decodeRow(row, output);
  }

  /** Prepares one activation for this and compatible matrices. */
  public RotatedCodebookMatrix.PreparedActivation prepare(float[] input) {
    return matrix.prepare(input);
  }

  /** Multiplies this matrix by a prepared activation. */
  public void multiply(RotatedCodebookMatrix.PreparedActivation activation, float[] output) {
    matrix.multiply(activation, output);
  }

  /** Returns whether an existing prepared activation can be reused for this matrix. */
  public boolean accepts(RotatedCodebookMatrix.PreparedActivation activation) {
    return matrix.accepts(activation);
  }

  private static Encoding encoding(int recordBits, int groupSize, float[] sharedCodebook) {
    return switch (recordBits) {
      case 2 ->
          new Encoding(RotatedCodebookMatrix.Encoding.CQ2, requireCodebook(sharedCodebook, 0, 4));
      case 3 ->
          new Encoding(RotatedCodebookMatrix.Encoding.CQ3, requireCodebook(sharedCodebook, 4, 12));
      case 4 ->
          new Encoding(RotatedCodebookMatrix.Encoding.CQ4, requireCodebook(sharedCodebook, 12, 28));
      case 5 -> {
        float scale = (float) (TERNARY_CENTROID / Math.sqrt(groupSize));
        yield new Encoding(
            RotatedCodebookMatrix.Encoding.TERNARY, new float[] {-scale, 0.0f, scale});
      }
      default ->
          throw new IllegalArgumentException("unsupported CACT CQ record width " + recordBits);
    };
  }

  private static float[] requireCodebook(float[] codebook, int start, int end) {
    if (codebook.length < end) {
      throw new IllegalArgumentException(
          "shared CACT codebook requires at least " + end + " entries; got " + codebook.length);
    }
    return Arrays.copyOfRange(codebook, start, end);
  }

  private record Encoding(RotatedCodebookMatrix.Encoding type, float[] codebook) {}
}
