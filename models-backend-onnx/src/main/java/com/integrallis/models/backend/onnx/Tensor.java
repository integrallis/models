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

import java.util.Arrays;
import java.util.Objects;

/**
 * A dense, row-major tensor with one of three backing stores.
 *
 * <p>Floating-point element types ({@code FLOAT}, {@code FLOAT16}, {@code BFLOAT16}, {@code
 * DOUBLE}) are stored as {@code float[]}; half and brain floats are widened on load and doubles are
 * narrowed, and the recorded {@link #type()} is {@code FLOAT} for all four, because every kernel
 * computes in single precision. Integer types {@code INT64} and {@code INT32} are stored as {@code
 * long[]} and keep their declared type. {@code INT8}, {@code UINT8} and {@code BOOL} are stored as
 * {@code byte[]}.
 *
 * <p>Accessors hand out the backing array itself. This is a runtime value type on the hot path of
 * every operator; kernels must treat another tensor's array as read-only.
 */
public final class Tensor {

  private final OnnxDataType type;
  private final long[] shape;
  private final float[] floats;
  private final double[] doubles;
  private final long[] longs;
  private final byte[] bytes;

  private Tensor(
      OnnxDataType type,
      long[] shape,
      float[] floats,
      double[] doubles,
      long[] longs,
      byte[] bytes) {
    this.type = type;
    this.shape = shape;
    this.floats = floats;
    this.doubles = doubles;
    this.longs = longs;
    this.bytes = bytes;
    long expected = elementCount(shape);
    int actual =
        floats != null
            ? floats.length
            : doubles != null ? doubles.length : longs != null ? longs.length : bytes.length;
    if (expected != actual) {
      throw new IllegalArgumentException(
          "shape " + Arrays.toString(shape) + " holds " + expected + " elements, got " + actual);
    }
  }

  /** A single-precision tensor over the given array, which the tensor keeps by reference. */
  public static Tensor ofFloats(long[] shape, float[] values) {
    return new Tensor(
        OnnxDataType.FLOAT, shape.clone(), Objects.requireNonNull(values), null, null, null);
  }

  /** A double-precision tensor over the given array, kept by reference. */
  public static Tensor ofDoubles(long[] shape, double[] values) {
    return new Tensor(
        OnnxDataType.DOUBLE, shape.clone(), null, Objects.requireNonNull(values), null, null);
  }

  /** A 64-bit integer tensor over the given array, kept by reference. */
  public static Tensor ofLongs(long[] shape, long[] values) {
    return new Tensor(
        OnnxDataType.INT64, shape.clone(), null, null, Objects.requireNonNull(values), null);
  }

  /** A 32-bit integer tensor; values are widened to {@code long} storage. */
  public static Tensor ofInts(long[] shape, long[] values) {
    return new Tensor(
        OnnxDataType.INT32, shape.clone(), null, null, Objects.requireNonNull(values), null);
  }

  /** A byte-backed tensor of type {@code INT8}, {@code UINT8} or {@code BOOL}. */
  public static Tensor ofBytes(OnnxDataType type, long[] shape, byte[] values) {
    if (type != OnnxDataType.INT8 && type != OnnxDataType.UINT8 && type != OnnxDataType.BOOL) {
      throw new IllegalArgumentException("byte storage does not hold " + type);
    }
    return new Tensor(type, shape.clone(), null, null, null, Objects.requireNonNull(values));
  }

  /** A boolean tensor from Java booleans. */
  public static Tensor ofBooleans(long[] shape, boolean[] values) {
    byte[] bytes = new byte[values.length];
    for (int i = 0; i < values.length; i++) {
      bytes[i] = (byte) (values[i] ? 1 : 0);
    }
    return ofBytes(OnnxDataType.BOOL, shape, bytes);
  }

  /** A rank-0 single-precision tensor. */
  public static Tensor scalar(float value) {
    return ofFloats(new long[0], new float[] {value});
  }

  /** A rank-0 64-bit integer tensor. */
  public static Tensor scalar(long value) {
    return ofLongs(new long[0], new long[] {value});
  }

  /** A rank-1 64-bit integer tensor, the shape of shapes. */
  public static Tensor vector(long... values) {
    return ofLongs(new long[] {values.length}, values.clone());
  }

  /** Element type as recorded; see the class comment for how floating types collapse. */
  public OnnxDataType type() {
    return type;
  }

  /** A copy of the dimensions. */
  public long[] shape() {
    return shape.clone();
  }

  /** Dimension {@code axis}, with negative axes counted from the end. */
  public long dim(int axis) {
    return shape[normalizeAxis(axis)];
  }

  /** Number of dimensions. */
  public int rank() {
    return shape.length;
  }

  /** Number of elements. */
  public int size() {
    return (int) elementCount(shape);
  }

  /** Whether the values live in {@code float[]} storage. */
  public boolean isFloat() {
    return floats != null;
  }

  /** Whether the values live in {@code double[]} storage. */
  public boolean isDouble() {
    return doubles != null;
  }

  /** Whether the values are float- or double-backed. */
  public boolean isFloatingPoint() {
    return floats != null || doubles != null;
  }

  /** Whether the tensor is a {@code BOOL}. */
  public boolean isBoolean() {
    return type == OnnxDataType.BOOL;
  }

  /** Whether the values live in {@code long[]} storage. */
  public boolean isIntegral() {
    return longs != null;
  }

  /** Whether the values live in {@code byte[]} storage. */
  public boolean isBytes() {
    return bytes != null;
  }

  /** The backing float array; throws when the tensor is not float-backed. */
  public float[] floats() {
    if (floats == null) {
      throw new IllegalStateException("tensor of type " + type + " is not float-backed");
    }
    return floats;
  }

  /** The backing double array; throws when the tensor is not double-backed. */
  public double[] doubles() {
    if (doubles == null) {
      throw new IllegalStateException("tensor of type " + type + " is not double-backed");
    }
    return doubles;
  }

  /** The backing long array; throws when the tensor is not integer-backed. */
  public long[] longs() {
    if (longs == null) {
      throw new IllegalStateException("tensor of type " + type + " is not integer-backed");
    }
    return longs;
  }

  /** The backing byte array; throws when the tensor is not byte-backed. */
  public byte[] bytes() {
    if (bytes == null) {
      throw new IllegalStateException("tensor of type " + type + " is not byte-backed");
    }
    return bytes;
  }

  /** Element {@code index} widened to float whatever the storage. */
  public float floatAt(int index) {
    if (floats != null) {
      return floats[index];
    }
    if (doubles != null) {
      return (float) doubles[index];
    }
    if (longs != null) {
      return (float) longs[index];
    }
    return type == OnnxDataType.UINT8 ? (bytes[index] & 0xFF) : bytes[index];
  }

  /** Element {@code index} as a double whatever the storage. */
  public double doubleAt(int index) {
    if (doubles != null) {
      return doubles[index];
    }
    if (floats != null) {
      return floats[index];
    }
    if (longs != null) {
      return (double) longs[index];
    }
    return type == OnnxDataType.UINT8 ? (bytes[index] & 0xFF) : bytes[index];
  }

  /** Element {@code index} as a long whatever the storage; floats are truncated toward zero. */
  public long longAt(int index) {
    if (longs != null) {
      return longs[index];
    }
    if (floats != null) {
      return (long) floats[index];
    }
    if (doubles != null) {
      return (long) doubles[index];
    }
    return type == OnnxDataType.UINT8 ? (bytes[index] & 0xFF) : bytes[index];
  }

  /** Element {@code index} as a boolean: non-zero is true. */
  public boolean booleanAt(int index) {
    if (bytes != null) {
      return bytes[index] != 0;
    }
    if (longs != null) {
      return longs[index] != 0;
    }
    return doubleAt(index) != 0.0;
  }

  /** The same values as a {@code FLOAT} tensor; returns {@code this} when already float-backed. */
  public Tensor toFloat() {
    if (floats != null) {
      return this;
    }
    float[] values = new float[size()];
    for (int i = 0; i < values.length; i++) {
      values[i] = floatAt(i);
    }
    return new Tensor(OnnxDataType.FLOAT, shape, values, null, null, null);
  }

  /**
   * The same values as a {@code DOUBLE} tensor; returns {@code this} when already double-backed.
   */
  public Tensor toDouble() {
    if (doubles != null) {
      return this;
    }
    double[] values = new double[size()];
    for (int i = 0; i < values.length; i++) {
      values[i] = doubleAt(i);
    }
    return new Tensor(OnnxDataType.DOUBLE, shape, null, values, null, null);
  }

  /** The same values as an {@code INT64} tensor; returns {@code this} when already so typed. */
  public Tensor toInt64() {
    if (longs != null && type == OnnxDataType.INT64) {
      return this;
    }
    long[] values = new long[size()];
    for (int i = 0; i < values.length; i++) {
      values[i] = longAt(i);
    }
    return new Tensor(OnnxDataType.INT64, shape, null, null, values, null);
  }

  /** The same values as a {@code BOOL} tensor; returns {@code this} when already boolean. */
  public Tensor toBoolean() {
    if (type == OnnxDataType.BOOL) {
      return this;
    }
    byte[] values = new byte[size()];
    for (int i = 0; i < values.length; i++) {
      values[i] = (byte) (booleanAt(i) ? 1 : 0);
    }
    return new Tensor(OnnxDataType.BOOL, shape, null, null, null, values);
  }

  /** This tensor's values under a new shape with the same element count. */
  public Tensor reshape(long[] newShape) {
    return new Tensor(type, newShape.clone(), floats, doubles, longs, bytes);
  }

  /** An empty tensor of this tensor's storage kind and type with the given shape. */
  public Tensor emptyLike(long[] newShape) {
    int count = (int) elementCount(newShape);
    if (floats != null) {
      return new Tensor(type, newShape.clone(), new float[count], null, null, null);
    }
    if (doubles != null) {
      return new Tensor(type, newShape.clone(), null, new double[count], null, null);
    }
    if (longs != null) {
      return new Tensor(type, newShape.clone(), null, null, new long[count], null);
    }
    return new Tensor(type, newShape.clone(), null, null, null, new byte[count]);
  }

  /** Copies element {@code from} of {@code source} into element {@code to} of this tensor. */
  public void copyElement(Tensor source, int from, int to) {
    if (floats != null) {
      floats[to] = source.floats[from];
    } else if (doubles != null) {
      doubles[to] = source.doubles[from];
    } else if (longs != null) {
      longs[to] = source.longs[from];
    } else {
      bytes[to] = source.bytes[from];
    }
  }

  /** Whether {@code other} uses the same storage kind and declared type. */
  public boolean sameKind(Tensor other) {
    return type == other.type
        && (floats != null) == (other.floats != null)
        && (doubles != null) == (other.doubles != null)
        && (longs != null) == (other.longs != null);
  }

  /** Row-major strides in elements. */
  public long[] strides() {
    return strides(shape);
  }

  static long[] strides(long[] shape) {
    long[] strides = new long[shape.length];
    long stride = 1;
    for (int i = shape.length - 1; i >= 0; i--) {
      strides[i] = stride;
      stride *= shape[i];
    }
    return strides;
  }

  static long elementCount(long[] shape) {
    long count = 1;
    for (long dim : shape) {
      if (dim < 0) {
        throw new IllegalArgumentException("negative dimension in " + Arrays.toString(shape));
      }
      count = Math.multiplyExact(count, dim);
    }
    return count;
  }

  int normalizeAxis(int axis) {
    int normalized = axis < 0 ? axis + shape.length : axis;
    if (normalized < 0 || normalized >= shape.length) {
      throw new IllegalArgumentException("axis " + axis + " out of range for rank " + shape.length);
    }
    return normalized;
  }

  @Override
  public String toString() {
    return type + Arrays.toString(shape);
  }
}
