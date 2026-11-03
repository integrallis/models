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
package com.integrallis.models.backend.nativekernel;

import com.integrallis.models.backend.onnx.OnnxKernels;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.nio.file.Path;
import java.util.Objects;

/**
 * The ONNX executor's matrix product, sine and activations on the Models-owned Rust kernels: a
 * single-threaded {@code matrixmultiply} sgemm, a double-precision, range-reduced sine, and
 * onnxruntime's MLAS exp, tanh, logistic and erf approximations. Calls are critical downcalls that
 * read and write the Java arrays in place, so nothing is copied per call. The library is the same
 * integrity-checked bundled artifact the other native kernels load.
 */
@SuppressWarnings("restricted")
public final class NativeOnnxKernels implements OnnxKernels {
  private static final Linker.Option CRITICAL = Linker.Option.critical(true);
  private final MethodHandle sgemm;
  private final MethodHandle sin;
  private final MethodHandle erf;
  private final MethodHandle exp;
  private final MethodHandle tanh;
  private final MethodHandle logistic;

  private NativeOnnxKernels(Path library) {
    SymbolLookup symbols = SymbolLookup.libraryLookup(library, Arena.global());
    Linker linker = Linker.nativeLinker();
    try {
      MethodHandle abi =
          linker.downcallHandle(
              symbols.find("jmodels_onnx_abi").orElseThrow(),
              FunctionDescriptor.of(ValueLayout.JAVA_INT));
      if ((int) abi.invokeExact() != 2) {
        throw new IllegalStateException("unsupported ONNX kernel ABI");
      }
    } catch (Throwable failure) {
      throw new IllegalStateException("cannot load the ONNX kernels from " + library, failure);
    }
    sgemm =
        linker.downcallHandle(
            symbols.find("jmodels_onnx_sgemm").orElseThrow(),
            FunctionDescriptor.of(
                ValueLayout.JAVA_INT,
                ValueLayout.JAVA_LONG,
                ValueLayout.JAVA_LONG,
                ValueLayout.JAVA_LONG,
                ValueLayout.ADDRESS,
                ValueLayout.JAVA_LONG,
                ValueLayout.ADDRESS,
                ValueLayout.JAVA_LONG,
                ValueLayout.ADDRESS,
                ValueLayout.JAVA_LONG),
            CRITICAL);
    sin = unary(linker, symbols, "jmodels_onnx_sin_f32");
    erf = unary(linker, symbols, "jmodels_onnx_erf_f32");
    exp = unary(linker, symbols, "jmodels_onnx_exp_f32");
    tanh = unary(linker, symbols, "jmodels_onnx_tanh_f32");
    logistic = unary(linker, symbols, "jmodels_onnx_logistic_f32");
  }

  private static MethodHandle unary(Linker linker, SymbolLookup symbols, String name) {
    return linker.downcallHandle(
        symbols.find(name).orElseThrow(),
        FunctionDescriptor.of(
            ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG),
        CRITICAL);
  }

  /** Loads from the integrity-checked bundled platform artifact. */
  public static NativeOnnxKernels load() throws IOException {
    return load(BundledNativeKernelLibrary.resolve());
  }

  /** Loads an explicitly pinned library for controlled experiments. */
  public static NativeOnnxKernels load(Path library) {
    return new NativeOnnxKernels(Objects.requireNonNull(library, "library"));
  }

  @Override
  public String name() {
    return "rust-ffm";
  }

  @Override
  public void gemm(
      float[] a,
      int aOffset,
      int lda,
      float[] b,
      int bOffset,
      int ldb,
      float[] c,
      int cOffset,
      int ldc,
      int m,
      int k,
      int n) {
    checkExtent(a, aOffset, m, k, lda, "A");
    checkExtent(b, bOffset, k, n, ldb, "B");
    checkExtent(c, cOffset, m, n, ldc, "C");
    int rc;
    try {
      rc =
          (int)
              sgemm.invokeExact(
                  (long) m,
                  (long) k,
                  (long) n,
                  slice(a, aOffset),
                  (long) lda,
                  slice(b, bOffset),
                  (long) ldb,
                  slice(c, cOffset),
                  (long) ldc);
    } catch (Throwable failure) {
      throw new IllegalStateException("native sgemm failed", failure);
    }
    if (rc != 0) {
      throw new IllegalStateException("native sgemm returned " + rc);
    }
  }

  @Override
  public void sin(float[] x, float[] y, int n) {
    applyUnary(sin, "sine", x, y, n);
  }

  @Override
  public void erf(float[] x, float[] y, int n) {
    applyUnary(erf, "erf", x, y, n);
  }

  @Override
  public void exp(float[] x, int xOffset, float[] y, int yOffset, int n) {
    applyUnary(exp, "exp", x, xOffset, y, yOffset, n);
  }

  @Override
  public void tanh(float[] x, int xOffset, float[] y, int yOffset, int n) {
    applyUnary(tanh, "tanh", x, xOffset, y, yOffset, n);
  }

  @Override
  public void logistic(float[] x, int xOffset, float[] y, int yOffset, int n) {
    applyUnary(logistic, "logistic", x, xOffset, y, yOffset, n);
  }

  private static void applyUnary(MethodHandle handle, String what, float[] x, float[] y, int n) {
    applyUnary(handle, what, x, 0, y, 0, n);
  }

  private static void applyUnary(
      MethodHandle handle, String what, float[] x, int xOffset, float[] y, int yOffset, int n) {
    if (n < 0
        || xOffset < 0
        || yOffset < 0
        || (long) xOffset + n > x.length
        || (long) yOffset + n > y.length) {
      throw new IllegalArgumentException(what + " extent " + n + " exceeds its buffers");
    }
    if (n == 0) {
      return;
    }
    int rc;
    try {
      rc = (int) handle.invokeExact(slice(x, xOffset), slice(y, yOffset), (long) n);
    } catch (Throwable failure) {
      throw new IllegalStateException("native " + what + " failed", failure);
    }
    if (rc != 0) {
      throw new IllegalStateException("native " + what + " returned " + rc);
    }
  }

  private static MemorySegment slice(float[] array, int offset) {
    MemorySegment whole = MemorySegment.ofArray(array);
    return offset == 0 ? whole : whole.asSlice((long) offset * Float.BYTES);
  }

  /** The last element a row-major (rows x cols, leading dimension ld) view touches must exist. */
  private static void checkExtent(
      float[] array, int offset, int rows, int cols, int ld, String name) {
    if (rows <= 0 || cols <= 0) {
      return;
    }
    long last = (long) offset + (long) (rows - 1) * ld + cols - 1;
    if (offset < 0 || ld < cols || last >= array.length) {
      throw new IllegalArgumentException(
          "matrix "
              + name
              + " ("
              + rows
              + "x"
              + cols
              + ", ld "
              + ld
              + ", offset "
              + offset
              + ") exceeds its "
              + array.length
              + "-element buffer");
    }
  }
}
