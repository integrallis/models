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

import com.integrallis.models.backend.purejava.deberta.DebertaV2Projection;
import java.io.IOException;
import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.nio.file.Path;
import java.util.Objects;

/** Owned synchronous F32 projection pool with reusable native staging buffers. */
@SuppressWarnings("restricted")
public final class NativeDenseProjection implements DebertaV2Projection, AutoCloseable {
  private final Arena arena = Arena.ofShared();
  private final MethodHandle project;
  private final MethodHandle destroy;
  private MemorySegment x = MemorySegment.NULL;
  private MemorySegment w = MemorySegment.NULL;
  private MemorySegment y = MemorySegment.NULL;
  private long handle;

  private NativeDenseProjection(Path library, int threads) {
    if (threads < 1 || threads > 256) {
      arena.close();
      throw new IllegalArgumentException("dense projection threads must be between 1 and 256");
    }
    try {
      SymbolLookup symbols = SymbolLookup.libraryLookup(library, arena);
      Linker linker = Linker.nativeLinker();
      MethodHandle abi =
          linker.downcallHandle(
              symbols.find("jmodels_dense_f32_abi").orElseThrow(),
              FunctionDescriptor.of(ValueLayout.JAVA_INT));
      if ((int) abi.invokeExact() != 1)
        throw new IllegalStateException("unsupported dense projection ABI");
      project =
          linker.downcallHandle(
              symbols.find("jmodels_dense_f32_project").orElseThrow(),
              FunctionDescriptor.of(
                  ValueLayout.JAVA_INT,
                  ValueLayout.JAVA_LONG,
                  ValueLayout.ADDRESS,
                  ValueLayout.JAVA_LONG,
                  ValueLayout.ADDRESS,
                  ValueLayout.JAVA_LONG,
                  ValueLayout.ADDRESS,
                  ValueLayout.JAVA_LONG,
                  ValueLayout.JAVA_LONG,
                  ValueLayout.JAVA_LONG,
                  ValueLayout.JAVA_LONG));
      destroy =
          linker.downcallHandle(
              symbols.find("jmodels_dense_f32_destroy").orElseThrow(),
              FunctionDescriptor.ofVoid(ValueLayout.JAVA_LONG));
      MethodHandle create =
          linker.downcallHandle(
              symbols.find("jmodels_dense_f32_create").orElseThrow(),
              FunctionDescriptor.of(ValueLayout.JAVA_LONG, ValueLayout.JAVA_INT));
      handle = (long) create.invokeExact(threads);
      if (handle == 0) throw new IllegalStateException("cannot create dense projection pool");
    } catch (Throwable failure) {
      arena.close();
      throw new IllegalStateException("cannot load dense projection kernel", failure);
    }
  }

  /** Loads from an integrity-checked bundled platform artifact. */
  public static NativeDenseProjection load(int threads) throws IOException {
    return load(BundledNativeKernelLibrary.resolve(), threads);
  }

  /** Loads an explicitly pinned library for controlled experiments. */
  public static NativeDenseProjection load(Path library, int threads) {
    return new NativeDenseProjection(Objects.requireNonNull(library, "library"), threads);
  }

  @Override
  public synchronized void project(
      float[] output, float[] input, int batch, float[] weights, int outputs, int inputs) {
    if (handle == 0) throw new IllegalStateException("dense projection is closed");
    Objects.requireNonNull(output, "output");
    Objects.requireNonNull(input, "input");
    Objects.requireNonNull(weights, "weights");
    if (batch < 1 || outputs < 1 || inputs < 1)
      throw new IllegalArgumentException("positive dimensions required");
    int xn = Math.multiplyExact(batch, inputs);
    int wn = Math.multiplyExact(outputs, inputs);
    int yn = Math.multiplyExact(batch, outputs);
    if (input.length < xn || weights.length != wn || output.length < yn)
      throw new IllegalArgumentException("projection buffer shape mismatch");
    x = reserve(x, xn);
    w = reserve(w, wn);
    y = reserve(y, yn);
    MemorySegment.copy(MemorySegment.ofArray(input), 0, x, 0, (long) xn * Float.BYTES);
    MemorySegment.copy(MemorySegment.ofArray(weights), 0, w, 0, (long) wn * Float.BYTES);
    try {
      int result =
          (int)
              project.invokeExact(
                  handle,
                  x,
                  (long) xn,
                  w,
                  (long) wn,
                  y,
                  (long) yn,
                  (long) batch,
                  (long) inputs,
                  (long) outputs);
      if (result != 0) throw new IllegalStateException("dense projection failed: " + result);
    } catch (Throwable failure) {
      throw new IllegalStateException("dense projection call failed", failure);
    }
    MemorySegment.copy(y, 0, MemorySegment.ofArray(output), 0, (long) yn * Float.BYTES);
  }

  private MemorySegment reserve(MemorySegment segment, int count) {
    long bytes = (long) count * Float.BYTES;
    return segment.byteSize() >= bytes ? segment : arena.allocate(bytes, 64);
  }

  @Override
  public synchronized void close() {
    if (handle == 0) return;
    long previous = handle;
    handle = 0;
    try {
      destroy.invokeExact(previous);
    } catch (Throwable failure) {
      throw new IllegalStateException("cannot destroy dense projection pool", failure);
    } finally {
      arena.close();
    }
  }
}
