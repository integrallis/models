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

import com.integrallis.models.backend.purejava.deberta.DebertaV2Kernels;
import java.io.IOException;
import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.nio.file.Path;
import java.util.Objects;

/**
 * Owned synchronous F32 matrix and disentangled-attention pool with reusable native staging
 * buffers.
 */
@SuppressWarnings("restricted")
public final class NativeDebertaKernels implements DebertaV2Kernels, AutoCloseable {
  private final Arena arena = Arena.ofShared();
  private final MethodHandle project;
  private final MethodHandle attention;
  private final MethodHandle denseAttention;
  private final MethodHandle causalAttention;
  private final MethodHandle destroy;
  private MemorySegment x = MemorySegment.NULL;
  private MemorySegment w = MemorySegment.NULL;
  private MemorySegment y = MemorySegment.NULL;
  private MemorySegment indices = MemorySegment.NULL;
  private long handle;
  private long projectCalls;
  private long attentionCalls;

  private NativeDebertaKernels(Path library, int threads) {
    if (threads < 1 || threads > 256) {
      arena.close();
      throw new IllegalArgumentException("dense projection threads must be between 1 and 256");
    }
    try {
      SymbolLookup symbols = SymbolLookup.libraryLookup(library, arena);
      Linker linker = Linker.nativeLinker();
      MethodHandle abi =
          linker.downcallHandle(
              symbols.find("jmodels_deberta_abi").orElseThrow(),
              FunctionDescriptor.of(ValueLayout.JAVA_INT));
      if ((int) abi.invokeExact() != 1)
        throw new IllegalStateException("unsupported dense projection ABI");
      project =
          linker.downcallHandle(
              symbols.find("jmodels_deberta_project").orElseThrow(),
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
      attention =
          linker.downcallHandle(
              symbols.find("jmodels_deberta_attend").orElseThrow(),
              FunctionDescriptor.of(
                  ValueLayout.JAVA_INT,
                  ValueLayout.JAVA_LONG,
                  ValueLayout.ADDRESS,
                  ValueLayout.JAVA_LONG,
                  ValueLayout.ADDRESS,
                  ValueLayout.JAVA_LONG,
                  ValueLayout.ADDRESS,
                  ValueLayout.JAVA_LONG,
                  ValueLayout.ADDRESS,
                  ValueLayout.JAVA_LONG,
                  ValueLayout.JAVA_LONG,
                  ValueLayout.JAVA_LONG,
                  ValueLayout.JAVA_LONG,
                  ValueLayout.JAVA_LONG));
      denseAttention =
          symbols
              .find("jmodels_dense_attention")
              .map(
                  symbol ->
                      linker.downcallHandle(
                          symbol,
                          FunctionDescriptor.of(
                              ValueLayout.JAVA_INT,
                              ValueLayout.JAVA_LONG,
                              ValueLayout.ADDRESS,
                              ValueLayout.JAVA_LONG,
                              ValueLayout.ADDRESS,
                              ValueLayout.JAVA_LONG,
                              ValueLayout.JAVA_LONG,
                              ValueLayout.JAVA_LONG,
                              ValueLayout.JAVA_LONG)))
              .orElse(null);
      causalAttention =
          symbols
              .find("jmodels_dense_causal_attention")
              .map(
                  symbol ->
                      linker.downcallHandle(
                          symbol,
                          FunctionDescriptor.of(
                              ValueLayout.JAVA_INT,
                              ValueLayout.JAVA_LONG,
                              ValueLayout.ADDRESS,
                              ValueLayout.JAVA_LONG,
                              ValueLayout.ADDRESS,
                              ValueLayout.JAVA_LONG,
                              ValueLayout.ADDRESS,
                              ValueLayout.JAVA_LONG,
                              ValueLayout.ADDRESS,
                              ValueLayout.JAVA_LONG,
                              ValueLayout.JAVA_LONG,
                              ValueLayout.JAVA_LONG,
                              ValueLayout.JAVA_LONG,
                              ValueLayout.JAVA_LONG),
                          Linker.Option.critical(true)))
              .orElse(null);
      destroy =
          linker.downcallHandle(
              symbols.find("jmodels_deberta_destroy").orElseThrow(),
              FunctionDescriptor.ofVoid(ValueLayout.JAVA_LONG));
      MethodHandle create =
          linker.downcallHandle(
              symbols.find("jmodels_deberta_create").orElseThrow(),
              FunctionDescriptor.of(ValueLayout.JAVA_LONG, ValueLayout.JAVA_INT));
      handle = (long) create.invokeExact(threads);
      if (handle == 0) throw new IllegalStateException("cannot create dense projection pool");
    } catch (Throwable failure) {
      arena.close();
      throw new IllegalStateException("cannot load dense projection kernel", failure);
    }
  }

  /** Loads from an integrity-checked bundled platform artifact. */
  public static NativeDebertaKernels load(int threads) throws IOException {
    return load(BundledNativeKernelLibrary.resolve(), threads);
  }

  /** Loads an explicitly pinned library for controlled experiments. */
  public static NativeDebertaKernels load(Path library, int threads) {
    return new NativeDebertaKernels(Objects.requireNonNull(library, "library"), threads);
  }

  @Override
  public String implementation() {
    return "rust-matrixmultiply";
  }

  @Override
  public synchronized Projection prepare(float[] weights, int outputs, int inputs) {
    if (handle == 0) throw new IllegalStateException("DeBERTa kernels are closed");
    Objects.requireNonNull(weights, "weights");
    if (outputs < 1 || inputs < 1)
      throw new IllegalArgumentException("positive dimensions required");
    int count = Math.multiplyExact(outputs, inputs);
    if (weights.length != count)
      throw new IllegalArgumentException("projection weight shape mismatch");
    MemorySegment matrix = arena.allocate((long) count * Float.BYTES, 64);
    MemorySegment.copy(MemorySegment.ofArray(weights), 0, matrix, 0, matrix.byteSize());
    return (output, input, batch) -> projectPrepared(output, input, batch, matrix, outputs, inputs);
  }

  private synchronized void projectPrepared(
      float[] output, float[] input, int batch, MemorySegment matrix, int outputs, int inputs) {
    if (handle == 0) throw new IllegalStateException("DeBERTa kernels are closed");
    Objects.requireNonNull(output, "output");
    Objects.requireNonNull(input, "input");
    if (batch < 1) throw new IllegalArgumentException("positive batch required");
    int xn = Math.multiplyExact(batch, inputs);
    int yn = Math.multiplyExact(batch, outputs);
    if (input.length < xn || output.length < yn)
      throw new IllegalArgumentException("projection buffer shape mismatch");
    x = reserve(x, xn);
    y = reserve(y, yn);
    MemorySegment.copy(MemorySegment.ofArray(input), 0, x, 0, (long) xn * Float.BYTES);
    try {
      int result =
          (int)
              project.invokeExact(
                  handle,
                  x,
                  (long) xn,
                  matrix,
                  matrix.byteSize() / Float.BYTES,
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
    projectCalls++;
  }

  boolean supportsCausalAttention() {
    return causalAttention != null;
  }

  synchronized void causalAttention(
      float[] output,
      float[] query,
      float[] key,
      float[] value,
      int batch,
      int positions,
      int dimension,
      int heads) {
    if (handle == 0) throw new IllegalStateException("Dense kernels are closed");
    if (batch < 1 || positions < batch || dimension < 1 || heads < 1 || dimension % heads != 0)
      throw new IllegalArgumentException("Invalid causal attention shape");
    int qn = Math.multiplyExact(batch, dimension), kn = Math.multiplyExact(positions, dimension);
    if (query.length < qn || output.length < qn || key.length < kn || value.length < kn)
      throw new IllegalArgumentException("Invalid causal attention buffer");
    if (causalAttention == null)
      throw new UnsupportedOperationException("Native causal attention unavailable");
    try {
      int result =
          (int)
              causalAttention.invokeExact(
                  handle,
                  MemorySegment.ofArray(query),
                  (long) query.length,
                  MemorySegment.ofArray(key),
                  (long) key.length,
                  MemorySegment.ofArray(value),
                  (long) value.length,
                  MemorySegment.ofArray(output),
                  (long) output.length,
                  (long) batch,
                  (long) positions,
                  (long) dimension,
                  (long) heads);
      if (result != 0) throw new IllegalStateException("Causal attention failed: " + result);
    } catch (Throwable failure) {
      throw new IllegalStateException("Causal attention call failed", failure);
    }
  }

  @Override
  public synchronized boolean attention(
      float[] output,
      float[] query,
      float[] key,
      float[] value,
      int tokens,
      int dimension,
      int heads) {
    if (handle == 0) throw new IllegalStateException("Dense kernels are closed");
    if (tokens < 1 || dimension < 1 || heads < 1 || dimension % heads != 0)
      throw new IllegalArgumentException("Invalid attention shape");
    int count = Math.multiplyExact(tokens, dimension);
    if (output.length < count || query.length < count || key.length < count || value.length < count)
      throw new IllegalArgumentException("Invalid attention buffer");
    if (denseAttention == null) return false;
    int packed = Math.multiplyExact(count, 3);
    x = reserve(x, packed);
    y = reserve(y, count);
    long bytes = (long) count * Float.BYTES;
    MemorySegment.copy(MemorySegment.ofArray(query), 0, x, 0, bytes);
    MemorySegment.copy(MemorySegment.ofArray(key), 0, x, bytes, bytes);
    MemorySegment.copy(MemorySegment.ofArray(value), 0, x, 2 * bytes, bytes);
    try {
      int result =
          (int)
              denseAttention.invokeExact(
                  handle,
                  x,
                  (long) packed,
                  y,
                  (long) count,
                  (long) tokens,
                  (long) dimension,
                  (long) heads);
      if (result != 0) throw new IllegalStateException("Dense attention failed: " + result);
    } catch (Throwable failure) {
      throw new IllegalStateException("Dense attention call failed", failure);
    }
    MemorySegment.copy(y, 0, MemorySegment.ofArray(output), 0, bytes);
    return true;
  }

  @Override
  public synchronized void attend(
      float[] output,
      float[] queries,
      float[] keys,
      float[] values,
      float[] relativeQueries,
      float[] relativeKeys,
      int[] relativeIndices,
      int sequenceLength,
      int dimension,
      int heads,
      int relativeRows) {
    if (handle == 0) throw new IllegalStateException("DeBERTa kernels are closed");
    if (sequenceLength < 1
        || dimension < 1
        || heads < 1
        || dimension % heads != 0
        || relativeRows < 1) throw new IllegalArgumentException("invalid attention dimensions");
    int count = Math.multiplyExact(sequenceLength, dimension);
    int relativeCount = Math.multiplyExact(relativeRows, dimension);
    int indexCount = Math.multiplyExact(sequenceLength, sequenceLength);
    if (output.length < count
        || queries.length < count
        || keys.length < count
        || values.length < count
        || relativeQueries.length != relativeCount
        || relativeKeys.length != relativeCount
        || relativeIndices.length < indexCount)
      throw new IllegalArgumentException("attention buffer shape mismatch");
    int packedCount = Math.multiplyExact(count, 3);
    int packedRelative = Math.multiplyExact(relativeCount, 2);
    x = reserve(x, packedCount);
    w = reserve(w, packedRelative);
    y = reserve(y, count);
    indices = reserve(indices, indexCount);
    long bytes = (long) count * Float.BYTES;
    MemorySegment.copy(MemorySegment.ofArray(queries), 0, x, 0, bytes);
    MemorySegment.copy(MemorySegment.ofArray(keys), 0, x, bytes, bytes);
    MemorySegment.copy(MemorySegment.ofArray(values), 0, x, 2 * bytes, bytes);
    long relativeBytes = (long) relativeCount * Float.BYTES;
    MemorySegment.copy(MemorySegment.ofArray(relativeQueries), 0, w, 0, relativeBytes);
    MemorySegment.copy(MemorySegment.ofArray(relativeKeys), 0, w, relativeBytes, relativeBytes);
    MemorySegment.copy(
        MemorySegment.ofArray(relativeIndices), 0, indices, 0, (long) indexCount * Integer.BYTES);
    try {
      int result =
          (int)
              attention.invokeExact(
                  handle,
                  x,
                  (long) packedCount,
                  w,
                  (long) packedRelative,
                  indices,
                  (long) indexCount,
                  y,
                  (long) count,
                  (long) sequenceLength,
                  (long) dimension,
                  (long) heads,
                  (long) relativeRows);
      if (result != 0) throw new IllegalStateException("native attention failed: " + result);
    } catch (Throwable failure) {
      throw new IllegalStateException("native attention call failed", failure);
    }
    MemorySegment.copy(y, 0, MemorySegment.ofArray(output), 0, bytes);
    attentionCalls++;
  }

  /** Completed projection calls, for verifying backend dispatch in qualification. */
  public synchronized long projectCalls() {
    return projectCalls;
  }

  /** Completed attention calls, for verifying backend dispatch in qualification. */
  public synchronized long attentionCalls() {
    return attentionCalls;
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
