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
package com.integrallis.models.backend.apple;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;

@SuppressWarnings("restricted")
final class FfmAppleFoundationModelsBridge implements AppleFoundationModelsBridge {

  private static final String AVAILABLE_SYMBOL = "jmodels_afm_available";
  private static final String GENERATE_SYMBOL = "jmodels_afm_generate";
  private static final String RESULT_FREE_SYMBOL = "jmodels_afm_result_free";
  private static final long MAX_NATIVE_RESULT_BYTES = 64L * 1024L * 1024L;
  private static final MemoryLayout RESULT_LAYOUT =
      MemoryLayout.structLayout(
          ValueLayout.JAVA_INT.withName("status"),
          ValueLayout.JAVA_INT.withName("value"),
          ValueLayout.JAVA_LONG.withName("length"),
          ValueLayout.ADDRESS.withName("data"));
  private static final long RESULT_STATUS_OFFSET =
      RESULT_LAYOUT.byteOffset(MemoryLayout.PathElement.groupElement("status"));
  private static final long RESULT_VALUE_OFFSET =
      RESULT_LAYOUT.byteOffset(MemoryLayout.PathElement.groupElement("value"));
  private static final long RESULT_LENGTH_OFFSET =
      RESULT_LAYOUT.byteOffset(MemoryLayout.PathElement.groupElement("length"));
  private static final long RESULT_DATA_OFFSET =
      RESULT_LAYOUT.byteOffset(MemoryLayout.PathElement.groupElement("data"));

  private static final Linker LINKER = Linker.nativeLinker();

  private final Arena arena;
  private final MethodHandle availableHandle;
  private final MethodHandle generateHandle;
  private final MethodHandle resultFreeHandle;

  private FfmAppleFoundationModelsBridge(
      Arena arena,
      MethodHandle availableHandle,
      MethodHandle generateHandle,
      MethodHandle resultFreeHandle) {
    this.arena = arena;
    this.availableHandle = availableHandle;
    this.generateHandle = generateHandle;
    this.resultFreeHandle = resultFreeHandle;
  }

  static FfmAppleFoundationModelsBridge open(Path libraryPath) {
    Arena arena = Arena.ofShared();
    try {
      SymbolLookup lookup = SymbolLookup.libraryLookup(libraryPath, arena);
      return new FfmAppleFoundationModelsBridge(
          arena,
          downcall(lookup, AVAILABLE_SYMBOL, FunctionDescriptor.of(ValueLayout.ADDRESS)),
          downcall(
              lookup,
              GENERATE_SYMBOL,
              FunctionDescriptor.of(
                  ValueLayout.ADDRESS,
                  ValueLayout.ADDRESS,
                  ValueLayout.JAVA_LONG,
                  ValueLayout.ADDRESS,
                  ValueLayout.JAVA_LONG,
                  ValueLayout.JAVA_INT)),
          downcall(lookup, RESULT_FREE_SYMBOL, FunctionDescriptor.ofVoid(ValueLayout.ADDRESS)));
    } catch (RuntimeException | LinkageError e) {
      arena.close();
      throw e;
    }
  }

  @Override
  public AppleFoundationModelsAvailability availability() {
    try {
      NativeResult result = consume((MemorySegment) availableHandle.invoke(), "availability");
      if (result.status() != 0) {
        return AppleFoundationModelsAvailability.unavailable(
            detailOrDefault(
                result.text(), "Apple Foundation Models native availability check failed"));
      }
      return new AppleFoundationModelsAvailability(
          true,
          result.value() == 1,
          detailOrDefault(
              result.text(), "Apple Foundation Models returned no availability detail"));
    } catch (Throwable t) {
      throw bridgeFailure("availability", t);
    }
  }

  @Override
  public AppleFoundationModelsResponse generate(AppleFoundationModelsRequest request) {
    try (Arena callArena = Arena.ofConfined()) {
      NativeBytes prompt = nativeBytes(callArena, request.prompt());
      NativeBytes instructions = nativeBytes(callArena, request.instructions());
      NativeResult result =
          consume(
              (MemorySegment)
                  generateHandle.invoke(
                      prompt.segment(),
                      prompt.length(),
                      instructions.segment(),
                      instructions.length(),
                      request.maxOutputTokens()),
              "generate");
      if (result.status() != 0) {
        throw new IllegalStateException(
            detailOrDefault(result.text(), "Apple Foundation Models native generation failed"));
      }
      return new AppleFoundationModelsResponse(result.text());
    } catch (IllegalStateException e) {
      throw e;
    } catch (Throwable t) {
      throw bridgeFailure("generate", t);
    }
  }

  @Override
  public void close() {
    arena.close();
  }

  private static MethodHandle downcall(
      SymbolLookup lookup, String symbolName, FunctionDescriptor descriptor) {
    MemorySegment symbol =
        lookup
            .find(symbolName)
            .orElseThrow(
                () -> new IllegalArgumentException("missing native symbol: " + symbolName));
    return LINKER.downcallHandle(symbol, descriptor);
  }

  private NativeResult consume(MemorySegment resultPointer, String operation) throws Throwable {
    if (resultPointer.equals(MemorySegment.NULL)) {
      throw new IllegalStateException(
          "Apple Foundation Models native bridge returned no result during " + operation);
    }
    try {
      MemorySegment result = resultPointer.reinterpret(RESULT_LAYOUT.byteSize());
      int status = result.get(ValueLayout.JAVA_INT, RESULT_STATUS_OFFSET);
      int value = result.get(ValueLayout.JAVA_INT, RESULT_VALUE_OFFSET);
      long length = result.get(ValueLayout.JAVA_LONG, RESULT_LENGTH_OFFSET);
      if (length < 0 || length > MAX_NATIVE_RESULT_BYTES) {
        throw new IllegalStateException(
            "Apple Foundation Models native bridge returned an invalid result length: " + length);
      }
      MemorySegment data = result.get(ValueLayout.ADDRESS, RESULT_DATA_OFFSET);
      if (length > 0 && data.equals(MemorySegment.NULL)) {
        throw new IllegalStateException(
            "Apple Foundation Models native bridge returned a null result payload");
      }
      byte[] bytes =
          length == 0 ? new byte[0] : data.reinterpret(length).toArray(ValueLayout.JAVA_BYTE);
      return new NativeResult(status, value, new String(bytes, StandardCharsets.UTF_8));
    } finally {
      resultFreeHandle.invoke(resultPointer);
    }
  }

  private static NativeBytes nativeBytes(Arena arena, String value) {
    byte[] bytes = value.getBytes(StandardCharsets.UTF_8);
    if (bytes.length == 0) {
      return new NativeBytes(MemorySegment.NULL, 0);
    }
    MemorySegment segment = arena.allocate(bytes.length, 1);
    segment.copyFrom(MemorySegment.ofArray(bytes));
    return new NativeBytes(segment, bytes.length);
  }

  private static String detailOrDefault(String detail, String fallback) {
    return detail == null || detail.isBlank() ? fallback : detail;
  }

  private static RuntimeException bridgeFailure(String operation, Throwable t) {
    if (t instanceof RuntimeException runtimeException) {
      return runtimeException;
    }
    if (t instanceof Error error) {
      throw error;
    }
    return new IllegalStateException(
        "Apple Foundation Models native bridge failed during " + operation, t);
  }

  private record NativeBytes(MemorySegment segment, long length) {}

  private record NativeResult(int status, int value, String text) {}
}
