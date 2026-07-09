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
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.nio.file.Path;

@SuppressWarnings("restricted")
final class FfmAppleFoundationModelsBridge implements AppleFoundationModelsBridge {

  private static final String AVAILABLE_SYMBOL = "jmodels_afm_available";
  private static final String GENERATE_SYMBOL = "jmodels_afm_generate";
  private static final String LAST_ERROR_SYMBOL = "jmodels_afm_last_error";
  private static final String FREE_SYMBOL = "jmodels_afm_free";
  private static final long MAX_NATIVE_STRING_BYTES = 64L * 1024L * 1024L;

  private static final Linker LINKER = Linker.nativeLinker();

  private final Arena arena;
  private final MethodHandle availableHandle;
  private final MethodHandle generateHandle;
  private final MethodHandle lastErrorHandle;
  private final MethodHandle freeHandle;

  private FfmAppleFoundationModelsBridge(
      Arena arena,
      MethodHandle availableHandle,
      MethodHandle generateHandle,
      MethodHandle lastErrorHandle,
      MethodHandle freeHandle) {
    this.arena = arena;
    this.availableHandle = availableHandle;
    this.generateHandle = generateHandle;
    this.lastErrorHandle = lastErrorHandle;
    this.freeHandle = freeHandle;
  }

  static FfmAppleFoundationModelsBridge open(Path libraryPath) {
    Arena arena = Arena.ofShared();
    try {
      SymbolLookup lookup = SymbolLookup.libraryLookup(libraryPath, arena);
      return new FfmAppleFoundationModelsBridge(
          arena,
          downcall(lookup, AVAILABLE_SYMBOL, FunctionDescriptor.of(ValueLayout.JAVA_INT)),
          downcall(
              lookup,
              GENERATE_SYMBOL,
              FunctionDescriptor.of(
                  ValueLayout.ADDRESS,
                  ValueLayout.ADDRESS,
                  ValueLayout.ADDRESS,
                  ValueLayout.JAVA_INT)),
          downcall(lookup, LAST_ERROR_SYMBOL, FunctionDescriptor.of(ValueLayout.ADDRESS)),
          downcall(lookup, FREE_SYMBOL, FunctionDescriptor.ofVoid(ValueLayout.ADDRESS)));
    } catch (RuntimeException | LinkageError e) {
      arena.close();
      throw e;
    }
  }

  @Override
  public AppleFoundationModelsAvailability availability() {
    try {
      int available = (int) availableHandle.invoke();
      if (available == 1) {
        return AppleFoundationModelsAvailability.availableNow();
      }
      return AppleFoundationModelsAvailability.unavailable(lastError());
    } catch (Throwable t) {
      throw bridgeFailure("availability", t);
    }
  }

  @Override
  public AppleFoundationModelsResponse generate(AppleFoundationModelsRequest request) {
    try (Arena callArena = Arena.ofConfined()) {
      MemorySegment prompt = callArena.allocateFrom(request.prompt());
      MemorySegment instructions = callArena.allocateFrom(request.instructions());
      MemorySegment result =
          (MemorySegment) generateHandle.invoke(prompt, instructions, request.maxOutputTokens());
      if (result.equals(MemorySegment.NULL)) {
        throw new IllegalStateException(lastError());
      }
      try {
        return new AppleFoundationModelsResponse(nativeString(result));
      } finally {
        freeHandle.invoke(result);
      }
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

  private String lastError() {
    try {
      MemorySegment error = (MemorySegment) lastErrorHandle.invoke();
      if (error.equals(MemorySegment.NULL)) {
        return "Apple Foundation Models native bridge returned no error detail";
      }
      try {
        return nativeString(error);
      } finally {
        freeHandle.invoke(error);
      }
    } catch (Throwable t) {
      throw bridgeFailure("last_error", t);
    }
  }

  private static String nativeString(MemorySegment pointer) {
    return pointer.reinterpret(MAX_NATIVE_STRING_BYTES).getString(0);
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
}
