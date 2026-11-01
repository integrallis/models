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

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Objects;

/** Owned FFM tokenizer using the pinned Hugging Face Rust implementation. */
@SuppressWarnings("restricted")
public final class NativeHuggingFaceTokenizer implements AutoCloseable {
  private final Arena arena;
  private final MethodHandle encode;
  private final MethodHandle destroy;
  private long handle;

  private NativeHuggingFaceTokenizer(Path library, Path tokenizer) throws IOException {
    arena = Arena.ofShared();
    try {
      SymbolLookup symbols = SymbolLookup.libraryLookup(library, arena);
      Linker linker = Linker.nativeLinker();
      MethodHandle abi =
          linker.downcallHandle(
              symbols.find("jmodels_tokenizer_abi").orElseThrow(),
              FunctionDescriptor.of(ValueLayout.JAVA_INT));
      if ((int) abi.invokeExact() != 1) {
        throw new IllegalStateException("unsupported tokenizer ABI");
      }
      MethodHandle create =
          linker.downcallHandle(
              symbols.find("jmodels_tokenizer_create").orElseThrow(),
              FunctionDescriptor.of(
                  ValueLayout.JAVA_LONG, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG));
      encode =
          linker.downcallHandle(
              symbols.find("jmodels_tokenizer_encode").orElseThrow(),
              FunctionDescriptor.of(
                  ValueLayout.JAVA_LONG,
                  ValueLayout.JAVA_LONG,
                  ValueLayout.ADDRESS,
                  ValueLayout.JAVA_LONG,
                  ValueLayout.ADDRESS,
                  ValueLayout.JAVA_LONG));
      destroy =
          linker.downcallHandle(
              symbols.find("jmodels_tokenizer_destroy").orElseThrow(),
              FunctionDescriptor.ofVoid(ValueLayout.JAVA_LONG));
      byte[] json = Files.readAllBytes(tokenizer);
      try (Arena call = Arena.ofConfined()) {
        MemorySegment bytes = call.allocateFrom(ValueLayout.JAVA_BYTE, json);
        handle = (long) create.invokeExact(bytes, (long) json.length);
      }
      if (handle == 0) {
        throw new IllegalArgumentException("native tokenizer rejected " + tokenizer);
      }
    } catch (Throwable failure) {
      arena.close();
      if (failure instanceof IOException io) {
        throw io;
      }
      throw new IllegalStateException("cannot load Hugging Face tokenizer", failure);
    }
  }

  /** Uses the existing platform-library resolver and integrity checks. */
  public static NativeHuggingFaceTokenizer load(Path tokenizer) throws IOException {
    return new NativeHuggingFaceTokenizer(BundledNativeKernelLibrary.resolve(), tokenizer);
  }

  /** Uses an explicitly supplied native library, for controlled qualification workers. */
  public static NativeHuggingFaceTokenizer load(Path library, Path tokenizer) throws IOException {
    return new NativeHuggingFaceTokenizer(library, tokenizer);
  }

  /** Encodes one publisher schema/text piece, recognizing added tokens without adding BOS/EOS. */
  public synchronized int[] encodePiece(String text) {
    Objects.requireNonNull(text, "text");
    if (handle == 0) {
      throw new IllegalStateException("tokenizer is closed");
    }
    byte[] utf8 = text.getBytes(StandardCharsets.UTF_8);
    try (Arena call = Arena.ofConfined()) {
      MemorySegment input = call.allocate(Math.max(1, utf8.length));
      input.asSlice(0, utf8.length).copyFrom(MemorySegment.ofArray(utf8));
      long capacity = Math.max(16, utf8.length + 16L);
      MemorySegment output = call.allocate(capacity * Integer.BYTES, Integer.BYTES);
      long count = (long) encode.invokeExact(handle, input, (long) utf8.length, output, capacity);
      if (count > capacity) {
        capacity = count;
        output = call.allocate(capacity * Integer.BYTES, Integer.BYTES);
        count = (long) encode.invokeExact(handle, input, (long) utf8.length, output, capacity);
      }
      if (count < 0 || count > capacity) {
        throw new IllegalStateException("native tokenization failed");
      }
      return output.asSlice(0, count * Integer.BYTES).toArray(ValueLayout.JAVA_INT);
    } catch (RuntimeException failure) {
      throw failure;
    } catch (Throwable failure) {
      throw new IllegalStateException("tokenizer FFM call failed", failure);
    }
  }

  @Override
  public synchronized void close() {
    if (handle == 0) {
      return;
    }
    try {
      destroy.invokeExact(handle);
    } catch (Throwable failure) {
      throw new IllegalStateException("tokenizer cleanup failed", failure);
    } finally {
      handle = 0;
      arena.close();
    }
  }
}
