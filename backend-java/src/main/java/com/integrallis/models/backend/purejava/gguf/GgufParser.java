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
package com.integrallis.models.backend.purejava.gguf;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;

/** Parses a GGUF file from disk using memory-mapped I/O for zero-copy tensor access. */
public final class GgufParser {

  private GgufParser() {}

  /** Parses a GGUF file at the given path, using the provided arena for memory management. */
  public static GgufFile parse(Path path, Arena arena) throws IOException {
    try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
      long fileSize = channel.size();
      MemorySegment fileSegment = channel.map(FileChannel.MapMode.READ_ONLY, 0, fileSize, arena);
      return parseSegment(fileSegment);
    }
  }

  /** Parses a GGUF file from an already-mapped memory segment. */
  public static GgufFile parseSegment(MemorySegment segment) {
    GgufHeaderParser.ParseResult headerResult = GgufHeaderParser.parse(segment);
    GgufHeader header = headerResult.header();
    GgufMetadata metadata = headerResult.metadata();

    var cursor = new GgufReader.Cursor(segment, headerResult.endOffset());
    if (header.tensorCount() > cursor.remaining() / 24) {
      throw new MalformedGgufException(
          "tensor count "
              + header.tensorCount()
              + " cannot fit in the remaining "
              + cursor.remaining()
              + " bytes");
    }

    List<GgufTensorInfo> tensorInfos = new ArrayList<>((int) header.tensorCount());
    for (long i = 0; i < header.tensorCount(); i++) {
      String name = cursor.readString();
      int nDimensions = cursor.readU32();
      if (nDimensions < 1 || nDimensions > 4) {
        throw new MalformedGgufException(
            "tensor '"
                + name
                + "' has invalid dimension count "
                + Integer.toUnsignedLong(nDimensions));
      }
      long[] shape = new long[nDimensions];
      for (int d = 0; d < nDimensions; d++) {
        shape[d] = cursor.readU64();
        if (shape[d] <= 0) {
          throw new MalformedGgufException(
              "tensor '" + name + "' has invalid dimension " + Long.toUnsignedString(shape[d]));
        }
      }
      int typeId = cursor.readU32();
      GgufTensorType type;
      try {
        type = GgufTensorType.fromId(typeId);
      } catch (IllegalArgumentException invalidType) {
        throw new MalformedGgufException(
            "tensor '" + name + "' has invalid type " + typeId, invalidType);
      }
      long offset = cursor.readU64();
      if (offset < 0) {
        throw new MalformedGgufException(
            "tensor '" + name + "' has unsupported offset " + Long.toUnsignedString(offset));
      }
      try {
        tensorInfos.add(new GgufTensorInfo(name, nDimensions, shape, type, offset));
      } catch (IllegalArgumentException | ArithmeticException invalidTensor) {
        throw new MalformedGgufException(
            "tensor '" + name + "' has invalid shape or block layout", invalidTensor);
      }
    }

    // Compute aligned tensor data start
    long alignment =
        Integer.toUnsignedLong(
            metadata.getUint32("general.alignment").orElse(GgufConstants.DEFAULT_ALIGNMENT));
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
      throw new MalformedGgufException(
          "general.alignment must be a positive power of two, but was " + alignment);
    }
    long tensorDataOffset = alignUp(cursor.offset(), alignment);
    if (tensorDataOffset > segment.byteSize()) {
      throw new MalformedGgufException(
          "aligned tensor data offset "
              + tensorDataOffset
              + " exceeds file size "
              + segment.byteSize());
    }
    for (GgufTensorInfo tensor : tensorInfos) {
      validateTensorRange(tensor, tensorDataOffset, segment.byteSize());
    }

    return new GgufFile(header, metadata, tensorInfos, tensorDataOffset, segment);
  }

  private static long alignUp(long value, long alignment) {
    long remainder = value & (alignment - 1);
    if (remainder == 0) {
      return value;
    }
    try {
      return Math.addExact(value, alignment - remainder);
    } catch (ArithmeticException overflow) {
      throw new MalformedGgufException("tensor data alignment overflows a 64-bit offset", overflow);
    }
  }

  private static void validateTensorRange(
      GgufTensorInfo tensor, long tensorDataOffset, long fileSize) {
    long tensorBytes;
    try {
      tensorBytes = tensor.byteSize();
    } catch (IllegalArgumentException | ArithmeticException invalidLayout) {
      throw new MalformedGgufException(
          "tensor '" + tensor.name() + "' has an invalid block layout", invalidLayout);
    }
    try {
      long start = Math.addExact(tensorDataOffset, tensor.offset());
      long end = Math.addExact(start, tensorBytes);
      if (end > fileSize) {
        throw new MalformedGgufException(
            "tensor '"
                + tensor.name()
                + "' extends past the file (end "
                + end
                + ", file size "
                + fileSize
                + ")");
      }
    } catch (ArithmeticException overflow) {
      throw new MalformedGgufException(
          "tensor '" + tensor.name() + "' range overflows a 64-bit offset", overflow);
    }
  }
}
