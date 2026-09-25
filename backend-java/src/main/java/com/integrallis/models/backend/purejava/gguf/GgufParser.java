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
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

/** Parses a GGUF file from disk using memory-mapped I/O for zero-copy tensor access. */
public final class GgufParser {

  private GgufParser() {}

  /** Parses a GGUF file at the given path, using the provided arena for memory management. */
  public static GgufFile parse(Path path, Arena arena) throws IOException {
    try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
      long fileSize = channel.size();
      // A mapped GGUF is 4 KiB pages, and a forward pass touches every weight once, so a multi
      // gigabyte model spends much of its time in page walks. See GgufHugePages for the numbers.
      MemorySegment fileSegment =
          GgufHugePages.isRequested(fileSize)
              ? GgufHugePages.copyInto(channel, fileSize, arena)
              : channel.map(FileChannel.MapMode.READ_ONLY, 0, fileSize, arena);
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
    validateTensorAlignment(tensorInfos, alignment);
    long tensorDataOffset = alignUp(cursor.offset(), alignment);
    if (tensorInfos.isEmpty()) {
      // A vocabulary-only file (llama.cpp's models/ggml-vocab-*.gguf) has no data section and need
      // not be padded after its metadata; llama.cpp only seeks to the aligned start when tensors
      // exist.
      tensorDataOffset = Math.min(tensorDataOffset, segment.byteSize());
    }
    if (tensorDataOffset > segment.byteSize()) {
      throw new MalformedGgufException(
          "aligned tensor data offset "
              + tensorDataOffset
              + " exceeds file size "
              + segment.byteSize());
    }
    validateTensorRanges(tensorInfos, tensorDataOffset, segment.byteSize());

    return new GgufFile(header, metadata, tensorInfos, tensorDataOffset, segment);
  }

  /**
   * Asserts that every tensor starts at a multiple of the file's alignment ({@code
   * general.alignment}, 32 when absent). GGUF writers pad tensor data to that boundary, and a
   * mapped reader that honours it can address the data in place; an offset off the boundary means
   * the tensor table and the data section disagree.
   */
  private static void validateTensorAlignment(List<GgufTensorInfo> tensors, long alignment) {
    for (GgufTensorInfo tensor : tensors) {
      if ((tensor.offset() & (alignment - 1)) != 0) {
        throw new MalformedGgufException(
            "tensor '"
                + tensor.name()
                + "' has data offset "
                + tensor.offset()
                + ", which is not a multiple of the file's alignment "
                + alignment);
      }
    }
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

  /**
   * Asserts that every tensor's data region holds exactly the bytes its type and shape require.
   *
   * <p>The expected length is {@code elements / blockSize * typeSize}. The bytes available to a
   * tensor are those between its start and the start of the next tensor in offset order (or the end
   * of the file for the last one). A region that is short, overlaps the next tensor, or runs past
   * the file is rejected with the tensor name, the expected length, and the available length; it is
   * never skipped or zero-filled.
   */
  private static void validateTensorRanges(
      List<GgufTensorInfo> tensors, long tensorDataOffset, long fileSize) {
    int count = tensors.size();
    long[] starts = new long[count];
    long[] expectedBytes = new long[count];
    Integer[] order = new Integer[count];
    for (int index = 0; index < count; index++) {
      GgufTensorInfo tensor = tensors.get(index);
      try {
        expectedBytes[index] = tensor.byteSize();
      } catch (IllegalArgumentException | ArithmeticException invalidLayout) {
        throw new MalformedGgufException(
            "tensor '" + tensor.name() + "' has an invalid block layout", invalidLayout);
      }
      try {
        starts[index] = Math.addExact(tensorDataOffset, tensor.offset());
      } catch (ArithmeticException overflow) {
        throw new MalformedGgufException(
            "tensor '" + tensor.name() + "' range overflows a 64-bit offset", overflow);
      }
      order[index] = index;
    }
    Arrays.sort(order, Comparator.comparingLong(index -> starts[index]));
    for (int rank = 0; rank < count; rank++) {
      int index = order[rank];
      GgufTensorInfo tensor = tensors.get(index);
      long start = starts[index];
      long expected = expectedBytes[index];
      if (start > fileSize) {
        throw new MalformedGgufException(
            sizeMessage(tensor, expected, 0)
                + ": data starts at "
                + start
                + ", past the end of the file (size "
                + fileSize
                + ")");
      }
      boolean last = rank + 1 == count;
      long limit = last ? fileSize : Math.min(starts[order[rank + 1]], fileSize);
      long available = limit - start;
      if (expected <= available) {
        continue;
      }
      if (last || limit == fileSize && starts[order[rank + 1]] >= fileSize) {
        throw new MalformedGgufException(
            sizeMessage(tensor, expected, available)
                + " before the end of file (data starts at "
                + start
                + ", file size "
                + fileSize
                + ")");
      }
      GgufTensorInfo next = tensors.get(order[rank + 1]);
      throw new MalformedGgufException(
          sizeMessage(tensor, expected, available)
              + ": it overlaps tensor '"
              + next.name()
              + "', which starts at data offset "
              + next.offset());
    }
  }

  private static String sizeMessage(GgufTensorInfo tensor, long expected, long available) {
    return "tensor '"
        + tensor.name()
        + "' ("
        + tensor.type()
        + ", shape "
        + Arrays.toString(tensor.shape())
        + ") expected "
        + expected
        + " bytes, available "
        + available;
  }
}
