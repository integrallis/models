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

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;

/** Strict, memory-mapped parser for the little-endian `.cact` artifact format. */
public final class CactParser {

  private static final int MAGIC = 0x05E12A83;
  private static final int HEADER_BYTES = 120;
  private static final int DIRECTORY_RECORD_BYTES = 44;
  private static final int ALIGNMENT = 64;
  private static final int CODEBOOK_LENGTH = 28;
  private static final ValueLayout.OfInt LE_INT =
      ValueLayout.JAVA_INT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
  private static final ValueLayout.OfLong LE_LONG =
      ValueLayout.JAVA_LONG_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
  private static final ValueLayout.OfFloat LE_FLOAT =
      ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

  private CactParser() {}

  /** Maps and parses a `.cact` file in the supplied arena. */
  public static CactFile parse(Path path, Arena arena) throws IOException {
    try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
      MemorySegment segment = channel.map(FileChannel.MapMode.READ_ONLY, 0, channel.size(), arena);
      return parseSegment(segment);
    }
  }

  /** Parses an already-mapped or heap-backed `.cact` segment. */
  public static CactFile parseSegment(MemorySegment segment) {
    if (segment.byteSize() < HEADER_BYTES) {
      throw new MalformedCactException(
          "header requires " + HEADER_BYTES + " bytes; got " + segment.byteSize());
    }
    try {
      Cursor cursor = new Cursor(segment);
      int magic = cursor.readInt();
      if (magic != MAGIC) {
        throw new MalformedCactException(
            "magic must be 0x"
                + Integer.toHexString(MAGIC)
                + "; got 0x"
                + Integer.toHexString(magic));
      }

      int tensorCount = positiveInt(cursor.readU32(), "tensor count");
      int codebookLength = positiveInt(cursor.readU32(), "codebook length");
      if (codebookLength != CODEBOOK_LENGTH) {
        throw new MalformedCactException(
            "codebook length must be " + CODEBOOK_LENGTH + "; got " + codebookLength);
      }
      int kvWindow = nonNegativeInt(cursor.readU32(), "KV window");
      int kvBits = positiveInt(cursor.readU32(), "KV bits");
      if (kvBits != 2 && kvBits != 3 && kvBits != 4 && kvBits != 8) {
        throw new MalformedCactException("KV bits must be 2, 3, 4, or 8; got " + kvBits);
      }
      int vocabularySize = positiveInt(cursor.readU32(), "vocabulary size");
      int modelWidth = positiveInt(cursor.readU32(), "model width");
      int queryHeadCount = positiveInt(cursor.readU32(), "query head count");
      int kvHeadCount = positiveInt(cursor.readU32(), "KV head count");
      int layerCount = positiveInt(cursor.readU32(), "layer count");
      int headWidth = positiveInt(cursor.readU32(), "head width");
      int maximumSequenceLength = positiveInt(cursor.readU32(), "maximum sequence length");
      int hadamardSize = positiveInt(cursor.readU32(), "Hadamard size");
      int mhcLanes = positiveInt(cursor.readU32(), "mHC lanes");
      int engramSlots = positiveInt(cursor.readU32(), "engram slots");
      int engramSubDimension = positiveInt(cursor.readU32(), "engram sub-dimension");
      int engramTableCount = positiveInt(cursor.readU32(), "engram table count");
      int engramConvolutionTaps = positiveInt(cursor.readU32(), "engram convolution taps");
      int engramConvolutionDilation = positiveInt(cursor.readU32(), "engram convolution dilation");
      int orderCount = boundedCount(cursor.readU32(), "engram order count");
      int[] orderSlots = cursor.readFourU32("engram order");
      int siteCount = boundedCount(cursor.readU32(), "engram site count");
      int[] siteSlots = cursor.readFourU32("engram site");
      float ropeTheta = cursor.readFloat();

      validateGeometry(
          modelWidth,
          queryHeadCount,
          kvHeadCount,
          layerCount,
          headWidth,
          maximumSequenceLength,
          hadamardSize,
          orderCount,
          orderSlots,
          siteCount,
          siteSlots,
          ropeTheta);
      List<Integer> orders = activeSlots(orderSlots, orderCount, "engram order");
      List<Integer> sites = activeSlots(siteSlots, siteCount, "engram site");
      CactHeader header =
          new CactHeader(
              tensorCount,
              codebookLength,
              kvWindow,
              kvBits,
              vocabularySize,
              modelWidth,
              queryHeadCount,
              kvHeadCount,
              layerCount,
              headWidth,
              maximumSequenceLength,
              hadamardSize,
              mhcLanes,
              engramSlots,
              engramSubDimension,
              engramTableCount,
              engramConvolutionTaps,
              engramConvolutionDilation,
              orders,
              sites,
              ropeTheta);

      float[] codebook = readCodebook(cursor, codebookLength);
      long directoryBytes = Math.multiplyExact((long) tensorCount, DIRECTORY_RECORD_BYTES);
      long directoryEnd = Math.addExact(cursor.offset, directoryBytes);
      if (directoryEnd > segment.byteSize()) {
        throw new MalformedCactException(
            "directory ends at " + directoryEnd + " beyond file size " + segment.byteSize());
      }

      List<CactTensorInfo> tensors = new ArrayList<>(tensorCount);
      long minimumDataOffset = align(directoryEnd);
      long previousEnd = minimumDataOffset;
      for (int index = 0; index < tensorCount; index++) {
        CactTensorInfo tensor = readTensor(cursor, index);
        validateTensor(tensor, segment.byteSize(), minimumDataOffset, previousEnd);
        tensors.add(tensor);
        previousEnd = Math.addExact(tensor.offset(), tensor.byteSize());
      }
      return new CactFile(header, codebook, tensors, segment);
    } catch (MalformedCactException malformed) {
      throw malformed;
    } catch (ArithmeticException | IndexOutOfBoundsException malformed) {
      throw new MalformedCactException("offset, range, or size overflow", malformed);
    }
  }

  private static CactTensorInfo readTensor(Cursor cursor, int index) {
    CactTensorType type = CactTensorType.fromId(cursor.readU8());
    int dimensions = cursor.readU8();
    if (dimensions < 0 || dimensions > 4) {
      throw new MalformedCactException(
          "tensor " + index + " dimension count must be from 0 to 4; got " + dimensions);
    }
    int padding = cursor.readU16();
    if (padding != 0) {
      throw new MalformedCactException("tensor " + index + " directory padding must be zero");
    }
    long[] shape = new long[dimensions];
    for (int dimension = 0; dimension < 4; dimension++) {
      long value = cursor.readU32();
      if (dimension < dimensions) {
        if (value == 0) {
          throw new MalformedCactException(
              "tensor " + index + " shape dimensions must be positive");
        }
        shape[dimension] = value;
      } else if (value != 0) {
        throw new MalformedCactException(
            "tensor " + index + " has non-zero unused shape slot " + dimension);
      }
    }
    long offset = cursor.readNonNegativeLong("tensor " + index + " offset");
    long bytes = cursor.readNonNegativeLong("tensor " + index + " byte count");
    int groupSize = nonNegativeInt(cursor.readU32(), "tensor " + index + " group size");
    int recordBits = nonNegativeInt(cursor.readU32(), "tensor " + index + " record bits");
    return new CactTensorInfo(index, type, shape, offset, bytes, groupSize, recordBits);
  }

  private static void validateTensor(
      CactTensorInfo tensor, long fileSize, long minimumDataOffset, long previousEnd) {
    if ((tensor.offset() & (ALIGNMENT - 1)) != 0) {
      throw new MalformedCactException(
          "tensor " + tensor.index() + " offset must be 64-byte aligned");
    }
    if (tensor.offset() < minimumDataOffset) {
      throw new MalformedCactException(
          "tensor " + tensor.index() + " overlaps the header or directory");
    }
    if (tensor.offset() < previousEnd) {
      throw new MalformedCactException("tensor " + tensor.index() + " overlaps its predecessor");
    }
    long end;
    try {
      end = Math.addExact(tensor.offset(), tensor.byteSize());
    } catch (ArithmeticException overflow) {
      throw new MalformedCactException("tensor " + tensor.index() + " range overflows", overflow);
    }
    if (tensor.byteSize() <= 0 || end > fileSize) {
      throw new MalformedCactException(
          "tensor " + tensor.index() + " range ends at " + end + " beyond file size " + fileSize);
    }

    long expectedBytes;
    try {
      expectedBytes = expectedTensorBytes(tensor);
    } catch (ArithmeticException invalid) {
      throw new MalformedCactException(
          "tensor " + tensor.index() + " shape or byte count overflows", invalid);
    }
    if (tensor.byteSize() != expectedBytes) {
      String layout = tensor.type() == CactTensorType.CQ ? "CQ byte count" : "byte count";
      throw new MalformedCactException(
          "tensor "
              + tensor.index()
              + " "
              + layout
              + " must be "
              + expectedBytes
              + "; got "
              + tensor.byteSize());
    }
  }

  private static long expectedTensorBytes(CactTensorInfo tensor) {
    return switch (tensor.type()) {
      case FP16 -> {
        requireDenseMetadata(tensor);
        yield Math.multiplyExact(tensor.elementCount(), Short.BYTES);
      }
      case FP32 -> {
        requireDenseMetadata(tensor);
        yield Math.multiplyExact(tensor.elementCount(), Float.BYTES);
      }
      case CQ -> {
        if (tensor.shape().length != 2) {
          throw new MalformedCactException(
              "tensor " + tensor.index() + " CQ storage requires two dimensions");
        }
        if (tensor.groupSize() < 8 || Integer.bitCount(tensor.groupSize()) != 1) {
          throw new MalformedCactException(
              "tensor " + tensor.index() + " CQ group size must be a power of two");
        }
        int bits = tensor.recordBits();
        if (bits != 2 && bits != 3 && bits != 4 && bits != 5) {
          throw new MalformedCactException(
              "tensor " + tensor.index() + " CQ record bits must be 2, 3, 4, or 5");
        }
        yield Math.addExact(tensor.packedCodeBytes(), tensor.normBytes());
      }
      case RAW -> {
        if (tensor.shape().length != 0 || tensor.groupSize() != 0 || tensor.recordBits() != 0) {
          throw new MalformedCactException(
              "tensor " + tensor.index() + " RAW metadata must have no shape, group, or bits");
        }
        yield tensor.byteSize();
      }
    };
  }

  private static void requireDenseMetadata(CactTensorInfo tensor) {
    if (tensor.shape().length == 0 || tensor.groupSize() != 0 || tensor.recordBits() != 0) {
      throw new MalformedCactException(
          "tensor " + tensor.index() + " dense metadata has invalid shape, group, or bits");
    }
  }

  private static float[] readCodebook(Cursor cursor, int length) {
    float[] codebook = new float[length];
    for (int index = 0; index < length; index++) {
      float value = cursor.readFloat();
      if (!Float.isFinite(value)) {
        throw new MalformedCactException("codebook value " + index + " must be finite");
      }
      codebook[index] = value;
    }
    int[] starts = {0, 4, 12};
    int[] sizes = {4, 8, 16};
    for (int section = 0; section < starts.length; section++) {
      for (int index = starts[section] + 1; index < starts[section] + sizes[section]; index++) {
        if (codebook[index] <= codebook[index - 1]) {
          throw new MalformedCactException("codebook sections must be strictly increasing");
        }
      }
    }
    return codebook;
  }

  private static void validateGeometry(
      int modelWidth,
      int queryHeadCount,
      int kvHeadCount,
      int layerCount,
      int headWidth,
      int maximumSequenceLength,
      int hadamardSize,
      int orderCount,
      int[] orderSlots,
      int siteCount,
      int[] siteSlots,
      float ropeTheta) {
    if (queryHeadCount < kvHeadCount || queryHeadCount % kvHeadCount != 0) {
      throw new MalformedCactException("query head count must be divisible by KV head count");
    }
    if (Integer.bitCount(hadamardSize) != 1 || hadamardSize < modelWidth) {
      throw new MalformedCactException("Hadamard size must be a power of two covering model width");
    }
    if (!Float.isFinite(ropeTheta) || ropeTheta <= 0.0f) {
      throw new MalformedCactException("RoPE theta must be finite and positive");
    }
    validateSlots(orderSlots, orderCount, "engram order", 1, maximumSequenceLength);
    validateSlots(siteSlots, siteCount, "engram site", 0, layerCount);
  }

  private static void validateSlots(
      int[] slots, int count, String name, int minimumInclusive, int upperExclusive) {
    for (int index = 0; index < slots.length; index++) {
      int value = slots[index];
      if (index < count) {
        if (value < minimumInclusive || value >= upperExclusive) {
          throw new MalformedCactException(name + " " + value + " is out of range");
        }
      } else if (value != 0) {
        throw new MalformedCactException(name + " has non-zero unused slot " + index);
      }
    }
  }

  private static List<Integer> activeSlots(int[] slots, int count, String name) {
    List<Integer> result = new ArrayList<>(count);
    for (int index = 0; index < count; index++) {
      int value = slots[index];
      if (result.contains(value)) {
        throw new MalformedCactException(name + " entries must be unique");
      }
      result.add(value);
    }
    return result;
  }

  private static int boundedCount(long value, String name) {
    int count = nonNegativeInt(value, name);
    if (count > 4) {
      throw new MalformedCactException(name + " must be at most 4; got " + count);
    }
    return count;
  }

  private static int positiveInt(long value, String name) {
    int converted = nonNegativeInt(value, name);
    if (converted == 0) {
      throw new MalformedCactException(name + " must be positive");
    }
    return converted;
  }

  private static int nonNegativeInt(long value, String name) {
    if (value > Integer.MAX_VALUE) {
      throw new MalformedCactException(name + " exceeds the supported Java range: " + value);
    }
    return (int) value;
  }

  private static long align(long value) {
    return Math.addExact(value, ALIGNMENT - 1) & -ALIGNMENT;
  }

  private static final class Cursor {
    private final MemorySegment segment;
    private long offset;

    private Cursor(MemorySegment segment) {
      this.segment = segment;
    }

    private int readU8() {
      int value = Byte.toUnsignedInt(segment.get(ValueLayout.JAVA_BYTE, offset));
      offset++;
      return value;
    }

    private int readU16() {
      int low = readU8();
      return low | (readU8() << Byte.SIZE);
    }

    private int readInt() {
      int value = segment.get(LE_INT, offset);
      offset += Integer.BYTES;
      return value;
    }

    private long readU32() {
      return Integer.toUnsignedLong(readInt());
    }

    private long readNonNegativeLong(String name) {
      long value = segment.get(LE_LONG, offset);
      offset += Long.BYTES;
      if (value < 0) {
        throw new MalformedCactException(name + " exceeds signed 64-bit range");
      }
      return value;
    }

    private float readFloat() {
      float value = segment.get(LE_FLOAT, offset);
      offset += Float.BYTES;
      return value;
    }

    private int[] readFourU32(String name) {
      int[] values = new int[4];
      for (int index = 0; index < values.length; index++) {
        values[index] = nonNegativeInt(readU32(), name);
      }
      return values;
    }
  }
}
