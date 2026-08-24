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
package com.integrallis.models.backend.purejava.safetensors;

import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonToken;
import com.fasterxml.jackson.core.StreamReadFeature;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/** Strict, memory-mapped reader for the Safetensors storage format. */
public final class SafetensorsParser {

  static final JsonFactory JSON =
      JsonFactory.builder().enable(StreamReadFeature.STRICT_DUPLICATE_DETECTION).build();
  private static final long MAX_HEADER_BYTES = 100_000_000;
  private static final long MAX_SIZE_T = (1L << 48) - 1;
  private static final long LENGTH_BYTES = Long.BYTES;
  private static final ValueLayout.OfLong LE_LONG =
      ValueLayout.JAVA_LONG_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

  private SafetensorsParser() {}

  /** Maps and validates a Safetensors file in the supplied arena. */
  public static SafetensorsFile parse(Path path, Arena arena) throws IOException {
    Objects.requireNonNull(path, "path");
    Objects.requireNonNull(arena, "arena");
    try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
      MemorySegment segment = channel.map(FileChannel.MapMode.READ_ONLY, 0, channel.size(), arena);
      return parseSegment(segment);
    }
  }

  /** Validates an already mapped or heap-backed Safetensors file. */
  public static SafetensorsFile parseSegment(MemorySegment segment) {
    Objects.requireNonNull(segment, "segment");
    if (segment.byteSize() < LENGTH_BYTES) {
      throw new MalformedSafetensorsException("header is smaller than the 8-byte length prefix");
    }
    try {
      long headerLength = segment.get(LE_LONG, 0);
      if (headerLength < 0 || headerLength > MAX_HEADER_BYTES) {
        throw new MalformedSafetensorsException(
            "header length must be from 0 through " + MAX_HEADER_BYTES + "; got " + headerLength);
      }
      long dataStart = Math.addExact(LENGTH_BYTES, headerLength);
      if (dataStart > segment.byteSize()) {
        throw new MalformedSafetensorsException(
            "header ends at " + dataStart + " beyond file size " + segment.byteSize());
      }
      if (headerLength == 0 || segment.get(ValueLayout.JAVA_BYTE, LENGTH_BYTES) != '{') {
        throw new MalformedSafetensorsException("JSON header must begin with '{'");
      }
      byte[] header = segment.asSlice(LENGTH_BYTES, headerLength).toArray(ValueLayout.JAVA_BYTE);
      ParsedHeader parsed = parseHeader(header);
      long dataBytes = segment.byteSize() - dataStart;
      validateLayout(parsed.tensors(), dataBytes);
      return new SafetensorsFile(
          parsed.metadata(), parsed.tensors(), segment.asSlice(dataStart, dataBytes));
    } catch (MalformedSafetensorsException malformed) {
      throw malformed;
    } catch (ArithmeticException | IndexOutOfBoundsException malformed) {
      throw new MalformedSafetensorsException("Safetensors size or offset overflow", malformed);
    }
  }

  private static ParsedHeader parseHeader(byte[] header) {
    Map<String, String> metadata = new LinkedHashMap<>();
    Map<String, SafetensorsTensorInfo> tensors = new LinkedHashMap<>();
    try (JsonParser parser = JSON.createParser(header)) {
      require(parser.nextToken(), JsonToken.START_OBJECT, "header root");
      while (parser.nextToken() != JsonToken.END_OBJECT) {
        require(parser.currentToken(), JsonToken.FIELD_NAME, "header entry");
        String name = parser.currentName();
        JsonToken value = parser.nextToken();
        if ("__metadata__".equals(name)) {
          parseMetadata(parser, value, metadata);
        } else {
          tensors.put(name, parseTensor(parser, value, name));
        }
      }
      if (parser.nextToken() != null) {
        throw new MalformedSafetensorsException("JSON header has content after its root object");
      }
      return new ParsedHeader(metadata, tensors);
    } catch (MalformedSafetensorsException malformed) {
      throw malformed;
    } catch (IOException malformed) {
      throw new MalformedSafetensorsException(
          "invalid or duplicate JSON header: " + malformed.getMessage(), malformed);
    }
  }

  private static void parseMetadata(
      JsonParser parser, JsonToken token, Map<String, String> metadata) throws IOException {
    require(token, JsonToken.START_OBJECT, "__metadata__");
    while (parser.nextToken() != JsonToken.END_OBJECT) {
      require(parser.currentToken(), JsonToken.FIELD_NAME, "metadata key");
      String key = parser.currentName();
      require(parser.nextToken(), JsonToken.VALUE_STRING, "metadata value for " + key);
      metadata.put(key, parser.getText());
    }
  }

  private static SafetensorsTensorInfo parseTensor(JsonParser parser, JsonToken token, String name)
      throws IOException {
    require(token, JsonToken.START_OBJECT, "tensor " + name);
    SafetensorsDtype dtype = null;
    long[] shape = null;
    long[] offsets = null;
    while (parser.nextToken() != JsonToken.END_OBJECT) {
      require(parser.currentToken(), JsonToken.FIELD_NAME, "tensor field for " + name);
      String field = parser.currentName();
      JsonToken value = parser.nextToken();
      switch (field) {
        case "dtype" -> {
          require(value, JsonToken.VALUE_STRING, "dtype for " + name);
          dtype = SafetensorsDtype.fromCode(parser.getText());
        }
        case "shape" -> shape = parseNonNegativeLongArray(parser, value, "shape for " + name);
        case "data_offsets" ->
            offsets = parseNonNegativeLongArray(parser, value, "data_offsets for " + name);
        default ->
            throw new MalformedSafetensorsException(
                "tensor " + name + " has unsupported header field: " + field);
      }
    }
    if (dtype == null || shape == null || offsets == null) {
      throw new MalformedSafetensorsException(
          "tensor " + name + " must declare dtype, shape, and data_offsets");
    }
    if (offsets.length != 2) {
      throw new MalformedSafetensorsException(
          "tensor " + name + " data_offsets must contain exactly two values");
    }
    return new SafetensorsTensorInfo(name, dtype, shape, offsets[0], offsets[1]);
  }

  private static long[] parseNonNegativeLongArray(
      JsonParser parser, JsonToken token, String description) throws IOException {
    require(token, JsonToken.START_ARRAY, description);
    List<Long> values = new ArrayList<>();
    while (parser.nextToken() != JsonToken.END_ARRAY) {
      if (parser.currentToken() != JsonToken.VALUE_NUMBER_INT) {
        throw new MalformedSafetensorsException(description + " must contain integers");
      }
      long value;
      try {
        value = parser.getLongValue();
      } catch (RuntimeException invalid) {
        throw new MalformedSafetensorsException(
            description + " contains an invalid integer", invalid);
      }
      if (value < 0) {
        throw new MalformedSafetensorsException(description + " must not contain negative values");
      }
      if (value > MAX_SIZE_T) {
        throw new MalformedSafetensorsException(
            description + " values must fit the official unsigned 48-bit size limit");
      }
      values.add(value);
    }
    long[] result = new long[values.size()];
    for (int index = 0; index < values.size(); index++) {
      result[index] = values.get(index);
    }
    return result;
  }

  private static void validateLayout(Map<String, SafetensorsTensorInfo> tensors, long dataBytes) {
    List<SafetensorsTensorInfo> byOffset =
        tensors.values().stream()
            .sorted(
                Comparator.comparingLong(SafetensorsTensorInfo::dataBegin)
                    .thenComparingLong(SafetensorsTensorInfo::dataEnd))
            .toList();
    long expectedBegin = 0;
    for (SafetensorsTensorInfo tensor : byOffset) {
      if (tensor.dataBegin() != expectedBegin || tensor.dataEnd() < tensor.dataBegin()) {
        throw new MalformedSafetensorsException(
            "tensor "
                + tensor.name()
                + " has invalid data offset range ["
                + tensor.dataBegin()
                + ", "
                + tensor.dataEnd()
                + "); expected begin "
                + expectedBegin);
      }
      long expectedBytes = tensor.expectedByteCount();
      if (tensor.byteCount() != expectedBytes) {
        throw new MalformedSafetensorsException(
            "tensor "
                + tensor.name()
                + " byte count must be "
                + expectedBytes
                + "; got "
                + tensor.byteCount());
      }
      expectedBegin = tensor.dataEnd();
    }
    if (expectedBegin != dataBytes) {
      throw new MalformedSafetensorsException(
          "tensor offsets must index the entire data buffer; indexed "
              + expectedBegin
              + " of "
              + dataBytes
              + " bytes");
    }
  }

  private static void require(JsonToken actual, JsonToken expected, String description) {
    if (actual != expected) {
      throw new MalformedSafetensorsException(
          description + " must be " + expected + "; got " + actual);
    }
  }

  private record ParsedHeader(
      Map<String, String> metadata, Map<String, SafetensorsTensorInfo> tensors) {}
}
