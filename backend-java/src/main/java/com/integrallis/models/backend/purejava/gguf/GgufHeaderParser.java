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

import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Parses the GGUF header (magic, version, counts) and metadata key-value pairs from a {@link
 * MemorySegment}.
 */
public final class GgufHeaderParser {

  private GgufHeaderParser() {}

  /** Result of parsing the header and metadata from a GGUF file segment. */
  public record ParseResult(GgufHeader header, GgufMetadata metadata, long endOffset) {}

  /** Parses the GGUF header and metadata starting from offset 0 of the given segment. */
  public static ParseResult parse(MemorySegment segment) {
    var cursor = new GgufReader.Cursor(segment, 0);

    int magic = cursor.readU32();
    if (magic != GgufConstants.MAGIC) {
      throw new IllegalArgumentException(
          String.format(
              "Invalid GGUF magic: 0x%08X (expected 0x%08X)", magic, GgufConstants.MAGIC));
    }

    int version = cursor.readU32();
    long tensorCount = cursor.readU64();
    long metadataKvCount = cursor.readU64();

    GgufHeader header = new GgufHeader(version, tensorCount, metadataKvCount);

    Map<String, GgufMetadataValue> entries = new LinkedHashMap<>();
    for (long i = 0; i < metadataKvCount; i++) {
      String key = cursor.readString();
      GgufMetadataValue value = readValue(cursor);
      entries.put(key, value);
    }

    return new ParseResult(header, new GgufMetadata(entries), cursor.offset());
  }

  private static GgufMetadataValue readValue(GgufReader.Cursor cursor) {
    int typeId = cursor.readU32();
    GgufValueType type = GgufValueType.fromId(typeId);
    return readTypedValue(cursor, type);
  }

  private static GgufMetadataValue readTypedValue(GgufReader.Cursor cursor, GgufValueType type) {
    return switch (type) {
      case UINT8 -> new GgufMetadataValue.Uint8Value(cursor.readU8());
      case INT8 -> new GgufMetadataValue.Int8Value(cursor.readI8());
      case UINT16 -> new GgufMetadataValue.Uint16Value(cursor.readU16());
      case INT16 -> new GgufMetadataValue.Int16Value(cursor.readI16());
      case UINT32 -> new GgufMetadataValue.Uint32Value(cursor.readU32());
      case INT32 -> new GgufMetadataValue.Int32Value(cursor.readI32());
      case FLOAT32 -> new GgufMetadataValue.Float32Value(cursor.readF32());
      case BOOL -> new GgufMetadataValue.BoolValue(cursor.readBool());
      case STRING -> new GgufMetadataValue.StringValue(cursor.readString());
      case ARRAY -> readArray(cursor);
      case UINT64 -> new GgufMetadataValue.Uint64Value(cursor.readU64());
      case INT64 -> new GgufMetadataValue.Int64Value(cursor.readI64());
      case FLOAT64 -> new GgufMetadataValue.Float64Value(cursor.readF64());
    };
  }

  private static GgufMetadataValue.ArrayValue readArray(GgufReader.Cursor cursor) {
    int elementTypeId = cursor.readU32();
    GgufValueType elementType = GgufValueType.fromId(elementTypeId);
    long length = cursor.readU64();
    List<GgufMetadataValue> elements = new ArrayList<>((int) length);
    for (long i = 0; i < length; i++) {
      elements.add(readTypedValue(cursor, elementType));
    }
    return new GgufMetadataValue.ArrayValue(elementType, elements);
  }
}
