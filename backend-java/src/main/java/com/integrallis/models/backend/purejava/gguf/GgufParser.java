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

    List<GgufTensorInfo> tensorInfos = new ArrayList<>((int) header.tensorCount());
    for (long i = 0; i < header.tensorCount(); i++) {
      String name = cursor.readString();
      int nDimensions = cursor.readU32();
      long[] shape = new long[nDimensions];
      for (int d = 0; d < nDimensions; d++) {
        shape[d] = cursor.readU64();
      }
      int typeId = cursor.readU32();
      GgufTensorType type = GgufTensorType.fromId(typeId);
      long offset = cursor.readU64();
      tensorInfos.add(new GgufTensorInfo(name, nDimensions, shape, type, offset));
    }

    // Compute aligned tensor data start
    int alignment = metadata.getUint32("general.alignment").orElse(GgufConstants.DEFAULT_ALIGNMENT);
    long tensorDataOffset = alignUp(cursor.offset(), alignment);

    return new GgufFile(header, metadata, tensorInfos, tensorDataOffset, segment);
  }

  private static long alignUp(long value, long alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
  }
}
