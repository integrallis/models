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

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

/**
 * Test utility that programmatically constructs synthetic GGUF binary data. Used to create
 * controlled test fixtures without real model files.
 */
public final class SyntheticGgufBuilder {

  private int version = 3;
  private final List<MetadataEntry> metadataEntries = new ArrayList<>();
  private final List<TensorEntry> tensorEntries = new ArrayList<>();
  private int alignment = GgufConstants.DEFAULT_ALIGNMENT;

  public SyntheticGgufBuilder version(int version) {
    this.version = version;
    return this;
  }

  public SyntheticGgufBuilder alignment(int alignment) {
    this.alignment = alignment;
    return this;
  }

  public SyntheticGgufBuilder addString(String key, String value) {
    metadataEntries.add(new MetadataEntry(key, GgufValueType.STRING, value));
    return this;
  }

  public SyntheticGgufBuilder addUint32(String key, int value) {
    metadataEntries.add(new MetadataEntry(key, GgufValueType.UINT32, value));
    return this;
  }

  public SyntheticGgufBuilder addInt32(String key, int value) {
    metadataEntries.add(new MetadataEntry(key, GgufValueType.INT32, value));
    return this;
  }

  public SyntheticGgufBuilder addFloat32(String key, float value) {
    metadataEntries.add(new MetadataEntry(key, GgufValueType.FLOAT32, value));
    return this;
  }

  public SyntheticGgufBuilder addBool(String key, boolean value) {
    metadataEntries.add(new MetadataEntry(key, GgufValueType.BOOL, value));
    return this;
  }

  public SyntheticGgufBuilder addUint64(String key, long value) {
    metadataEntries.add(new MetadataEntry(key, GgufValueType.UINT64, value));
    return this;
  }

  public SyntheticGgufBuilder addStringArray(String key, List<String> values) {
    metadataEntries.add(new MetadataEntry(key, GgufValueType.ARRAY, new StringArrayData(values)));
    return this;
  }

  public SyntheticGgufBuilder addFloat32Array(String key, List<Float> values) {
    metadataEntries.add(new MetadataEntry(key, GgufValueType.ARRAY, new Float32ArrayData(values)));
    return this;
  }

  public SyntheticGgufBuilder addTensor(
      String name, GgufTensorType type, long[] shape, byte[] data) {
    tensorEntries.add(new TensorEntry(name, type, shape, data));
    return this;
  }

  /** Builds the synthetic GGUF binary data. */
  public byte[] build() {
    try {
      ByteArrayOutputStream out = new ByteArrayOutputStream();

      // Header: magic + version + tensorCount + metadataKvCount
      writeU32(out, GgufConstants.MAGIC);
      writeU32(out, version);
      writeU64(out, tensorEntries.size());
      writeU64(out, metadataEntries.size());

      // Metadata KV pairs
      for (MetadataEntry entry : metadataEntries) {
        writeString(out, entry.key);
        writeMetadataValue(out, entry);
      }

      // Tensor infos
      long dataOffset = 0;
      for (TensorEntry tensor : tensorEntries) {
        writeString(out, tensor.name);
        writeU32(out, tensor.shape.length);
        for (long dim : tensor.shape) {
          writeU64(out, dim);
        }
        writeU32(out, tensor.type.id());
        writeU64(out, dataOffset);
        dataOffset += tensor.data.length;
      }

      // Alignment padding before tensor data
      int currentSize = out.size();
      int aligned = alignUp(currentSize, alignment);
      for (int i = currentSize; i < aligned; i++) {
        out.write(0);
      }

      // Tensor data
      for (TensorEntry tensor : tensorEntries) {
        out.write(tensor.data);
      }

      return out.toByteArray();
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  private void writeMetadataValue(ByteArrayOutputStream out, MetadataEntry entry)
      throws IOException {
    writeU32(out, entry.type.id());
    switch (entry.type) {
      case STRING -> writeString(out, (String) entry.value);
      case UINT32 -> writeU32(out, (Integer) entry.value);
      case INT32 -> writeI32(out, (Integer) entry.value);
      case FLOAT32 -> writeF32(out, (Float) entry.value);
      case BOOL -> out.write(((Boolean) entry.value) ? 1 : 0);
      case UINT64 -> writeU64(out, (Long) entry.value);
      case ARRAY -> {
        if (entry.value instanceof StringArrayData sad) {
          writeU32(out, GgufValueType.STRING.id());
          writeU64(out, sad.values.size());
          for (String s : sad.values) {
            writeString(out, s);
          }
        } else if (entry.value instanceof Float32ArrayData fad) {
          writeU32(out, GgufValueType.FLOAT32.id());
          writeU64(out, fad.values.size());
          for (Float f : fad.values) {
            writeF32(out, f);
          }
        }
      }
      default -> throw new UnsupportedOperationException("Type not supported: " + entry.type);
    }
  }

  private static void writeU32(ByteArrayOutputStream out, int value) throws IOException {
    byte[] buf = new byte[4];
    ByteBuffer.wrap(buf).order(ByteOrder.LITTLE_ENDIAN).putInt(value);
    out.write(buf);
  }

  private static void writeI32(ByteArrayOutputStream out, int value) throws IOException {
    writeU32(out, value);
  }

  private static void writeU64(ByteArrayOutputStream out, long value) throws IOException {
    byte[] buf = new byte[8];
    ByteBuffer.wrap(buf).order(ByteOrder.LITTLE_ENDIAN).putLong(value);
    out.write(buf);
  }

  private static void writeF32(ByteArrayOutputStream out, float value) throws IOException {
    byte[] buf = new byte[4];
    ByteBuffer.wrap(buf).order(ByteOrder.LITTLE_ENDIAN).putFloat(value);
    out.write(buf);
  }

  private static void writeString(ByteArrayOutputStream out, String value) throws IOException {
    byte[] bytes = value.getBytes(StandardCharsets.UTF_8);
    writeU64(out, bytes.length);
    out.write(bytes);
  }

  private static int alignUp(int value, int alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
  }

  private record MetadataEntry(String key, GgufValueType type, Object value) {}

  private record TensorEntry(String name, GgufTensorType type, long[] shape, byte[] data) {}

  private record StringArrayData(List<String> values) {}

  private record Float32ArrayData(List<Float> values) {}
}
