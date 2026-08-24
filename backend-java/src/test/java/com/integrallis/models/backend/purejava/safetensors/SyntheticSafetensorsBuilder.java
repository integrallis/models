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

import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

final class SyntheticSafetensorsBuilder {

  private record Tensor(String name, String dtype, long[] shape, byte[] data) {}

  private final List<Tensor> tensors = new ArrayList<>();
  private String metadata;

  SyntheticSafetensorsBuilder metadata(String key, String value) {
    metadata = "\"__metadata__\":{\"" + key + "\":\"" + value + "\"}";
    return this;
  }

  SyntheticSafetensorsBuilder add(String name, String dtype, long[] shape, int... values) {
    byte[] data = new byte[values.length];
    for (int index = 0; index < values.length; index++) {
      if (values[index] < Byte.MIN_VALUE || values[index] > 0xff) {
        throw new IllegalArgumentException("test byte is outside [-128, 255]: " + values[index]);
      }
      data[index] = (byte) values[index];
    }
    tensors.add(new Tensor(name, dtype, shape.clone(), data));
    return this;
  }

  byte[] build() {
    StringBuilder header = new StringBuilder("{");
    boolean comma = false;
    if (metadata != null) {
      header.append(metadata);
      comma = true;
    }
    long offset = 0;
    for (Tensor tensor : tensors) {
      if (comma) {
        header.append(',');
      }
      long end = offset + tensor.data().length;
      header
          .append('"')
          .append(tensor.name())
          .append("\":{\"dtype\":\"")
          .append(tensor.dtype())
          .append("\",\"shape\":[");
      for (int dimension = 0; dimension < tensor.shape().length; dimension++) {
        if (dimension > 0) {
          header.append(',');
        }
        header.append(tensor.shape()[dimension]);
      }
      header.append("],\"data_offsets\":[").append(offset).append(',').append(end).append("]}");
      offset = end;
      comma = true;
    }
    header.append('}');
    return file(header.toString(), tensorData());
  }

  private byte[] tensorData() {
    ByteArrayOutputStream output = new ByteArrayOutputStream();
    for (Tensor tensor : tensors) {
      output.writeBytes(tensor.data());
    }
    return output.toByteArray();
  }

  static byte[] file(String rawHeader, byte[] tensorData) {
    byte[] unpadded = rawHeader.getBytes(StandardCharsets.UTF_8);
    int headerLength = Math.addExact(unpadded.length, (8 - (unpadded.length & 7)) & 7);
    ByteBuffer file =
        ByteBuffer.allocate(Math.addExact(8 + headerLength, tensorData.length))
            .order(ByteOrder.LITTLE_ENDIAN);
    file.putLong(headerLength).put(unpadded);
    while (file.position() < 8 + headerLength) {
      file.put((byte) ' ');
    }
    file.put(tensorData);
    return file.array();
  }
}
