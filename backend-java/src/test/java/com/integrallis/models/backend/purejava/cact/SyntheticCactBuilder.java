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

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;

/** Small deterministic `.cact` writer used only for parser tests. */
final class SyntheticCactBuilder {

  private static final int ALIGNMENT = 64;
  private static final int HEADER_BYTES = 120;
  private static final int RECORD_BYTES = 44;

  private final List<Tensor> tensors = new ArrayList<>();
  private int magic = 0x05E12A83;

  SyntheticCactBuilder addFp16(int... shape) {
    long elements = elements(shape);
    return add(CactTensorType.FP16, shape, new byte[Math.toIntExact(elements * Short.BYTES)], 0, 0);
  }

  SyntheticCactBuilder addCq(int rows, int columns, int groupSize, int recordBits) {
    int padded = Math.ceilDiv(columns, groupSize) * groupSize;
    int packedBits = recordBits == 5 ? 2 : recordBits;
    int packedBytes = rows * padded * packedBits / Byte.SIZE;
    int normBytes = rows * (padded / groupSize) * Short.BYTES;
    byte[] blob = new byte[packedBytes + normBytes];
    ByteBuffer norms =
        ByteBuffer.wrap(blob, packedBytes, normBytes).slice().order(ByteOrder.LITTLE_ENDIAN);
    while (norms.hasRemaining()) {
      norms.putShort(Float.floatToFloat16(1.0f));
    }
    return add(CactTensorType.CQ, new int[] {rows, columns}, blob, groupSize, recordBits);
  }

  SyntheticCactBuilder addRaw(byte... bytes) {
    return add(CactTensorType.RAW, new int[0], bytes, 0, 0);
  }

  SyntheticCactBuilder withMagic(int value) {
    magic = value;
    return this;
  }

  byte[] build() {
    int directoryEnd = HEADER_BYTES + 28 * Float.BYTES + tensors.size() * RECORD_BYTES;
    int cursor = directoryEnd;
    for (Tensor tensor : tensors) {
      cursor = align(cursor);
      tensor.offset = cursor;
      cursor = Math.addExact(cursor, tensor.blob.length);
    }

    ByteBuffer data = ByteBuffer.allocate(cursor).order(ByteOrder.LITTLE_ENDIAN);
    writeHeader(data);
    for (int index = 0; index < 28; index++) {
      data.putFloat((index - 13.5f) / 128.0f);
    }
    for (Tensor tensor : tensors) {
      data.put(tensor.type.id());
      data.put((byte) tensor.shape.length);
      data.putShort((short) 0);
      for (int dimension = 0; dimension < 4; dimension++) {
        data.putInt(dimension < tensor.shape.length ? tensor.shape[dimension] : 0);
      }
      data.putLong(tensor.offset);
      data.putLong(tensor.blob.length);
      data.putInt(tensor.groupSize);
      data.putInt(tensor.recordBits);
    }
    for (Tensor tensor : tensors) {
      data.position(tensor.offset);
      data.put(tensor.blob);
    }
    return data.array();
  }

  private SyntheticCactBuilder add(
      CactTensorType type, int[] shape, byte[] blob, int groupSize, int recordBits) {
    tensors.add(new Tensor(type, shape.clone(), blob.clone(), groupSize, recordBits));
    return this;
  }

  private void writeHeader(ByteBuffer data) {
    int[] fields = {
      magic,
      tensors.size(),
      28,
      16,
      8,
      32,
      8,
      2,
      1,
      1,
      4,
      64,
      8,
      1,
      8,
      4,
      2,
      4,
      3,
      2,
      2,
      3,
      0,
      0,
      1,
      0,
      0,
      0,
      0
    };
    for (int field : fields) {
      data.putInt(field);
    }
    data.putFloat(10_000.0f);
  }

  private static long elements(int[] shape) {
    long elements = 1;
    for (int dimension : shape) {
      elements = Math.multiplyExact(elements, dimension);
    }
    return elements;
  }

  private static int align(int value) {
    return (value + ALIGNMENT - 1) & -ALIGNMENT;
  }

  private static final class Tensor {
    private final CactTensorType type;
    private final int[] shape;
    private final byte[] blob;
    private final int groupSize;
    private final int recordBits;
    private int offset;

    private Tensor(CactTensorType type, int[] shape, byte[] blob, int groupSize, int recordBits) {
      this.type = type;
      this.shape = shape;
      this.blob = blob;
      this.groupSize = groupSize;
      this.recordBits = recordBits;
    }
  }
}
