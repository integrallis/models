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
package com.integrallis.models.backend.onnx;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/** Builders for small, valid ONNX protobuf messages used across the unit tests. */
final class OnnxFixtures {

  private OnnxFixtures() {}

  static ProtobufWriter floatTensor(String name, long[] dims, float... values) {
    ByteBuffer raw = ByteBuffer.allocate(values.length * 4).order(ByteOrder.LITTLE_ENDIAN);
    for (float value : values) {
      raw.putFloat(value);
    }
    return new ProtobufWriter()
        .packedVarints(1, dims)
        .varint(2, OnnxDataType.FLOAT.code())
        .string(8, name)
        .bytes(9, raw.array());
  }

  static ProtobufWriter int64Tensor(String name, long[] dims, long... values) {
    ByteBuffer raw = ByteBuffer.allocate(values.length * 8).order(ByteOrder.LITTLE_ENDIAN);
    for (long value : values) {
      raw.putLong(value);
    }
    return new ProtobufWriter()
        .packedVarints(1, dims)
        .varint(2, OnnxDataType.INT64.code())
        .string(8, name)
        .bytes(9, raw.array());
  }

  static ProtobufWriter valueInfo(String name, OnnxDataType type, Object... dims) {
    ProtobufWriter shape = new ProtobufWriter();
    for (Object dim : dims) {
      ProtobufWriter dimension = new ProtobufWriter();
      if (dim instanceof String param) {
        dimension.string(2, param);
      } else {
        dimension.varint(1, ((Number) dim).longValue());
      }
      shape.message(1, dimension);
    }
    ProtobufWriter tensorType = new ProtobufWriter().varint(1, type.code()).message(2, shape);
    ProtobufWriter typeProto = new ProtobufWriter().message(1, tensorType);
    return new ProtobufWriter().string(1, name).message(2, typeProto);
  }

  static ProtobufWriter attributeInt(String name, long value) {
    return new ProtobufWriter()
        .string(1, name)
        .varint(3, value)
        .varint(20, OnnxAttribute.Type.INT.ordinal());
  }

  static ProtobufWriter attributeFloat(String name, float value) {
    return new ProtobufWriter()
        .string(1, name)
        .fixed32(2, value)
        .varint(20, OnnxAttribute.Type.FLOAT.ordinal());
  }

  static ProtobufWriter attributeString(String name, String value) {
    return new ProtobufWriter()
        .string(1, name)
        .string(4, value)
        .varint(20, OnnxAttribute.Type.STRING.ordinal());
  }

  static ProtobufWriter attributeInts(String name, long... values) {
    return new ProtobufWriter()
        .string(1, name)
        .packedVarints(8, values)
        .varint(20, OnnxAttribute.Type.INTS.ordinal());
  }

  static ProtobufWriter attributeFloats(String name, float... values) {
    return new ProtobufWriter()
        .string(1, name)
        .packedFloats(7, values)
        .varint(20, OnnxAttribute.Type.FLOATS.ordinal());
  }

  static ProtobufWriter attributeTensor(String name, ProtobufWriter tensor) {
    return new ProtobufWriter()
        .string(1, name)
        .message(5, tensor)
        .varint(20, OnnxAttribute.Type.TENSOR.ordinal());
  }

  static ProtobufWriter node(String opType, String name, String[] inputs, String[] outputs) {
    ProtobufWriter node = new ProtobufWriter();
    for (String input : inputs) {
      node.string(1, input);
    }
    for (String output : outputs) {
      node.string(2, output);
    }
    return node.string(3, name).string(4, opType);
  }

  /** A model whose graph is {@code y = Add(x, w)} with one feed input and one initializer. */
  static byte[] addModel() {
    ProtobufWriter node = node("Add", "add0", new String[] {"x", "w"}, new String[] {"y"});
    ProtobufWriter graph =
        new ProtobufWriter()
            .message(1, node)
            .string(2, "tiny")
            .message(5, floatTensor("w", new long[] {2}, 0.5f, -1.5f))
            .message(11, valueInfo("x", OnnxDataType.FLOAT, 1, "n"))
            .message(11, valueInfo("w", OnnxDataType.FLOAT, 2))
            .message(12, valueInfo("y", OnnxDataType.FLOAT, 1, "n"));
    ProtobufWriter opset = new ProtobufWriter().string(1, "").varint(2, 17);
    return new ProtobufWriter()
        .varint(1, 8)
        .string(2, "pytorch")
        .string(3, "2.14.1")
        .message(8, opset)
        .message(7, graph)
        .toByteArray();
  }

  /** A model from explicit parts; every list may be empty. Opset ai.onnx 17, IR 8. */
  static byte[] model(
      java.util.List<ProtobufWriter> nodes,
      java.util.List<ProtobufWriter> initializers,
      java.util.List<ProtobufWriter> inputs,
      java.util.List<ProtobufWriter> outputs) {
    ProtobufWriter graph = new ProtobufWriter();
    for (ProtobufWriter node : nodes) {
      graph.message(1, node);
    }
    graph.string(2, "fixture");
    for (ProtobufWriter initializer : initializers) {
      graph.message(5, initializer);
    }
    for (ProtobufWriter input : inputs) {
      graph.message(11, input);
    }
    for (ProtobufWriter output : outputs) {
      graph.message(12, output);
    }
    ProtobufWriter opset = new ProtobufWriter().string(1, "").varint(2, 17);
    return new ProtobufWriter().varint(1, 8).message(8, opset).message(7, graph).toByteArray();
  }

  static ProtobufWriter node(
      String opType, String[] inputs, String[] outputs, ProtobufWriter... attributes) {
    ProtobufWriter node = node(opType, "", inputs, outputs);
    for (ProtobufWriter attribute : attributes) {
      node.message(5, attribute);
    }
    return node;
  }
}
