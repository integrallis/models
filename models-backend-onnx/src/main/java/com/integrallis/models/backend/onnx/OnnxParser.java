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
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Field-by-field decoding of the ONNX protobuf schema into the module's records. */
final class OnnxParser {

  private OnnxParser() {}

  static OnnxModel parseModel(byte[] bytes) {
    ProtobufReader reader = new ProtobufReader(bytes);
    long irVersion = 0;
    String producerName = "";
    String producerVersion = "";
    Map<String, Long> opsets = new LinkedHashMap<>();
    OnnxGraph graph = null;
    while (reader.hasMore()) {
      int tag = reader.readTag();
      int wire = ProtobufReader.wireType(tag);
      switch (ProtobufReader.fieldNumber(tag)) {
        case 1 -> irVersion = reader.readVarint();
        case 2 -> producerName = reader.readString();
        case 3 -> producerVersion = reader.readString();
        case 7 -> graph = parseGraph(reader.readMessage());
        case 8 -> {
          ProtobufReader opset = reader.readMessage();
          String domain = "";
          long version = 0;
          while (opset.hasMore()) {
            int opsetTag = opset.readTag();
            switch (ProtobufReader.fieldNumber(opsetTag)) {
              case 1 -> domain = opset.readString();
              case 2 -> version = opset.readVarint();
              default -> opset.skip(ProtobufReader.wireType(opsetTag));
            }
          }
          opsets.put(domain.isEmpty() ? "ai.onnx" : domain, version);
        }
        default -> reader.skip(wire);
      }
    }
    if (graph == null) {
      throw new IllegalArgumentException("ModelProto has no graph");
    }
    return new OnnxModel(irVersion, producerName, producerVersion, opsets, graph);
  }

  static OnnxGraph parseGraph(ProtobufReader reader) {
    String name = "";
    List<OnnxNode> nodes = new ArrayList<>();
    Map<String, Tensor> initializers = new LinkedHashMap<>();
    List<OnnxValueInfo> inputs = new ArrayList<>();
    List<OnnxValueInfo> outputs = new ArrayList<>();
    List<OnnxValueInfo> valueInfos = new ArrayList<>();
    while (reader.hasMore()) {
      int tag = reader.readTag();
      switch (ProtobufReader.fieldNumber(tag)) {
        case 1 -> nodes.add(parseNode(reader.readMessage()));
        case 2 -> name = reader.readString();
        case 5 -> {
          NamedTensor initializer = parseTensor(reader.readMessage());
          initializers.put(initializer.name(), initializer.tensor());
        }
        case 11 -> inputs.add(parseValueInfo(reader.readMessage()));
        case 12 -> outputs.add(parseValueInfo(reader.readMessage()));
        case 13 -> valueInfos.add(parseValueInfo(reader.readMessage()));
        default -> reader.skip(ProtobufReader.wireType(tag));
      }
    }
    return new OnnxGraph(name, nodes, initializers, inputs, outputs, valueInfos);
  }

  static OnnxNode parseNode(ProtobufReader reader) {
    String name = "";
    String opType = "";
    String domain = "";
    List<String> inputs = new ArrayList<>();
    List<String> outputs = new ArrayList<>();
    Map<String, OnnxAttribute> attributes = new LinkedHashMap<>();
    while (reader.hasMore()) {
      int tag = reader.readTag();
      switch (ProtobufReader.fieldNumber(tag)) {
        case 1 -> inputs.add(reader.readString());
        case 2 -> outputs.add(reader.readString());
        case 3 -> name = reader.readString();
        case 4 -> opType = reader.readString();
        case 5 -> {
          OnnxAttribute attribute = parseAttribute(reader.readMessage());
          attributes.put(attribute.name(), attribute);
        }
        case 7 -> domain = reader.readString();
        default -> reader.skip(ProtobufReader.wireType(tag));
      }
    }
    return new OnnxNode(name, opType, domain, inputs, outputs, attributes);
  }

  static OnnxAttribute parseAttribute(ProtobufReader reader) {
    String name = "";
    OnnxAttribute.Type type = OnnxAttribute.Type.UNDEFINED;
    float f = 0;
    long i = 0;
    String s = "";
    Tensor t = null;
    float[] floats = new float[0];
    long[] ints = new long[0];
    List<String> strings = new ArrayList<>();
    boolean sawFloat = false;
    boolean sawInt = false;
    boolean sawString = false;
    while (reader.hasMore()) {
      int tag = reader.readTag();
      int wire = ProtobufReader.wireType(tag);
      switch (ProtobufReader.fieldNumber(tag)) {
        case 1 -> name = reader.readString();
        case 2 -> {
          f = reader.readFloat();
          sawFloat = true;
        }
        case 3 -> {
          i = reader.readVarint();
          sawInt = true;
        }
        case 4 -> {
          s = reader.readString();
          sawString = true;
        }
        case 5 -> t = parseTensor(reader.readMessage()).tensor();
        case 7 -> floats = ProtobufReader.appendFloats(floats, reader, wire);
        case 8 -> ints = ProtobufReader.appendVarints(ints, reader, wire);
        case 9 -> strings.add(reader.readString());
        case 20 -> type = OnnxAttribute.Type.fromCode(reader.readVarint());
        default -> reader.skip(wire);
      }
    }
    if (type == OnnxAttribute.Type.UNDEFINED) {
      // Older exporters omit the type field; infer it from whichever slot was written.
      if (t != null) {
        type = OnnxAttribute.Type.TENSOR;
      } else if (floats.length > 0) {
        type = OnnxAttribute.Type.FLOATS;
      } else if (ints.length > 0) {
        type = OnnxAttribute.Type.INTS;
      } else if (!strings.isEmpty()) {
        type = OnnxAttribute.Type.STRINGS;
      } else if (sawFloat) {
        type = OnnxAttribute.Type.FLOAT;
      } else if (sawInt) {
        type = OnnxAttribute.Type.INT;
      } else if (sawString) {
        type = OnnxAttribute.Type.STRING;
      }
    }
    return new OnnxAttribute(name, type, f, i, s, t, floats, ints, strings);
  }

  static OnnxValueInfo parseValueInfo(ProtobufReader reader) {
    String name = "";
    OnnxDataType elementType = OnnxDataType.UNDEFINED;
    List<Long> dims = new ArrayList<>();
    List<String> dimParams = new ArrayList<>();
    while (reader.hasMore()) {
      int tag = reader.readTag();
      switch (ProtobufReader.fieldNumber(tag)) {
        case 1 -> name = reader.readString();
        case 2 -> {
          ProtobufReader typeProto = reader.readMessage();
          while (typeProto.hasMore()) {
            int typeTag = typeProto.readTag();
            if (ProtobufReader.fieldNumber(typeTag) == 1) {
              ProtobufReader tensorType = typeProto.readMessage();
              while (tensorType.hasMore()) {
                int tensorTag = tensorType.readTag();
                switch (ProtobufReader.fieldNumber(tensorTag)) {
                  case 1 -> elementType = OnnxDataType.fromCode(tensorType.readVarint());
                  case 2 -> parseShape(tensorType.readMessage(), dims, dimParams);
                  default -> tensorType.skip(ProtobufReader.wireType(tensorTag));
                }
              }
            } else {
              typeProto.skip(ProtobufReader.wireType(typeTag));
            }
          }
        }
        default -> reader.skip(ProtobufReader.wireType(tag));
      }
    }
    long[] dimArray = new long[dims.size()];
    for (int index = 0; index < dimArray.length; index++) {
      dimArray[index] = dims.get(index);
    }
    return new OnnxValueInfo(name, elementType, dimArray, dimParams);
  }

  private static void parseShape(ProtobufReader shape, List<Long> dims, List<String> dimParams) {
    while (shape.hasMore()) {
      int tag = shape.readTag();
      if (ProtobufReader.fieldNumber(tag) != 1) {
        shape.skip(ProtobufReader.wireType(tag));
        continue;
      }
      ProtobufReader dim = shape.readMessage();
      long value = -1;
      String param = "";
      while (dim.hasMore()) {
        int dimTag = dim.readTag();
        switch (ProtobufReader.fieldNumber(dimTag)) {
          case 1 -> value = dim.readVarint();
          case 2 -> param = dim.readString();
          default -> dim.skip(ProtobufReader.wireType(dimTag));
        }
      }
      dims.add(param.isEmpty() ? value : -1L);
      dimParams.add(param);
    }
  }

  record NamedTensor(String name, Tensor tensor) {}

  static NamedTensor parseTensor(ProtobufReader reader) {
    String name = "";
    OnnxDataType dataType = OnnxDataType.UNDEFINED;
    long[] dims = new long[0];
    float[] floatData = new float[0];
    long[] int32Data = new long[0];
    long[] int64Data = new long[0];
    double[] doubleData = new double[0];
    byte[] rawData = null;
    boolean external = false;
    while (reader.hasMore()) {
      int tag = reader.readTag();
      int wire = ProtobufReader.wireType(tag);
      switch (ProtobufReader.fieldNumber(tag)) {
        case 1 -> dims = ProtobufReader.appendVarints(dims, reader, wire);
        case 2 -> dataType = OnnxDataType.fromCode(reader.readVarint());
        case 4 -> floatData = ProtobufReader.appendFloats(floatData, reader, wire);
        case 5 -> int32Data = ProtobufReader.appendVarints(int32Data, reader, wire);
        case 7 -> int64Data = ProtobufReader.appendVarints(int64Data, reader, wire);
        case 8 -> name = reader.readString();
        case 9 -> rawData = reader.readBytes();
        case 10 -> doubleData = ProtobufReader.appendDoubles(doubleData, reader, wire);
        case 13 -> {
          external = true;
          reader.skip(wire);
        }
        default -> reader.skip(wire);
      }
    }
    if (external) {
      throw new UnsupportedOperationException(
          "tensor '" + name + "' stores its data externally, which this module does not read");
    }
    Tensor tensor =
        rawData != null
            ? decodeRaw(name, dataType, dims, rawData)
            : decodeTyped(name, dataType, dims, floatData, int32Data, int64Data, doubleData);
    return new NamedTensor(name, tensor);
  }

  private static Tensor decodeRaw(String name, OnnxDataType type, long[] dims, byte[] raw) {
    ByteBuffer buffer = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN);
    int count = (int) Tensor.elementCount(dims);
    switch (type) {
      case FLOAT -> {
        float[] values = new float[count];
        buffer.asFloatBuffer().get(values);
        return Tensor.ofFloats(dims, values);
      }
      case FLOAT16 -> {
        float[] values = new float[count];
        for (int i = 0; i < count; i++) {
          values[i] = Float.float16ToFloat(buffer.getShort(i * 2));
        }
        return Tensor.ofFloats(dims, values);
      }
      case BFLOAT16 -> {
        float[] values = new float[count];
        for (int i = 0; i < count; i++) {
          values[i] = Float.intBitsToFloat((buffer.getShort(i * 2) & 0xFFFF) << 16);
        }
        return Tensor.ofFloats(dims, values);
      }
      case DOUBLE -> {
        double[] values = new double[count];
        buffer.asDoubleBuffer().get(values);
        return Tensor.ofDoubles(dims, values);
      }
      case INT64 -> {
        long[] values = new long[count];
        buffer.asLongBuffer().get(values);
        return Tensor.ofLongs(dims, values);
      }
      case INT32 -> {
        long[] values = new long[count];
        for (int i = 0; i < count; i++) {
          values[i] = buffer.getInt(i * 4);
        }
        return Tensor.ofInts(dims, values);
      }
      case INT8, UINT8, BOOL -> {
        if (raw.length != count) {
          throw new IllegalArgumentException(
              "tensor '" + name + "': " + raw.length + " raw bytes for " + count + " elements");
        }
        return Tensor.ofBytes(type, dims, raw);
      }
      default ->
          throw new UnsupportedOperationException(
              "tensor '"
                  + name
                  + "' has element type "
                  + type
                  + ", which this module does not read");
    }
  }

  private static Tensor decodeTyped(
      String name,
      OnnxDataType type,
      long[] dims,
      float[] floatData,
      long[] int32Data,
      long[] int64Data,
      double[] doubleData) {
    switch (type) {
      case FLOAT -> {
        return Tensor.ofFloats(dims, floatData);
      }
      case DOUBLE -> {
        return Tensor.ofDoubles(dims, doubleData);
      }
      case FLOAT16, BFLOAT16 -> {
        // Half floats travel in int32_data, one value per entry.
        float[] values = new float[int32Data.length];
        for (int i = 0; i < values.length; i++) {
          short bits = (short) int32Data[i];
          values[i] =
              type == OnnxDataType.FLOAT16
                  ? Float.float16ToFloat(bits)
                  : Float.intBitsToFloat((bits & 0xFFFF) << 16);
        }
        return Tensor.ofFloats(dims, values);
      }
      case INT64 -> {
        return Tensor.ofLongs(dims, int64Data);
      }
      case INT32 -> {
        return Tensor.ofInts(dims, int32Data);
      }
      case INT8, UINT8, BOOL -> {
        byte[] values = new byte[int32Data.length];
        for (int i = 0; i < values.length; i++) {
          values[i] = (byte) int32Data[i];
        }
        return Tensor.ofBytes(type, dims, values);
      }
      default ->
          throw new UnsupportedOperationException(
              "tensor '"
                  + name
                  + "' has element type "
                  + type
                  + ", which this module does not read");
    }
  }
}
