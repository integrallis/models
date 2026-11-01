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

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class OnnxModelTest {

  @Test
  void decodesModelGraphNodesInitializersAndSignatures() {
    OnnxModel model = OnnxModel.parse(OnnxFixtures.addModel());

    assertThat(model.irVersion()).isEqualTo(8);
    assertThat(model.producerName()).isEqualTo("pytorch");
    assertThat(model.producerVersion()).isEqualTo("2.14.1");
    assertThat(model.defaultOpset()).isEqualTo(17);
    assertThat(model.opsets()).containsEntry("ai.onnx", 17L);

    OnnxGraph graph = model.graph();
    assertThat(graph.name()).isEqualTo("tiny");
    assertThat(graph.nodes()).hasSize(1);
    OnnxNode node = graph.nodes().get(0);
    assertThat(node.opType()).isEqualTo("Add");
    assertThat(node.name()).isEqualTo("add0");
    assertThat(node.inputs()).containsExactly("x", "w");
    assertThat(node.outputs()).containsExactly("y");
    assertThat(node.label()).isEqualTo("Add 'add0'");

    assertThat(graph.initializers()).containsOnlyKeys("w");
    assertThat(graph.initializers().get("w").floats()).containsExactly(0.5f, -1.5f);
    assertThat(graph.feedInputs()).extracting(OnnxValueInfo::name).containsExactly("x");
    OnnxValueInfo x = graph.inputs().get(0);
    assertThat(x.elementType()).isEqualTo(OnnxDataType.FLOAT);
    assertThat(x.dims()).containsExactly(1, -1);
    assertThat(x.dimParams()).containsExactly("", "n");
    assertThat(graph.outputs()).extracting(OnnxValueInfo::name).containsExactly("y");
  }

  @Test
  void decodesEveryAttributeKindAndInfersTypeWhenOmitted() {
    ProtobufWriter node =
        OnnxFixtures.node("Conv", "", new String[] {"x", "", "b"}, new String[] {"y"})
            .message(5, OnnxFixtures.attributeInt("group", 2))
            .message(5, OnnxFixtures.attributeFloat("alpha", 0.25f))
            .message(5, OnnxFixtures.attributeString("mode", "linear"))
            .message(5, OnnxFixtures.attributeInts("pads", 1, 2))
            .message(5, OnnxFixtures.attributeFloats("scales", 1f, 2f))
            .message(
                5,
                OnnxFixtures.attributeTensor(
                    "value", OnnxFixtures.int64Tensor("", new long[] {2}, 7, -8)))
            // An attribute with no type field, as old exporters wrote them.
            .message(5, new ProtobufWriter().string(1, "untyped").varint(3, 5))
            .message(5, new ProtobufWriter().string(1, "untypedStrings").string(9, "a"))
            .string(7, "custom");
    OnnxNode parsed = OnnxParser.parseNode(new ProtobufReader(node.toByteArray()));

    assertThat(parsed.domain()).isEqualTo("custom");
    assertThat(parsed.input(1)).isNull();
    assertThat(parsed.hasInput(2)).isTrue();
    assertThat(parsed.input(5)).isNull();
    assertThat(parsed.label()).isEqualTo("Conv");
    assertThat(parsed.intAttribute("group", 1)).isEqualTo(2);
    assertThat(parsed.intAttribute("missing", 1)).isEqualTo(1);
    assertThat(parsed.floatAttribute("alpha", 0f)).isEqualTo(0.25f);
    assertThat(parsed.floatAttribute("missing", 3f)).isEqualTo(3f);
    assertThat(parsed.stringAttribute("mode", "nearest")).isEqualTo("linear");
    assertThat(parsed.stringAttribute("missing", "nearest")).isEqualTo("nearest");
    assertThat(parsed.intsAttribute("pads", new long[0])).containsExactly(1, 2);
    assertThat(parsed.intsAttribute("missing", new long[] {0})).containsExactly(0);
    assertThat(parsed.floatsAttribute("scales", new float[0])).containsExactly(1f, 2f);
    assertThat(parsed.floatsAttribute("missing", new float[] {1f})).containsExactly(1f);
    assertThat(parsed.tensorAttribute("value").longs()).containsExactly(7, -8);
    assertThat(parsed.tensorAttribute("missing")).isNull();
    assertThat(parsed.attributes().get("untyped").type()).isEqualTo(OnnxAttribute.Type.INT);
    assertThat(parsed.attributes().get("untyped").i()).isEqualTo(5);
    assertThat(parsed.attributes().get("untypedStrings").type())
        .isEqualTo(OnnxAttribute.Type.STRINGS);
    assertThat(parsed.attributes().get("untypedStrings").strings()).containsExactly("a");
    assertThat(parsed.attributes().get("pads").toString()).isEqualTo("pads=[1, 2]");
    assertThat(parsed.attributes().get("alpha").toString()).isEqualTo("alpha=0.25");
    assertThat(parsed.attributes().get("mode").toString()).isEqualTo("mode=linear");
    assertThat(parsed.attributes().get("scales").toString()).isEqualTo("scales=[1.0, 2.0]");
    assertThat(parsed.attributes().get("value").toString()).startsWith("value=INT64");
    assertThat(parsed.attributes().get("group").toString()).isEqualTo("group=2");
  }

  @Test
  void decodesTensorPayloadsOfEveryStoredType() {
    ByteBuffer half = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN);
    half.putShort(Float.floatToFloat16(1.5f)).putShort(Float.floatToFloat16(-0.25f));
    ByteBuffer bf16 = ByteBuffer.allocate(2).order(ByteOrder.LITTLE_ENDIAN);
    bf16.putShort((short) (Float.floatToIntBits(2.0f) >>> 16));
    ByteBuffer doubles = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putDouble(0.75);
    ByteBuffer int32 = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putInt(-5).putInt(6);

    assertThat(rawTensor(OnnxDataType.FLOAT16, new long[] {2}, half.array()).floats())
        .containsExactly(1.5f, -0.25f);
    assertThat(rawTensor(OnnxDataType.BFLOAT16, new long[] {1}, bf16.array()).floats())
        .containsExactly(2.0f);
    Tensor rawDouble = rawTensor(OnnxDataType.DOUBLE, new long[] {1}, doubles.array());
    assertThat(rawDouble.type()).isEqualTo(OnnxDataType.DOUBLE);
    assertThat(rawDouble.doubles()).containsExactly(0.75);
    Tensor ints = rawTensor(OnnxDataType.INT32, new long[] {2}, int32.array());
    assertThat(ints.type()).isEqualTo(OnnxDataType.INT32);
    assertThat(ints.longs()).containsExactly(-5, 6);
    Tensor int8 = rawTensor(OnnxDataType.INT8, new long[] {3}, new byte[] {-1, 0, 1});
    assertThat(int8.bytes()).containsExactly(-1, 0, 1);
    assertThat(rawTensor(OnnxDataType.BOOL, new long[] {2}, new byte[] {1, 0}).type())
        .isEqualTo(OnnxDataType.BOOL);

    // Typed (non-raw) payload fields.
    Tensor typedFloat =
        tensor(new ProtobufWriter().packedVarints(1, 2).varint(2, 1).packedFloats(4, 1f, 2f));
    assertThat(typedFloat.floats()).containsExactly(1f, 2f);
    Tensor typedInt64 =
        tensor(new ProtobufWriter().packedVarints(1, 2).varint(2, 7).packedVarints(7, 3, -3));
    assertThat(typedInt64.longs()).containsExactly(3, -3);
    Tensor typedInt32 =
        tensor(new ProtobufWriter().packedVarints(1, 1).varint(2, 6).packedVarints(5, 9));
    assertThat(typedInt32.longs()).containsExactly(9);
    Tensor typedInt8 =
        tensor(new ProtobufWriter().packedVarints(1, 1).varint(2, 3).packedVarints(5, -2));
    assertThat(typedInt8.bytes()).containsExactly(-2);
    Tensor typedHalf =
        tensor(
            new ProtobufWriter()
                .packedVarints(1, 1)
                .varint(2, 10)
                .packedVarints(5, Float.floatToFloat16(3f) & 0xFFFF));
    assertThat(typedHalf.floats()).containsExactly(3f);
    Tensor typedDouble =
        tensor(new ProtobufWriter().packedVarints(1, 1).varint(2, 11).bytes(10, doubleBytes(1.25)));
    assertThat(typedDouble.doubles()).containsExactly(1.25);
    assertThat(typedDouble.floatAt(0)).isEqualTo(1.25f);
    Tensor typedBf16 =
        tensor(
            new ProtobufWriter()
                .packedVarints(1, 1)
                .varint(2, 16)
                .packedVarints(5, (Float.floatToIntBits(4f) >>> 16) & 0xFFFF));
    assertThat(typedBf16.floats()).containsExactly(4f);
    Tensor scalar = tensor(new ProtobufWriter().varint(2, 1).packedFloats(4, 9f));
    assertThat(scalar.rank()).isZero();
  }

  @Test
  void refusesWhatItCannotRepresent() {
    assertThatThrownBy(() -> rawTensor(OnnxDataType.STRING, new long[] {1}, new byte[] {1}))
        .isInstanceOf(UnsupportedOperationException.class);
    assertThatThrownBy(() -> rawTensor(OnnxDataType.INT8, new long[] {2}, new byte[] {1}))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(
            () ->
                tensor(
                    new ProtobufWriter()
                        .packedVarints(1, 1)
                        .varint(2, 1)
                        .string(8, "ext")
                        .message(13, new ProtobufWriter().string(1, "location"))))
        .isInstanceOf(UnsupportedOperationException.class)
        .hasMessageContaining("ext");
    assertThatThrownBy(() -> tensor(new ProtobufWriter().packedVarints(1, 1).varint(2, 8)))
        .isInstanceOf(UnsupportedOperationException.class);
    assertThatThrownBy(() -> OnnxModel.parse(new ProtobufWriter().varint(1, 8).toByteArray()))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("no graph");
    assertThat(OnnxDataType.fromCode(99)).isEqualTo(OnnxDataType.UNDEFINED);
    assertThat(OnnxDataType.INT64.code()).isEqualTo(7);
  }

  @Test
  void toleratesUnknownFieldsEverywhere() {
    ProtobufWriter graph =
        new ProtobufWriter()
            .varint(99, 1)
            .message(
                1,
                OnnxFixtures.node("Relu", "r", new String[] {"a"}, new String[] {"b"})
                    .varint(42, 1))
            .message(11, OnnxFixtures.valueInfo("a", OnnxDataType.FLOAT, 3).varint(50, 2))
            .message(12, OnnxFixtures.valueInfo("b", OnnxDataType.FLOAT, 3));
    byte[] bytes = new ProtobufWriter().varint(1, 9).varint(77, 3).message(7, graph).toByteArray();
    OnnxModel model = OnnxModel.parse(bytes);
    assertThat(model.graph().nodes()).hasSize(1);
    assertThat(model.defaultOpset()).isEqualTo(-1);
    assertThat(model.graph().inputs().get(0).dims()).containsExactly(3);
  }

  private static Tensor rawTensor(OnnxDataType type, long[] dims, byte[] raw) {
    return tensor(
        new ProtobufWriter()
            .packedVarints(1, dims)
            .varint(2, type.code())
            .string(8, "t")
            .bytes(9, raw));
  }

  private static Tensor tensor(ProtobufWriter proto) {
    return OnnxParser.parseTensor(new ProtobufReader(proto.toByteArray())).tensor();
  }

  private static byte[] doubleBytes(double value) {
    return ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putDouble(value).array();
  }
}
