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

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class OnnxGraphExecutorTest {

  @Test
  void runsAGraphAndFoldsWhatIsConstant() {
    // c = Constant; s = Add(w, c) is constant and folds; y = Mul(x, s) runs per call.
    byte[] bytes =
        OnnxFixtures.model(
            List.of(
                OnnxFixtures.node(
                    "Constant",
                    new String[0],
                    new String[] {"c"},
                    OnnxFixtures.attributeTensor(
                        "value", OnnxFixtures.floatTensor("", new long[] {2}, 1f, 1f))),
                OnnxFixtures.node("Add", new String[] {"w", "c"}, new String[] {"s"}),
                OnnxFixtures.node("Mul", new String[] {"x", "s"}, new String[] {"y"})),
            List.of(OnnxFixtures.floatTensor("w", new long[] {2}, 0.5f, 1.5f)),
            List.of(OnnxFixtures.valueInfo("x", OnnxDataType.FLOAT, 2)),
            List.of(OnnxFixtures.valueInfo("y", OnnxDataType.FLOAT, 2)));
    OnnxGraphExecutor executor = new OnnxGraphExecutor(OnnxModel.parse(bytes));
    assertThat(executor.dynamicNodeCount()).isEqualTo(1);
    assertThat(executor.model().graph().name()).isEqualTo("fixture");

    Map<String, Tensor> out =
        executor.run(Map.of("x", Tensor.ofFloats(new long[] {2}, new float[] {2, 2})));
    assertThat(out).containsOnlyKeys("y");
    assertThat(out.get("y").floats()).containsExactly(3f, 5f);

    List<String> seen = new ArrayList<>();
    Map<String, Tensor> both =
        executor.run(
            Map.of("x", Tensor.ofFloats(new long[] {2}, new float[] {1, 1})),
            List.of("y", "s"),
            (name, tensor) -> seen.add(name));
    assertThat(both.get("s").floats()).containsExactly(1.5f, 2.5f);
    assertThat(seen).containsExactly("y");
  }

  @Test
  void reportsMissingFeedsUnknownOutputsAndUnsupportedOperators() {
    OnnxGraphExecutor executor = new OnnxGraphExecutor(OnnxModel.parse(OnnxFixtures.addModel()));
    assertThatThrownBy(() -> executor.run(Map.of()))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("'x' was not fed");
    Map<String, Tensor> feeds = Map.of("x", Tensor.ofFloats(new long[] {1, 2}, new float[] {1, 2}));
    assertThatThrownBy(() -> executor.run(feeds, List.of("nope"), null))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("nope");
    assertThat(executor.run(feeds).get("y").floats()).containsExactly(1.5f, 0.5f);

    byte[] unsupported =
        OnnxFixtures.model(
            List.of(OnnxFixtures.node("Loop", new String[] {"x"}, new String[] {"y"})),
            List.of(),
            List.of(OnnxFixtures.valueInfo("x", OnnxDataType.FLOAT, 1)),
            List.of(OnnxFixtures.valueInfo("y", OnnxDataType.FLOAT, 1)));
    OnnxGraphExecutor loop = new OnnxGraphExecutor(OnnxModel.parse(unsupported));
    assertThatThrownBy(() -> loop.run(Map.of("x", Tensor.scalar(1f))))
        .isInstanceOf(IllegalStateException.class)
        .hasMessageContaining("Loop")
        .hasMessageContaining("not implemented");

    byte[] dangling =
        OnnxFixtures.model(
            List.of(OnnxFixtures.node("Add", new String[] {"x", "ghost"}, new String[] {"y"})),
            List.of(),
            List.of(OnnxFixtures.valueInfo("x", OnnxDataType.FLOAT, 1)),
            List.of(OnnxFixtures.valueInfo("y", OnnxDataType.FLOAT, 1)));
    OnnxGraphExecutor ghost = new OnnxGraphExecutor(OnnxModel.parse(dangling));
    assertThatThrownBy(() -> ghost.run(Map.of("x", Tensor.scalar(1f))))
        .isInstanceOf(IllegalStateException.class)
        .hasMessageContaining("ghost");
  }

  @Test
  void drawsNoiseFromTheConfiguredSourceAndNeverFoldsIt() {
    byte[] bytes =
        OnnxFixtures.model(
            List.of(
                OnnxFixtures.node(
                    "RandomNormalLike",
                    new String[] {"w"},
                    new String[] {"n"},
                    OnnxFixtures.attributeInt("dtype", 1)),
                OnnxFixtures.node("Add", new String[] {"n", "w"}, new String[] {"y"})),
            List.of(OnnxFixtures.floatTensor("w", new long[] {3}, 1f, 2f, 3f)),
            List.of(),
            List.of(OnnxFixtures.valueInfo("y", OnnxDataType.FLOAT, 3)));
    OnnxModel model = OnnxModel.parse(bytes);
    OnnxGraphExecutor seeded = new OnnxGraphExecutor(model);
    assertThat(seeded.dynamicNodeCount()).isEqualTo(2);
    float[] first = seeded.run(Map.of()).get("y").floats();
    float[] second =
        new OnnxGraphExecutor(model, NoiseSource.seeded(0)).run(Map.of()).get("y").floats();
    assertThat(first).containsExactly(second);
    float[] other =
        new OnnxGraphExecutor(model, NoiseSource.seeded(1)).run(Map.of()).get("y").floats();
    assertThat(other).isNotEqualTo(first);

    Tensor injected = Tensor.ofFloats(new long[] {3}, new float[] {0.5f, 0.5f, 0.5f});
    OnnxGraphExecutor exact =
        new OnnxGraphExecutor(model, NoiseSource.injected(Map.of("n", injected)));
    assertThat(exact.run(Map.of()).get("y").floats()).containsExactly(1.5f, 2.5f, 3.5f);
    OnnxGraphExecutor wrongShape =
        new OnnxGraphExecutor(model, NoiseSource.injected(Map.of("n", Tensor.scalar(1f))));
    assertThatThrownBy(() -> wrongShape.run(Map.of()))
        .isInstanceOf(IllegalStateException.class)
        .hasMessageContaining("shape");
    OnnxGraphExecutor missing = new OnnxGraphExecutor(model, NoiseSource.injected(Map.of()));
    assertThatThrownBy(() -> missing.run(Map.of()))
        .isInstanceOf(IllegalStateException.class)
        .hasMessageContaining("no injected noise");
    OnnxGraphExecutor fallback =
        new OnnxGraphExecutor(model, NoiseSource.injected(Map.of(), NoiseSource.seeded(0)));
    assertThat(fallback.run(Map.of()).get("y").floats()).containsExactly(first);
    Map<String, Tensor> overridden =
        seeded.run(Map.of(), List.of("y"), null, NoiseSource.injected(Map.of("n", injected)));
    assertThat(overridden.get("y").floats()).containsExactly(1.5f, 2.5f, 3.5f);
  }

  @Test
  void scalesNoiseByMeanAndScaleAndReadsEveryConstantSpelling() {
    byte[] bytes =
        OnnxFixtures.model(
            List.of(
                OnnxFixtures.node(
                    "RandomNormalLike",
                    new String[] {"w"},
                    new String[] {"n"},
                    OnnxFixtures.attributeFloat("mean", 10f),
                    OnnxFixtures.attributeFloat("scale", 2f)),
                OnnxFixtures.node(
                    "Constant",
                    new String[0],
                    new String[] {"cf"},
                    OnnxFixtures.attributeFloat("value_float", 1.5f)),
                OnnxFixtures.node(
                    "Constant",
                    new String[0],
                    new String[] {"ci"},
                    OnnxFixtures.attributeInt("value_int", 4)),
                OnnxFixtures.node(
                    "Constant",
                    new String[0],
                    new String[] {"cfs"},
                    OnnxFixtures.attributeFloats("value_floats", 1f, 2f)),
                OnnxFixtures.node(
                    "Constant",
                    new String[0],
                    new String[] {"cis"},
                    OnnxFixtures.attributeInts("value_ints", 3, 4)),
                OnnxFixtures.node("Clip", new String[] {"n", "", "cf"}, new String[] {"y"})),
            List.of(OnnxFixtures.floatTensor("w", new long[] {2}, 0f, 0f)),
            List.of(),
            List.of(OnnxFixtures.valueInfo("y", OnnxDataType.FLOAT, 2)));
    Tensor unit = Tensor.ofFloats(new long[] {2}, new float[] {-1f, 1f});
    OnnxGraphExecutor executor =
        new OnnxGraphExecutor(OnnxModel.parse(bytes), NoiseSource.injected(Map.of("n", unit)));
    Map<String, Tensor> out =
        executor.run(Map.of(), List.of("y", "n", "cf", "ci", "cfs", "cis"), null);
    assertThat(out.get("n").floats()).containsExactly(8f, 12f);
    assertThat(out.get("y").floats()).containsExactly(1.5f, 1.5f);
    assertThat(out.get("cf").floats()).containsExactly(1.5f);
    assertThat(out.get("ci").longs()).containsExactly(4);
    assertThat(out.get("cfs").floats()).containsExactly(1f, 2f);
    assertThat(out.get("cis").longs()).containsExactly(3, 4);

    byte[] empty =
        OnnxFixtures.model(
            List.of(OnnxFixtures.node("Constant", new String[0], new String[] {"c"})),
            List.of(),
            List.of(),
            List.of(OnnxFixtures.valueInfo("c", OnnxDataType.FLOAT, 1)));
    assertThatThrownBy(() -> new OnnxGraphExecutor(OnnxModel.parse(empty)))
        .isInstanceOf(UnsupportedOperationException.class);
  }

  @Test
  void dispatchesEveryElementwiseStructuralAndReductionOperator() {
    // Each node exercises the dispatcher's wiring and default attributes, not the kernels.
    List<ProtobufWriter> nodes =
        List.of(
            OnnxFixtures.node("Identity", new String[] {"x"}, new String[] {"id"}),
            OnnxFixtures.node("Sub", new String[] {"x", "x"}, new String[] {"sub"}),
            OnnxFixtures.node("Div", new String[] {"x", "x"}, new String[] {"div"}),
            OnnxFixtures.node("Pow", new String[] {"x", "x"}, new String[] {"pow"}),
            OnnxFixtures.node("Mod", new String[] {"i", "i"}, new String[] {"mod"}),
            OnnxFixtures.node(
                "Mod",
                new String[] {"x", "x"},
                new String[] {"fmod"},
                OnnxFixtures.attributeInt("fmod", 1)),
            OnnxFixtures.node("Min", new String[] {"x", "x", "x"}, new String[] {"min"}),
            OnnxFixtures.node("Sqrt", new String[] {"x"}, new String[] {"sqrt"}),
            OnnxFixtures.node("Exp", new String[] {"x"}, new String[] {"exp"}),
            OnnxFixtures.node("Tanh", new String[] {"x"}, new String[] {"tanh"}),
            OnnxFixtures.node("Sigmoid", new String[] {"x"}, new String[] {"sigmoid"}),
            OnnxFixtures.node("Sin", new String[] {"x"}, new String[] {"sin"}),
            OnnxFixtures.node("Cos", new String[] {"x"}, new String[] {"cos"}),
            OnnxFixtures.node("Erf", new String[] {"x"}, new String[] {"erf"}),
            OnnxFixtures.node("Neg", new String[] {"x"}, new String[] {"neg"}),
            OnnxFixtures.node("Floor", new String[] {"x"}, new String[] {"floor"}),
            OnnxFixtures.node("Round", new String[] {"x"}, new String[] {"round"}),
            OnnxFixtures.node("IsNaN", new String[] {"x"}, new String[] {"isnan"}),
            OnnxFixtures.node("LeakyRelu", new String[] {"x"}, new String[] {"leaky"}),
            OnnxFixtures.node(
                "Cast",
                new String[] {"x"},
                new String[] {"cast"},
                OnnxFixtures.attributeInt("to", 7)),
            OnnxFixtures.node("Equal", new String[] {"x", "x"}, new String[] {"eq"}),
            OnnxFixtures.node("Less", new String[] {"x", "x"}, new String[] {"lt"}),
            OnnxFixtures.node("Greater", new String[] {"x", "x"}, new String[] {"gt"}),
            OnnxFixtures.node("GreaterOrEqual", new String[] {"x", "x"}, new String[] {"ge"}),
            OnnxFixtures.node("And", new String[] {"eq", "ge"}, new String[] {"and"}),
            OnnxFixtures.node("Where", new String[] {"lt", "x", "x"}, new String[] {"where"}),
            OnnxFixtures.node("Shape", new String[] {"x"}, new String[] {"shape"}),
            OnnxFixtures.node("Reshape", new String[] {"x", "shape"}, new String[] {"reshape"}),
            OnnxFixtures.node("Flatten", new String[] {"x"}, new String[] {"flatten"}),
            OnnxFixtures.node("Unsqueeze", new String[] {"x", "zero"}, new String[] {"unsq"}),
            OnnxFixtures.node("Squeeze", new String[] {"unsq"}, new String[] {"sq"}),
            OnnxFixtures.node("Transpose", new String[] {"x"}, new String[] {"tr"}),
            OnnxFixtures.node(
                "Concat",
                new String[] {"x", "x"},
                new String[] {"cat"},
                OnnxFixtures.attributeInt("axis", 0)),
            OnnxFixtures.node("Slice", new String[] {"x", "zero", "one"}, new String[] {"slice"}),
            OnnxFixtures.node("Gather", new String[] {"x", "zero"}, new String[] {"gather"}),
            OnnxFixtures.node("GatherElements", new String[] {"x", "zero"}, new String[] {"ge2"}),
            OnnxFixtures.node("Expand", new String[] {"x", "shape"}, new String[] {"expand"}),
            OnnxFixtures.node("ConstantOfShape", new String[] {"shape"}, new String[] {"cos2"}),
            OnnxFixtures.node(
                "Range", new String[] {"zero0", "two0", "one0"}, new String[] {"range"}),
            OnnxFixtures.node("ReduceSum", new String[] {"x"}, new String[] {"rsum"}),
            OnnxFixtures.node("CumSum", new String[] {"x", "zero0"}, new String[] {"csum"}),
            OnnxFixtures.node("Softmax", new String[] {"x"}, new String[] {"soft"}));
    List<ProtobufWriter> initializers =
        List.of(
            OnnxFixtures.int64Tensor("i", new long[] {2}, 7, 3),
            OnnxFixtures.int64Tensor("zero", new long[] {1}, 0),
            OnnxFixtures.int64Tensor("one", new long[] {1}, 1),
            OnnxFixtures.int64Tensor("zero0", new long[0], 0),
            OnnxFixtures.int64Tensor("one0", new long[0], 1),
            OnnxFixtures.int64Tensor("two0", new long[0], 2));
    List<String> outputs =
        List.of(
            "id", "sub", "div", "pow", "mod", "fmod", "min", "sqrt", "exp", "tanh", "sigmoid",
            "sin", "cos", "erf", "neg", "floor", "round", "isnan", "leaky", "cast", "eq", "lt",
            "gt", "ge", "and", "where", "shape", "reshape", "flatten", "unsq", "sq", "tr", "cat",
            "slice", "gather", "ge2", "expand", "cos2", "range", "rsum", "csum", "soft");
    List<ProtobufWriter> outputInfos = new ArrayList<>();
    for (String name : outputs) {
      outputInfos.add(OnnxFixtures.valueInfo(name, OnnxDataType.FLOAT, 2));
    }
    byte[] bytes =
        OnnxFixtures.model(
            nodes,
            initializers,
            List.of(OnnxFixtures.valueInfo("x", OnnxDataType.FLOAT, 2)),
            outputInfos);
    OnnxGraphExecutor executor = new OnnxGraphExecutor(OnnxModel.parse(bytes));
    Map<String, Tensor> out =
        executor.run(Map.of("x", Tensor.ofFloats(new long[] {2}, new float[] {1f, 4f})));
    assertThat(out).containsKeys(outputs.toArray(new String[0]));
    assertThat(out.get("sub").floats()).containsExactly(0f, 0f);
    assertThat(out.get("mod").longs()).containsExactly(0, 0);
    assertThat(out.get("fmod").floats()).containsExactly(0f, 0f);
    assertThat(out.get("min").floats()).containsExactly(1f, 4f);
    assertThat(out.get("cast").longs()).containsExactly(1, 4);
    assertThat(out.get("and").bytes()).containsExactly(1, 1);
    assertThat(out.get("where").floats()).containsExactly(1f, 4f);
    assertThat(out.get("flatten").shape()).containsExactly(2, 1);
    assertThat(out.get("sq").shape()).containsExactly(2);
    assertThat(out.get("cat").shape()).containsExactly(4);
    assertThat(out.get("slice").floats()).containsExactly(1f);
    assertThat(out.get("gather").floats()).containsExactly(1f);
    assertThat(out.get("ge2").floats()).containsExactly(1f);
    assertThat(out.get("cos2").floats()).containsExactly(0f, 0f);
    assertThat(out.get("range").longs()).containsExactly(0, 1);
    assertThat(out.get("rsum").floats()).containsExactly(5f);
    assertThat(out.get("csum").floats()).containsExactly(1f, 5f);
    assertThat(out.get("leaky").floats()).containsExactly(1f, 4f);
  }
}
