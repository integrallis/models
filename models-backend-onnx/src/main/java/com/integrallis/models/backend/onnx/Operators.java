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

/** Dispatch from an ONNX operator name to its kernel. One place lists what the module executes. */
final class Operators {

  private Operators() {}

  /**
   * Runs {@code node} on its resolved inputs ({@code null} where omitted) and returns its outputs.
   */
  static Tensor[] apply(OnnxNode node, Tensor[] in, NoiseSource noise) {
    return switch (node.opType()) {
      case "Identity" -> one(in[0]);
      case "Constant" -> one(constant(node));
      case "Add" -> one(Elementwise.arithmetic(Elementwise.Arithmetic.ADD, in[0], in[1]));
      case "Sub" -> one(Elementwise.arithmetic(Elementwise.Arithmetic.SUB, in[0], in[1]));
      case "Mul" -> one(Elementwise.arithmetic(Elementwise.Arithmetic.MUL, in[0], in[1]));
      case "Div" -> one(Elementwise.arithmetic(Elementwise.Arithmetic.DIV, in[0], in[1]));
      case "Pow" -> one(Elementwise.arithmetic(Elementwise.Arithmetic.POW, in[0], in[1]));
      case "Mod" ->
          one(
              Elementwise.arithmetic(
                  node.intAttribute("fmod", 0) == 1
                      ? Elementwise.Arithmetic.MOD_TRUNC
                      : Elementwise.Arithmetic.MOD_FLOOR,
                  in[0],
                  in[1]));
      case "Min" -> {
        Tensor result = in[0];
        for (int i = 1; i < in.length; i++) {
          result = Elementwise.arithmetic(Elementwise.Arithmetic.MIN, result, in[i]);
        }
        yield one(result);
      }
      case "Sqrt" -> one(Elementwise.sqrt(in[0]));
      case "Exp" -> one(Elementwise.exp(in[0]));
      case "Tanh" -> one(Elementwise.tanh(in[0]));
      case "Sigmoid" -> one(Elementwise.sigmoid(in[0]));
      case "Sin" -> one(Elementwise.sin(in[0]));
      case "Cos" -> one(Elementwise.cos(in[0]));
      case "Erf" -> one(Elementwise.erf(in[0]));
      case "Neg" -> one(Elementwise.neg(in[0]));
      case "Floor" -> one(Elementwise.floor(in[0]));
      case "Round" -> one(Elementwise.round(in[0]));
      case "IsNaN" -> one(Elementwise.isNaN(in[0]));
      case "LeakyRelu" -> one(Elementwise.leakyRelu(in[0], node.floatAttribute("alpha", 0.01f)));
      case "Clip" -> one(Elementwise.clip(in[0], input(in, 1), input(in, 2)));
      case "Cast" -> one(Elementwise.cast(in[0], node.intAttribute("to", 1)));
      case "Equal" -> one(Elementwise.compare(Elementwise.Comparison.EQUAL, in[0], in[1]));
      case "Less" -> one(Elementwise.compare(Elementwise.Comparison.LESS, in[0], in[1]));
      case "Greater" -> one(Elementwise.compare(Elementwise.Comparison.GREATER, in[0], in[1]));
      case "GreaterOrEqual" ->
          one(Elementwise.compare(Elementwise.Comparison.GREATER_OR_EQUAL, in[0], in[1]));
      case "And" -> one(Elementwise.compare(Elementwise.Comparison.AND, in[0], in[1]));
      case "Where" -> one(Elementwise.where(in[0], in[1], in[2]));
      case "Shape" ->
          one(
              Structural.shape(
                  in[0], node.intAttribute("start", 0), node.intAttribute("end", Long.MAX_VALUE)));
      case "Reshape" -> one(Structural.reshape(in[0], in[1]));
      case "Flatten" -> one(Structural.flatten(in[0], node.intAttribute("axis", 1)));
      case "Unsqueeze" -> one(Structural.unsqueeze(in[0], in[1]));
      case "Squeeze" -> one(Structural.squeeze(in[0], input(in, 1)));
      case "Transpose" -> one(Structural.transpose(in[0], node.intsAttribute("perm", new long[0])));
      case "Concat" -> one(Structural.concat(in, node.intAttribute("axis", 0)));
      case "Slice" -> one(Structural.slice(in[0], in[1], in[2], input(in, 3), input(in, 4)));
      case "Gather" -> one(Structural.gather(in[0], in[1], node.intAttribute("axis", 0)));
      case "GatherElements" ->
          one(Structural.gatherElements(in[0], in[1], node.intAttribute("axis", 0)));
      case "Expand" -> one(Structural.expand(in[0], in[1]));
      case "Tile" -> one(Structural.tile(in[0], in[1]));
      case "ConstantOfShape" ->
          one(Structural.constantOfShape(in[0], node.tensorAttribute("value")));
      case "Range" -> one(Structural.range(in[0], in[1], in[2]));
      case "ReduceSum" ->
          one(
              Reductions.reduceSum(
                  in[0],
                  input(in, 1),
                  node.intAttribute("keepdims", 1) == 1,
                  node.intAttribute("noop_with_empty_axes", 0) == 1));
      case "CumSum" ->
          one(
              Reductions.cumSum(
                  in[0],
                  in[1],
                  node.intAttribute("exclusive", 0) == 1,
                  node.intAttribute("reverse", 0) == 1));
      case "Softmax" -> one(Reductions.softmax(in[0], node.intAttribute("axis", -1)));
      case "RandomNormalLike" -> one(randomNormalLike(node, in[0], noise));
      case "MatMul" -> one(Linear.matMul(in[0], in[1]));
      case "Gemm" ->
          one(
              Linear.gemm(
                  in[0],
                  in[1],
                  input(in, 2),
                  node.floatAttribute("alpha", 1f),
                  node.floatAttribute("beta", 1f),
                  node.intAttribute("transA", 0) == 1,
                  node.intAttribute("transB", 0) == 1));
      case "Conv" ->
          one(
              Convolution.conv(
                  in[0],
                  in[1],
                  input(in, 2),
                  node.intsAttribute("dilations", new long[0]),
                  node.intAttribute("group", 1),
                  node.intsAttribute("pads", new long[0]),
                  node.intsAttribute("strides", new long[0])));
      case "ConvTranspose" ->
          one(
              Convolution.convTranspose(
                  in[0],
                  in[1],
                  input(in, 2),
                  node.intsAttribute("dilations", new long[0]),
                  node.intAttribute("group", 1),
                  node.intsAttribute("output_padding", new long[0]),
                  node.intsAttribute("pads", new long[0]),
                  node.intsAttribute("strides", new long[0])));
      case "LayerNormalization" ->
          Normalization.layerNorm(
              in[0],
              in[1],
              input(in, 2),
              node.intAttribute("axis", -1),
              node.floatAttribute("epsilon", 1e-5f));
      case "InstanceNormalization" ->
          one(
              Normalization.instanceNorm(
                  in[0], in[1], input(in, 2), node.floatAttribute("epsilon", 1e-5f)));
      case "LSTM" ->
          Recurrent.lstm(
              in[0],
              in[1],
              in[2],
              input(in, 3),
              input(in, 4),
              input(in, 5),
              input(in, 6),
              node.stringAttribute("direction", "forward"),
              (int) node.intAttribute("hidden_size", 0));
      case "Resize" ->
          one(
              Resampling.resizeNearest(
                  in[0],
                  input(in, 2),
                  input(in, 3),
                  node.stringAttribute("mode", "nearest"),
                  node.stringAttribute("coordinate_transformation_mode", "half_pixel"),
                  node.stringAttribute("nearest_mode", "round_prefer_floor")));
      case "Pad" ->
          one(Resampling.pad(in[0], in[1], input(in, 2), node.stringAttribute("mode", "constant")));
      case "DequantizeLinear" ->
          one(
              Quantization.dequantizeLinear(
                  in[0], in[1], input(in, 2), node.intAttribute("axis", 1)));
      default ->
          throw new UnsupportedOperationException(
              "operator " + node.opType() + " (" + node.label() + ") is not implemented");
    };
  }

  private static Tensor[] one(Tensor t) {
    return new Tensor[] {t};
  }

  private static Tensor input(Tensor[] in, int index) {
    return index < in.length ? in[index] : null;
  }

  private static Tensor constant(OnnxNode node) {
    Tensor value = node.tensorAttribute("value");
    if (value != null) {
      return value;
    }
    OnnxAttribute floatValue = node.attributes().get("value_float");
    if (floatValue != null) {
      return Tensor.scalar(floatValue.f());
    }
    OnnxAttribute intValue = node.attributes().get("value_int");
    if (intValue != null) {
      return Tensor.scalar(intValue.i());
    }
    OnnxAttribute floats = node.attributes().get("value_floats");
    if (floats != null) {
      return Tensor.ofFloats(new long[] {floats.floats().length}, floats.floats());
    }
    OnnxAttribute ints = node.attributes().get("value_ints");
    if (ints != null) {
      return Tensor.vector(ints.ints());
    }
    throw new UnsupportedOperationException(node.label() + " has no supported value attribute");
  }

  private static Tensor randomNormalLike(OnnxNode node, Tensor like, NoiseSource noise) {
    Tensor sample = noise.sample(node.outputs().get(0), like.shape());
    float mean = node.floatAttribute("mean", 0f);
    float scale = node.floatAttribute("scale", 1f);
    if (mean == 0f && scale == 1f) {
      return sample;
    }
    float[] values = sample.floats().clone();
    for (int i = 0; i < values.length; i++) {
      values[i] = values[i] * scale + mean;
    }
    return Tensor.ofFloats(sample.shape(), values);
  }
}
