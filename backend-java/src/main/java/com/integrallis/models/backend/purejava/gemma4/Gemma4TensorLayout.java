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
package com.integrallis.models.backend.purejava.gemma4;

import com.integrallis.models.backend.purejava.gguf.GgufFile;
import com.integrallis.models.backend.purejava.gguf.GgufTensorInfo;
import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/** Validated resident and streamed tensor layout for a Gemma 4 GGUF. */
public final class Gemma4TensorLayout {

  /** A contiguous tensor range in the source GGUF. */
  public record TensorSlice(
      String tensorName, GgufTensorType type, long fileOffset, long byteSize, long[] shape) {

    public TensorSlice {
      Objects.requireNonNull(tensorName, "tensorName");
      Objects.requireNonNull(type, "type");
      Objects.requireNonNull(shape, "shape");
      shape = shape.clone();
      if (fileOffset < 0) {
        throw new IllegalArgumentException("fileOffset must be >= 0: " + fileOffset);
      }
      if (byteSize <= 0) {
        throw new IllegalArgumentException("byteSize must be > 0: " + byteSize);
      }
    }

    @Override
    public long[] shape() {
      return shape.clone();
    }
  }

  /** The two source ranges required to execute one routed expert. */
  public record ExpertWeights(int layer, int expert, TensorSlice gateUp, TensorSlice down) {
    public ExpertWeights {
      Objects.requireNonNull(gateUp, "gateUp");
      Objects.requireNonNull(down, "down");
    }

    public long totalBytes() {
      return Math.addExact(gateUp.byteSize(), down.byteSize());
    }
  }

  /** Per-layer routed tensor descriptors. */
  public static final class LayerLayout {
    private final int layer;
    private final int numExperts;
    private final TensorDescriptor gateUp;
    private final TensorDescriptor down;
    private final GgufTensorInfo downScale;

    private LayerLayout(
        int layer,
        int numExperts,
        TensorDescriptor gateUp,
        TensorDescriptor down,
        GgufTensorInfo downScale) {
      this.layer = layer;
      this.numExperts = numExperts;
      this.gateUp = gateUp;
      this.down = down;
      this.downScale = downScale;
    }

    public int layer() {
      return layer;
    }

    public GgufTensorInfo downScale() {
      return downScale;
    }

    public ExpertWeights expert(int expert) {
      if (expert < 0 || expert >= numExperts) {
        throw new IllegalArgumentException("expert out of range: " + expert);
      }
      return new ExpertWeights(layer, expert, gateUp.slice(expert), down.slice(expert));
    }
  }

  private record TensorDescriptor(
      String name,
      GgufTensorType type,
      long absoluteOffset,
      long bytesPerExpert,
      long[] expertShape) {

    private TensorDescriptor {
      expertShape = expertShape.clone();
    }

    private TensorSlice slice(int expert) {
      long offset = Math.addExact(absoluteOffset, Math.multiplyExact(expert, bytesPerExpert));
      return new TensorSlice(name, type, offset, bytesPerExpert, expertShape);
    }
  }

  private final List<LayerLayout> layers;
  private final List<GgufTensorInfo> residentTensors;
  private final long routedExpertBytes;
  private final long residentBytes;

  private Gemma4TensorLayout(
      List<LayerLayout> layers,
      List<GgufTensorInfo> residentTensors,
      long routedExpertBytes,
      long residentBytes) {
    this.layers = List.copyOf(layers);
    this.residentTensors = List.copyOf(residentTensors);
    this.routedExpertBytes = routedExpertBytes;
    this.residentBytes = residentBytes;
  }

  public static Gemma4TensorLayout fromGgufFile(GgufFile file, Gemma4Config config) {
    Objects.requireNonNull(file, "file");
    return fromTensorInfos(file.tensorDataOffset(), config, file.tensorInfos());
  }

  public static Gemma4TensorLayout fromTensorInfos(
      long tensorDataOffset, Gemma4Config config, List<GgufTensorInfo> tensorInfos) {
    if (tensorDataOffset < 0) {
      throw new IllegalArgumentException("tensorDataOffset must be >= 0: " + tensorDataOffset);
    }
    Objects.requireNonNull(config, "config");
    Objects.requireNonNull(tensorInfos, "tensorInfos");

    Map<String, GgufTensorInfo> tensorsByName = indexByName(tensorInfos);
    Set<String> streamedNames = new HashSet<>();
    List<LayerLayout> layers = new ArrayList<>(config.numLayers());

    for (int layer = 0; layer < config.numLayers(); layer++) {
      String prefix = "blk." + layer + ".";
      GgufTensorInfo gateUp = requiredTensor(tensorsByName, prefix + "ffn_gate_up_exps.weight");
      requireShape(
          gateUp,
          new long[] {config.embeddingDim(), 2L * config.expertHiddenDim(), config.numExperts()});
      requireType(gateUp, Set.of(GgufTensorType.Q4_K));

      GgufTensorInfo down = requiredTensor(tensorsByName, prefix + "ffn_down_exps.weight");
      requireShape(
          down, new long[] {config.expertHiddenDim(), config.embeddingDim(), config.numExperts()});
      requireType(down, Set.of(GgufTensorType.Q5_0, GgufTensorType.Q8_0));

      GgufTensorInfo downScale = requiredTensor(tensorsByName, prefix + "ffn_down_exps.scale");
      requireShape(downScale, new long[] {config.numExperts()});
      requireType(downScale, Set.of(GgufTensorType.F32));

      streamedNames.add(gateUp.name());
      streamedNames.add(down.name());
      layers.add(
          new LayerLayout(
              layer,
              config.numExperts(),
              descriptor(tensorDataOffset, gateUp, config.numExperts()),
              descriptor(tensorDataOffset, down, config.numExperts()),
              downScale));
    }

    for (GgufTensorInfo tensor : tensorInfos) {
      if (tensor.name().contains("_exps.weight") && !streamedNames.contains(tensor.name())) {
        throw new IllegalArgumentException(
            "Unsupported Gemma 4 routed-expert tensor: " + tensor.name());
      }
    }

    List<GgufTensorInfo> resident = new ArrayList<>();
    long routedBytes = 0;
    long residentBytes = 0;
    for (GgufTensorInfo tensor : tensorInfos) {
      if (streamedNames.contains(tensor.name())) {
        routedBytes = Math.addExact(routedBytes, tensor.byteSize());
      } else {
        resident.add(tensor);
        residentBytes = Math.addExact(residentBytes, tensor.byteSize());
      }
    }
    return new Gemma4TensorLayout(layers, resident, routedBytes, residentBytes);
  }

  public LayerLayout layer(int layer) {
    if (layer < 0 || layer >= layers.size()) {
      throw new IllegalArgumentException("layer out of range: " + layer);
    }
    return layers.get(layer);
  }

  public List<GgufTensorInfo> residentTensors() {
    return residentTensors;
  }

  public long routedExpertBytes() {
    return routedExpertBytes;
  }

  public long residentBytes() {
    return residentBytes;
  }

  private static Map<String, GgufTensorInfo> indexByName(List<GgufTensorInfo> tensorInfos) {
    Map<String, GgufTensorInfo> result = new HashMap<>(tensorInfos.size() * 2);
    for (GgufTensorInfo tensor : tensorInfos) {
      Objects.requireNonNull(tensor, "tensorInfos element");
      if (result.putIfAbsent(tensor.name(), tensor) != null) {
        throw new IllegalArgumentException("Duplicate GGUF tensor: " + tensor.name());
      }
    }
    return result;
  }

  private static GgufTensorInfo requiredTensor(Map<String, GgufTensorInfo> tensors, String name) {
    GgufTensorInfo tensor = tensors.get(name);
    if (tensor == null) {
      throw new IllegalArgumentException("Tensor not found: " + name);
    }
    return tensor;
  }

  private static void requireShape(GgufTensorInfo tensor, long[] expected) {
    long[] actual = tensor.shape();
    if (!Arrays.equals(expected, actual)) {
      throw new IllegalArgumentException(
          tensor.name()
              + " shape must be "
              + Arrays.toString(expected)
              + ", found "
              + Arrays.toString(actual));
    }
  }

  private static void requireType(GgufTensorInfo tensor, Set<GgufTensorType> expected) {
    if (!expected.contains(tensor.type())) {
      throw new IllegalArgumentException(
          tensor.name() + " type must be one of " + expected + ", found " + tensor.type());
    }
  }

  private static TensorDescriptor descriptor(
      long tensorDataOffset, GgufTensorInfo tensor, int numExperts) {
    long totalBytes = tensor.byteSize();
    if (totalBytes % numExperts != 0) {
      throw new IllegalArgumentException(
          tensor.name() + " byte size is not divisible by " + numExperts + ": " + totalBytes);
    }
    long[] shape = tensor.shape();
    return new TensorDescriptor(
        tensor.name(),
        tensor.type(),
        Math.addExact(tensorDataOffset, tensor.offset()),
        totalBytes / numExperts,
        new long[] {shape[0], shape[1]});
  }
}
