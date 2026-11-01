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

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.BiConsumer;

/**
 * Eager, node-by-node execution of a decoded ONNX graph.
 *
 * <p>Construction folds every node whose inputs are all constants (initializers or the outputs of
 * already-folded nodes) and caches the result, so weight dequantization, shape constants and the
 * like are computed once. {@code RandomNormalLike} is never folded. Each {@link #run} then walks
 * the remaining nodes in file order, which ONNX requires to be topological.
 */
public final class OnnxGraphExecutor {

  private final OnnxModel model;
  private final Map<String, Tensor> constants;
  private final List<OnnxNode> dynamicNodes;
  private final NoiseSource noise;

  /** Prepares {@code model} with a fixed-seed noise source. */
  public OnnxGraphExecutor(OnnxModel model) {
    this(model, NoiseSource.seeded(0));
  }

  /** Prepares {@code model}; {@code noise} answers every {@code RandomNormalLike} draw. */
  public OnnxGraphExecutor(OnnxModel model, NoiseSource noise) {
    this.model = Objects.requireNonNull(model, "model");
    this.noise = Objects.requireNonNull(noise, "noise");
    this.constants = new HashMap<>(model.graph().initializers());
    this.dynamicNodes = new ArrayList<>();
    for (OnnxNode node : model.graph().nodes()) {
      if (isFoldable(node)) {
        Tensor[] outputs = Operators.apply(node, resolve(node, constants), noise);
        store(node, outputs, constants);
      } else {
        dynamicNodes.add(node);
      }
    }
  }

  /** The decoded model this executor runs. */
  public OnnxModel model() {
    return model;
  }

  /** Nodes that run per call, after constant folding. */
  public int dynamicNodeCount() {
    return dynamicNodes.size();
  }

  /** Runs the graph for {@code feeds} and returns every declared graph output by name. */
  public Map<String, Tensor> run(Map<String, Tensor> feeds) {
    List<String> outputs = new ArrayList<>();
    for (OnnxValueInfo output : model.graph().outputs()) {
      outputs.add(output.name());
    }
    return run(feeds, outputs, null);
  }

  /**
   * Runs the graph and returns the requested value names, which may be intermediates. {@code
   * observer}, when given, sees every node output as it is produced; parity tests use it.
   */
  public Map<String, Tensor> run(
      Map<String, Tensor> feeds,
      Collection<String> requested,
      BiConsumer<String, Tensor> observer) {
    return run(feeds, requested, observer, noise);
  }

  /**
   * Runs the graph with {@code noiseSource} answering this call's {@code RandomNormalLike} draws
   * instead of the executor's own source; everything else is as {@link #run(Map, Collection,
   * BiConsumer)}.
   */
  public Map<String, Tensor> run(
      Map<String, Tensor> feeds,
      Collection<String> requested,
      BiConsumer<String, Tensor> observer,
      NoiseSource noiseSource) {
    Objects.requireNonNull(noiseSource, "noiseSource");
    for (OnnxValueInfo input : model.graph().feedInputs()) {
      if (!feeds.containsKey(input.name())) {
        throw new IllegalArgumentException("graph input '" + input.name() + "' was not fed");
      }
    }
    Map<String, Tensor> values = new HashMap<>(feeds);
    for (OnnxNode node : dynamicNodes) {
      Tensor[] inputs = resolve(node, values);
      Tensor[] outputs;
      try {
        outputs = Operators.apply(node, inputs, noiseSource);
      } catch (RuntimeException failure) {
        throw new IllegalStateException(
            describe(node, inputs) + ": " + failure.getMessage(), failure);
      }
      store(node, outputs, values);
      if (observer != null) {
        for (int i = 0; i < outputs.length && i < node.outputs().size(); i++) {
          observer.accept(node.outputs().get(i), outputs[i]);
        }
      }
    }
    Map<String, Tensor> result = new LinkedHashMap<>();
    for (String name : requested) {
      Tensor value = values.containsKey(name) ? values.get(name) : constants.get(name);
      if (value == null) {
        throw new IllegalArgumentException("'" + name + "' is not a value this graph produces");
      }
      result.put(name, value);
    }
    return result;
  }

  private boolean isFoldable(OnnxNode node) {
    if (node.opType().equals("RandomNormalLike")) {
      return false;
    }
    boolean any = false;
    for (String input : node.inputs()) {
      if (input.isEmpty()) {
        continue;
      }
      any = true;
      if (!constants.containsKey(input)) {
        return false;
      }
    }
    return any || node.opType().equals("Constant");
  }

  private Tensor[] resolve(OnnxNode node, Map<String, Tensor> values) {
    Tensor[] inputs = new Tensor[node.inputs().size()];
    for (int i = 0; i < inputs.length; i++) {
      String name = node.inputs().get(i);
      if (name.isEmpty()) {
        continue;
      }
      Tensor value = values.get(name);
      if (value == null) {
        value = constants.get(name);
      }
      if (value == null) {
        throw new IllegalStateException(
            node.label() + " needs '" + name + "', which nothing produced");
      }
      inputs[i] = value;
    }
    return inputs;
  }

  private static void store(OnnxNode node, Tensor[] outputs, Map<String, Tensor> values) {
    for (int i = 0; i < outputs.length && i < node.outputs().size(); i++) {
      String name = node.outputs().get(i);
      if (!name.isEmpty() && outputs[i] != null) {
        values.put(name, outputs[i]);
      }
    }
  }

  private static String describe(OnnxNode node, Tensor[] inputs) {
    StringBuilder text = new StringBuilder(node.label()).append(" inputs=[");
    for (int i = 0; i < inputs.length; i++) {
      text.append(i == 0 ? "" : ", ").append(inputs[i] == null ? "-" : inputs[i].toString());
    }
    return text.append(']').toString();
  }
}
