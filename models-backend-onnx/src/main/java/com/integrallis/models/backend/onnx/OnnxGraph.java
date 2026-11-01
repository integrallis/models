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
import java.util.List;
import java.util.Map;

/**
 * A decoded {@code GraphProto}: nodes in file order, initializers by name, and value signatures.
 *
 * @param name the graph name
 * @param nodes the nodes in the order the file lists them, which ONNX requires to be topological
 * @param initializers constant tensors by value name
 * @param inputs declared inputs, which may include initializers on older exporters
 * @param outputs declared outputs
 * @param valueInfos declared intermediate value signatures
 */
public record OnnxGraph(
    String name,
    List<OnnxNode> nodes,
    Map<String, Tensor> initializers,
    List<OnnxValueInfo> inputs,
    List<OnnxValueInfo> outputs,
    List<OnnxValueInfo> valueInfos) {

  /** Defensive copies. */
  public OnnxGraph {
    nodes = List.copyOf(nodes);
    initializers = Map.copyOf(initializers);
    inputs = List.copyOf(inputs);
    outputs = List.copyOf(outputs);
    valueInfos = List.copyOf(valueInfos);
  }

  /** The inputs a caller must feed: declared inputs that are not also initializers. */
  public List<OnnxValueInfo> feedInputs() {
    List<OnnxValueInfo> feeds = new ArrayList<>();
    for (OnnxValueInfo input : inputs) {
      if (!initializers.containsKey(input.name())) {
        feeds.add(input);
      }
    }
    return List.copyOf(feeds);
  }
}
