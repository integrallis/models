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

import java.util.List;
import java.util.Map;

/**
 * One graph node, decoded from {@code NodeProto}.
 *
 * <p>An empty string in {@link #inputs()} is ONNX's spelling of an omitted optional input and is
 * kept in place so positional semantics survive.
 *
 * @param name the node name, possibly empty
 * @param opType the operator name
 * @param domain the operator set domain; empty means {@code ai.onnx}
 * @param inputs input value names in operator order
 * @param outputs output value names in operator order
 * @param attributes attributes by name
 */
public record OnnxNode(
    String name,
    String opType,
    String domain,
    List<String> inputs,
    List<String> outputs,
    Map<String, OnnxAttribute> attributes) {

  /** Defensive copies of the lists and map. */
  public OnnxNode {
    inputs = List.copyOf(inputs);
    outputs = List.copyOf(outputs);
    attributes = Map.copyOf(attributes);
  }

  /** Input {@code index}, or {@code null} when absent or omitted. */
  public String input(int index) {
    if (index >= inputs.size()) {
      return null;
    }
    String value = inputs.get(index);
    return value.isEmpty() ? null : value;
  }

  /** Whether input {@code index} is present and not the empty placeholder. */
  public boolean hasInput(int index) {
    return input(index) != null;
  }

  /** Integer attribute or the default. */
  public long intAttribute(String attribute, long defaultValue) {
    OnnxAttribute found = attributes.get(attribute);
    return found == null ? defaultValue : found.i();
  }

  /** Float attribute or the default. */
  public float floatAttribute(String attribute, float defaultValue) {
    OnnxAttribute found = attributes.get(attribute);
    return found == null ? defaultValue : found.f();
  }

  /** String attribute or the default. */
  public String stringAttribute(String attribute, String defaultValue) {
    OnnxAttribute found = attributes.get(attribute);
    return found == null ? defaultValue : found.s();
  }

  /** Integer-list attribute, or the default when absent. */
  public long[] intsAttribute(String attribute, long[] defaultValue) {
    OnnxAttribute found = attributes.get(attribute);
    return found == null ? defaultValue.clone() : found.ints();
  }

  /** Float-list attribute, or the default when absent. */
  public float[] floatsAttribute(String attribute, float[] defaultValue) {
    OnnxAttribute found = attributes.get(attribute);
    return found == null ? defaultValue.clone() : found.floats();
  }

  /** Tensor attribute, or {@code null} when absent. */
  public Tensor tensorAttribute(String attribute) {
    OnnxAttribute found = attributes.get(attribute);
    return found == null ? null : found.t();
  }

  /** A short label for error messages: the op type and, when present, the node name. */
  public String label() {
    return name.isEmpty() ? opType : opType + " '" + name + "'";
  }
}
