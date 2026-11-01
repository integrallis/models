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
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.TreeMap;
import java.util.TreeSet;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Reads the pinned Paradee-8M v1.2 graphs when {@code MODELS_PARADEE_ONNX_DIR} points at a
 * directory holding {@code onnx_paradee.onnx} and {@code onnx_paradee_int8.onnx}, and checks the
 * decoded structure against the artifact triage (2,117 nodes over 53 operator types in fp32; 2,535
 * over 56 in int8; inputs {@code input_ids}, {@code speed}, {@code voice}; output {@code
 * waveform}).
 */
@Tag("integration")
class ParadeeGraphStructureIntegrationTest {

  @Test
  void decodesTheFp32GraphAsTheTriageDescribesIt() throws Exception {
    OnnxModel model = load("onnx_paradee.onnx");
    OnnxGraph graph = model.graph();
    assertThat(model.irVersion()).isEqualTo(8);
    assertThat(model.defaultOpset()).isEqualTo(17);
    assertThat(graph.nodes()).hasSize(2117);
    assertThat(operatorHistogram(graph)).hasSize(53);
    assertThat(graph.feedInputs())
        .extracting(OnnxValueInfo::name)
        .containsExactly("input_ids", "speed", "voice");
    assertThat(graph.outputs()).extracting(OnnxValueInfo::name).containsExactly("waveform");
    long elements = graph.initializers().values().stream().mapToLong(Tensor::size).sum();
    assertThat(elements).isEqualTo(8_581_449L);
    describe("fp32", graph);
  }

  @Test
  void decodesTheInt8GraphAsTheTriageDescribesIt() throws Exception {
    OnnxGraph graph = load("onnx_paradee_int8.onnx").graph();
    assertThat(graph.nodes()).hasSize(2535);
    assertThat(operatorHistogram(graph)).hasSize(56);
    long elements = graph.initializers().values().stream().mapToLong(Tensor::size).sum();
    assertThat(elements).isEqualTo(7_867_227L);
    long int8 =
        graph.initializers().values().stream()
            .filter(t -> t.type() == OnnxDataType.INT8)
            .mapToLong(Tensor::size)
            .sum();
    assertThat(int8).isEqualTo(7_795_388L);
    describe("int8", graph);
  }

  private static OnnxModel load(String file) throws Exception {
    String configured = System.getenv("MODELS_PARADEE_ONNX_DIR");
    assumeTrue(configured != null, "MODELS_PARADEE_ONNX_DIR not set");
    Path path = Path.of(configured, file);
    assumeTrue(Files.isRegularFile(path), path + " missing");
    return OnnxModel.load(path);
  }

  private static Map<String, Integer> operatorHistogram(OnnxGraph graph) {
    Map<String, Integer> histogram = new TreeMap<>();
    for (OnnxNode node : graph.nodes()) {
      histogram.merge(node.opType(), 1, Integer::sum);
    }
    return histogram;
  }

  /** Prints what an executor has to implement: per operator, the attribute names it carries. */
  private static void describe(String label, OnnxGraph graph) {
    Map<String, TreeSet<String>> attributes = new TreeMap<>();
    Map<String, TreeSet<String>> attributeValues = new TreeMap<>();
    for (OnnxNode node : graph.nodes()) {
      TreeSet<String> names = attributes.computeIfAbsent(node.opType(), k -> new TreeSet<>());
      for (OnnxAttribute attribute : node.attributes().values()) {
        names.add(attribute.name());
        if (attribute.type() != OnnxAttribute.Type.TENSOR) {
          attributeValues
              .computeIfAbsent(node.opType() + "." + attribute.name(), k -> new TreeSet<>())
              .add(attribute.toString());
        }
      }
    }
    System.out.println("PARADEE_GRAPH " + label + " operators=" + operatorHistogram(graph));
    attributes.forEach(
        (op, names) ->
            System.out.println("PARADEE_GRAPH " + label + " " + op + " attributes=" + names));
    attributeValues.forEach(
        (key, values) -> {
          String rendered = values.size() > 8 ? values.size() + " distinct" : values.toString();
          System.out.println("PARADEE_GRAPH " + label + " " + key + " values=" + rendered);
        });
    Map<OnnxDataType, Integer> types = new TreeMap<>();
    for (Tensor t : graph.initializers().values()) {
      types.merge(t.type(), 1, Integer::sum);
    }
    System.out.println("PARADEE_GRAPH " + label + " initializerTypes=" + types);
  }
}
