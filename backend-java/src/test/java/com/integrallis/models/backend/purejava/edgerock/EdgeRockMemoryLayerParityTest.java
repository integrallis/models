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
package com.integrallis.models.backend.purejava.edgerock;

import static org.assertj.core.api.Assertions.assertThat;

import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonToken;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Checks this memory layer against EdgeRock's own implementation.
 *
 * <p>The fixture is produced by {@code experiments/edgerock-decoder/parity/dump_memory_layer.py},
 * which instantiates the reference {@code IndependentGQAMemory} out of {@code modeling_edgerock.py}
 * at small dimensions with seeded weights and runs seven tokens through it. Seven tokens against a
 * commit chunk of two means the bank commits three times, so the expected output depends on
 * admission, eviction and the read together rather than on one of them.
 *
 * <p>Two earlier fixtures were discarded for being worthless, which is worth recording. The first
 * dumped a real layer of the 4B checkpoint: 848 MB, and with three tokens against the real chunk of
 * 256 the bank never committed, so every expected value was zero and the test would have passed
 * whatever the read did. The second kept the reference's zero-initialised {@code write_down_proj},
 * {@code admission_up} and {@code read_scale}, which make an untrained layer the identity -- it
 * would only have proved that zero times anything is zero.
 *
 * <p>The comparison is also across code paths on purpose. The reference ran all seven tokens in one
 * call, which takes its sequence path and reduces admission over a balanced tree; this runs one
 * token at a time through the decode path, which merges once per commit. They agree only if the
 * admission rule really is associative.
 */
@Tag("unit")
class EdgeRockMemoryLayerParityTest {

  private static final String FIXTURE = "/edgerock/memory-layer-parity.json";

  /** Float32 across two languages and two reduction orders; this is the agreement we can claim. */
  private static final float TOLERANCE = 2.0e-4f;

  @Test
  void reproducesTheReferenceAcrossThreeCommitBoundaries() throws IOException {
    Fixture fixture = Fixture.load();
    int tokens = fixture.config("tokens");
    int hidden = fixture.config("hidden");
    int ropeWidth = fixture.config("ropeWidth");

    EdgeRockConfig config = fixture.edgeRockConfig();
    EdgeRockMemoryLayer layer = new EdgeRockMemoryLayer(config, fixture.weights());
    MemoryBank bank =
        new MemoryBank(
            config.headCountKv(),
            config.memoryCapacity(),
            config.memoryCommitChunk(),
            config.headDim());

    float[] out = new float[hidden];
    for (int token = 0; token < tokens; token++) {
      float[] hiddenState = fixture.slice("hidden", token * hidden, hidden);
      float[] cos = fixture.slice("cos", token * ropeWidth, ropeWidth);
      float[] sin = fixture.slice("sin", token * ropeWidth, ropeWidth);

      layer.forward(hiddenState, token, cos, sin, bank, out);

      float[] expected = fixture.slice("output", token * hidden, hidden);
      assertThat(out)
          .describedAs("token %d of %d, after %d commits", token, tokens, token / 2)
          .usingComparatorWithPrecision(TOLERANCE)
          .containsExactly(expected);
    }
  }

  @Test
  void theFirstTokenReadsNothingAndTheLaterOnesDoNot() {
    // Guards the fixture itself. If a future regeneration produced all zeros again -- the bank
    // never committing -- the parity test above would pass while checking nothing.
    Fixture fixture = Fixture.load();
    int hidden = fixture.config("hidden");
    float[] first = fixture.slice("output", 0, hidden);
    assertThat(first).usingComparatorWithPrecision(1.0e-9f).containsOnly(0f);

    int nonZero = 0;
    float[] all = fixture.array("output");
    for (float value : all) {
      if (Math.abs(value) > 1.0e-9f) {
        nonZero++;
      }
    }
    assertThat(nonZero)
        .describedAs("the fixture must exercise a committed bank, not just an empty one")
        .isGreaterThan(all.length / 2);
  }

  /** Minimal reader: jackson-core streaming, since databind is not on this module's path. */
  private record Fixture(Map<String, float[]> arrays, Map<String, Integer> config) {

    static Fixture load() {
      Map<String, float[]> arrays = new HashMap<>();
      Map<String, Integer> config = new HashMap<>();
      try (InputStream stream = Fixture.class.getResourceAsStream(FIXTURE)) {
        if (stream == null) {
          throw new IllegalStateException("missing fixture " + FIXTURE);
        }
        JsonParser parser = new JsonFactory().createParser(stream);
        read(parser, arrays, config, "");
      } catch (IOException error) {
        throw new IllegalStateException("could not read " + FIXTURE, error);
      }
      return new Fixture(arrays, config);
    }

    private static void read(
        JsonParser parser, Map<String, float[]> arrays, Map<String, Integer> config, String prefix)
        throws IOException {
      String field = null;
      while (parser.nextToken() != null) {
        JsonToken token = parser.currentToken();
        if (token == JsonToken.FIELD_NAME) {
          field = parser.currentName();
        } else if (token == JsonToken.START_ARRAY) {
          List<Float> values = new ArrayList<>();
          while (parser.nextToken() != JsonToken.END_ARRAY) {
            values.add(parser.getFloatValue());
          }
          float[] out = new float[values.size()];
          for (int i = 0; i < out.length; i++) {
            out[i] = values.get(i);
          }
          arrays.put(prefix.isEmpty() ? field : prefix + "." + field, out);
        } else if (token == JsonToken.START_OBJECT && field != null) {
          read(parser, arrays, config, "config".equals(field) ? "config" : field);
        } else if (token == JsonToken.END_OBJECT) {
          return;
        } else if (token == JsonToken.VALUE_NUMBER_INT && "config".equals(prefix)) {
          config.put(field, parser.getIntValue());
        }
      }
    }

    int config(String name) {
      Integer value = config.get(name);
      if (value == null) {
        throw new IllegalStateException(
            "fixture config has no " + name + "; have " + config.keySet());
      }
      return value;
    }

    float[] array(String name) {
      float[] value = arrays.get(name);
      if (value == null) {
        throw new IllegalStateException(
            "fixture has no array " + name + "; have " + arrays.keySet());
      }
      return value;
    }

    float[] slice(String name, int offset, int length) {
      float[] source = array(name);
      float[] out = new float[length];
      System.arraycopy(source, offset, out, 0, length);
      return out;
    }

    EdgeRockConfig edgeRockConfig() {
      int heads = config("heads");
      int groups = config("groups");
      return new EdgeRockConfig(
          4,
          config("hidden"),
          16,
          heads,
          groups,
          config("dim"),
          32,
          128,
          1.0e-6f,
          10_000f,
          config("ropeWidth") * 2,
          List.of(),
          4,
          8,
          groups,
          2,
          8,
          4,
          config("window"),
          config("capacity"),
          config("chunk"),
          config("rank"),
          config("writer"));
    }

    EdgeRockMemoryLayer.Weights weights() {
      int hidden = config("hidden");
      int heads = config("heads");
      int groups = config("groups");
      int dim = config("dim");
      int writer = config("writer");
      int rank = config("rank");
      return new EdgeRockMemoryLayer.Weights(
          array("weights.input_norm.weight"),
          projection("weights.q_proj.weight", heads * dim, hidden),
          array("weights.q_norm.weight"),
          projection("weights.k_proj.weight", groups * dim, hidden),
          array("weights.k_norm.weight"),
          projection("weights.v_proj.weight", groups * dim, hidden),
          projection("weights.o_proj.weight", hidden, heads * dim),
          projection("weights.write_gate_proj.weight", writer, hidden),
          projection("weights.write_up_proj.weight", writer, hidden),
          projection("weights.write_down_proj.weight", hidden, writer),
          projection("weights.admission_down.weight", rank, hidden),
          projection("weights.admission_up.weight", groups, rank),
          rows("weights.read_gate_weight", heads, dim),
          array("weights.read_gate_bias"),
          array("weights.read_scale"));
    }

    /** PyTorch stores nn.Linear as [out, in], so row r dotted with the input gives out[r]. */
    private EdgeRockMemoryLayer.Projection projection(String name, int rows, int columns) {
      float[] weight = array(name);
      if (weight.length != rows * columns) {
        throw new IllegalStateException(
            name + " holds " + weight.length + " but " + rows + "x" + columns + " was expected");
      }
      return (input, output) -> {
        for (int row = 0; row < rows; row++) {
          float sum = 0f;
          int base = row * columns;
          for (int column = 0; column < columns; column++) {
            sum += weight[base + column] * input[column];
          }
          output[row] = sum;
        }
      };
    }

    private float[][] rows(String name, int rows, int columns) {
      float[] flat = array(name);
      float[][] out = new float[rows][columns];
      for (int row = 0; row < rows; row++) {
        System.arraycopy(flat, row * columns, out[row], 0, columns);
      }
      return out;
    }
  }
}
