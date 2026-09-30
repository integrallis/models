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
package com.integrallis.models.backend.purejava;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.models.backend.purejava.gemma4.Gemma4Config;
import com.integrallis.models.backend.purejava.gguf.GgufFile;
import com.integrallis.models.backend.purejava.gguf.GgufParser;
import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.gguf.SyntheticGgufBuilder;
import com.integrallis.models.backend.purejava.plan.ModelTopology;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Planning topology for a Gemma 4 E-series model, whose later layers share another layer's
 * key-value cache.
 *
 * <p>A sharing layer carries no {@code attn_k} or {@code attn_v} of its own. The planner read those
 * names for every layer regardless, and refused Gemma 4 E4B outright with "Tensor not found:
 * blk.24.attn_k.weight". The forward pass had been taught about shared key-value layers; the
 * planner sitting in front of it had not, and nothing exercised the planner on an E-series model
 * because the E-series tests build the graph directly.
 *
 * <p>The fixture carries only the tensors the planner reads. It is not a loadable model and is not
 * meant to be: the defect is entirely in which tensor names are asked for.
 */
@Tag("unit")
class Gemma4SharedKvTopologyTest {

  private static final int DIM = 4;
  private static final int LAYERS = 3;

  /**
   * Layers 0 and 1 own a cache; layer 2 shares one.
   *
   * <p>{@code kvOwningLayers = 3 - 1 = 2}, and layer 2 is sliding, so it reads {@code 2 - 2 = 0}.
   * Layer 0's tensors are given a type nothing else uses, which is what makes "reports the right
   * source layer" distinguishable from "reports some layer that happens to exist".
   */
  private static Gemma4Config sharedKvConfig() {
    return new Gemma4Config(
        DIM,
        LAYERS,
        1,
        List.of(1, 1, 1),
        DIM,
        DIM,
        DIM,
        DIM,
        DIM,
        8,
        3,
        0,
        0,
        0,
        1_000_000.0f,
        10_000.0f,
        DIM,
        DIM,
        1.0e-6f,
        2,
        List.of(true, false, true),
        30.0f,
        /* sharedKvLayers= */ 1,
        /* perLayerEmbeddingDim= */ 2);
  }

  @Test
  void aSharingLayerReportsTheKeyValueTypesOfTheLayerItReadsFrom() {
    Gemma4Config config = sharedKvConfig();
    assertThat(config.ownsKvCache(0)).isTrue();
    assertThat(config.ownsKvCache(1)).isTrue();
    assertThat(config.ownsKvCache(2))
        .describedAs("the fixture must actually share, or this tests nothing")
        .isFalse();
    assertThat(config.kvSourceLayer(2)).isEqualTo(0);

    GgufFile file = sharedKvFile();

    // Reaching past this line is the regression: blk.2 has no attn_k or attn_v at all.
    ModelTopology topology = PureJavaBackend.gemma4Topology(file, config);

    assertThat(topology.layers()).hasSize(LAYERS);
    // Layer 0 owns its cache, and its key/value tensors carry the distinguishing type.
    assertThat(topology.layers().get(0).key()).isEqualTo(GgufTensorType.Q8_0);
    assertThat(topology.layers().get(0).value()).isEqualTo(GgufTensorType.Q8_0);
    // Layer 1 owns its own, which is F32, so the two owners are distinguishable.
    assertThat(topology.layers().get(1).key()).isEqualTo(GgufTensorType.F32);
    // Layer 2 shares layer 0's, so it must report layer 0's types and not layer 1's or its own.
    assertThat(topology.layers().get(2).key())
        .describedAs("a sharing layer reports the key tensor it will actually multiply")
        .isEqualTo(GgufTensorType.Q8_0);
    assertThat(topology.layers().get(2).value()).isEqualTo(GgufTensorType.Q8_0);
    // Its own query and feed-forward are still its own.
    assertThat(topology.layers().get(2).query()).isEqualTo(GgufTensorType.F32);
    assertThat(topology.layers().get(2).gate()).isEqualTo(GgufTensorType.F32);
  }

  /** Only the tensors the planner reads, with layer 0's key and value in a distinguishing type. */
  private static GgufFile sharedKvFile() {
    SyntheticGgufBuilder builder =
        new SyntheticGgufBuilder().addString("general.architecture", "gemma4");
    for (int layer = 0; layer < LAYERS; layer++) {
      String prefix = "blk." + layer + ".";
      matrix(builder, prefix + "attn_q.weight", GgufTensorType.F32);
      matrix(builder, prefix + "attn_output.weight", GgufTensorType.F32);
      matrix(builder, prefix + "ffn_gate.weight", GgufTensorType.F32);
      matrix(builder, prefix + "ffn_up.weight", GgufTensorType.F32);
      matrix(builder, prefix + "ffn_down.weight", GgufTensorType.F32);
      if (layer == 2) {
        // The whole point: a sharing layer publishes no key or value projection.
        continue;
      }
      GgufTensorType kvType = layer == 0 ? GgufTensorType.Q8_0 : GgufTensorType.F32;
      matrix(builder, prefix + "attn_k.weight", kvType);
      if (layer == 0) {
        // Sliding layers carry a value projection; layer 1 is not sliding.
        matrix(builder, prefix + "attn_v.weight", kvType);
      }
    }
    byte[] data = builder.build();
    MemorySegment segment = Arena.ofAuto().allocate(data.length);
    MemorySegment.copy(data, 0, segment, ValueLayout.JAVA_BYTE, 0, data.length);
    return GgufParser.parseSegment(segment);
  }

  private static void matrix(SyntheticGgufBuilder builder, String name, GgufTensorType type) {
    if (type == GgufTensorType.Q8_0) {
      // One 32-weight Q8_0 block per row: a 2-byte scale and 32 codes.
      builder.addTensor(name, type, new long[] {32, DIM}, new byte[DIM * 34]);
      return;
    }
    ByteBuffer buffer = ByteBuffer.allocate(DIM * DIM * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    for (int index = 0; index < DIM * DIM; index++) {
      buffer.putFloat(0.01f * index);
    }
    builder.addTensor(name, type, new long[] {DIM, DIM}, buffer.array());
  }
}
