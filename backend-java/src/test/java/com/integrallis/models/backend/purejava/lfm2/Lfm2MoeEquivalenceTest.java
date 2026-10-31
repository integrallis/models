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
package com.integrallis.models.backend.purejava.lfm2;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.models.backend.purejava.gguf.GgufFile;
import com.integrallis.models.backend.purejava.gguf.GgufParser;
import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.gguf.SyntheticGgufBuilder;
import java.lang.foreign.Arena;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Routes a toy {@code lfm2moe} whose experts are <b>identical copies of the dense feed-forward</b>,
 * and requires it to produce the dense model's logits.
 *
 * <p><b>Why this shape.</b> A routed feed-forward with N identical experts and renormalised weights
 * summing to one computes exactly what one dense feed-forward computes: {@code sum_i w_i * F(x)} is
 * {@code F(x)} when {@code sum_i w_i = 1} and every {@code F} is the same function. So the dense
 * model is an oracle for the routed one, with no hand-computed expectation to get wrong and no
 * dependence on which experts the router happens to pick.
 *
 * <p>That matters because it isolates the three things a first MoE port actually gets wrong — the
 * expert tensor slice shapes, the accumulation into the output, and the weight normalisation — from
 * the arithmetic of SwiGLU, which the dense path already covers. If the slices were transposed, or
 * the accumulation overwrote instead of summing, or the weights did not sum to one, the two models
 * would differ here while both still producing finite, plausible logits.
 *
 * <p>Found with it: the real {@code LFM2.5-8B-A1B-Q4_0.gguf} port generated {@code " France is
 * France is France is"} where llama.cpp at {@code 3f03b1772} generated {@code " Paris. The city of
 * Paris is known for its beautiful landmarks"}. A 4.8 GB artifact is a poor place to localise that;
 * four experts of width six is a good one.
 */
@Tag("unit")
class Lfm2MoeEquivalenceTest {

  private static final int DIM = Lfm2ToyModel.DIM;
  private static final int LAYERS = Lfm2ToyModel.LAYERS;
  private static final int HIDDEN = Lfm2ToyModel.HIDDEN;
  private static final int VOCAB = Lfm2ToyModel.VOCAB;
  private static final int EXPERTS = 4;
  private static final int EXPERTS_USED = 2;
  private static final int LEADING_DENSE = 1;

  @Test
  void aRoutedModelWithIdenticalExpertsReproducesTheDenseModelsLogits() {
    GgufFile dense = Lfm2ToyModel.file(3, 7);
    GgufFile routed = parseMoe(moeBytes(3, 7));
    {
      Lfm2Config denseConfig = Lfm2Config.fromMetadata(dense.metadata());
      Lfm2Config routedConfig = Lfm2Config.fromMetadata(routed.metadata());

      assertThat(denseConfig.isMixtureOfExperts()).isFalse();
      assertThat(routedConfig.isMixtureOfExperts()).isTrue();
      // The routed model must actually route on the layers under test, or this proves nothing.
      assertThat(routedConfig.usesExperts(0)).isFalse();
      assertThat(routedConfig.usesExperts(1)).isTrue();
      assertThat(routedConfig.usesExperts(2)).isTrue();

      float[] denseLogits =
          Lfm2ForwardPass.fromGgufFile(dense, denseConfig, 8).forward(1, 0).clone();
      float[] routedLogits =
          Lfm2ForwardPass.fromGgufFile(routed, routedConfig, 8).forward(1, 0).clone();

      assertThat(routedLogits).hasSize(VOCAB);
      for (int token = 0; token < VOCAB; token++) {
        assertThat(routedLogits[token])
            .describedAs(
                "logit %d: routed %s against dense %s",
                token, routedLogits[token], denseLogits[token])
            .isCloseTo(denseLogits[token], org.assertj.core.data.Offset.offset(1.0e-4f));
      }
    }
  }

  @Test
  void theRoutedModelIsNotTriviallyEqualToTheDenseOne() {
    // A negative control. If the routed branch silently fell through to the dense tensors -- which
    // are also present on a routed layer in this fixture -- the test above would pass without the
    // expert path running at all. Give one expert a different weight and require the logits to
    // move: that can only happen if the router selected it and the expert matrices were read.
    {
      GgufFile sameExperts = parseMoe(moeBytes(3, 7));
      GgufFile oddExpert = parseMoe(moeBytes(3, 7, true));

      float[] same =
          Lfm2ForwardPass.fromGgufFile(
                  sameExperts, Lfm2Config.fromMetadata(sameExperts.metadata()), 8)
              .forward(1, 0)
              .clone();
      float[] different =
          Lfm2ForwardPass.fromGgufFile(oddExpert, Lfm2Config.fromMetadata(oddExpert.metadata()), 8)
              .forward(1, 0)
              .clone();

      assertThat(different)
          .describedAs("perturbing one expert must change the logits, or the experts never ran")
          .isNotEqualTo(same);
    }
  }

  /** Parses the synthetic bytes the same way {@link Lfm2ToyModel} does. */
  private static GgufFile parseMoe(byte[] data) {
    java.lang.foreign.MemorySegment segment = Arena.ofAuto().allocate(data.length);
    java.lang.foreign.MemorySegment.copy(
        data, 0, segment, java.lang.foreign.ValueLayout.JAVA_BYTE, 0, data.length);
    return GgufParser.parseSegment(segment);
  }

  private static byte[] moeBytes(int convKernelSeed, int attentionKeySeed) {
    return moeBytes(convKernelSeed, attentionKeySeed, false);
  }

  /**
   * The toy model as {@code lfm2moe}: dense layer 0, routed layers 1 and 2, every expert a copy of
   * that layer's dense feed-forward.
   *
   * @param perturbOneExpert give expert 0 of layer 1 different weights, for the negative control
   */
  private static byte[] moeBytes(
      int convKernelSeed, int attentionKeySeed, boolean perturbOneExpert) {
    SyntheticGgufBuilder builder =
        new SyntheticGgufBuilder()
            .addString("general.architecture", "lfm2moe")
            .addUint32("lfm2moe.embedding_length", DIM)
            .addUint32("lfm2moe.block_count", LAYERS)
            .addUint32("lfm2moe.attention.head_count", Lfm2ToyModel.HEADS)
            .addUint32("lfm2moe.context_length", 64)
            .addUint32("lfm2moe.feed_forward_length", HIDDEN)
            .addUint32("lfm2moe.vocab_size", VOCAB)
            .addUint32("lfm2moe.shortconv.l_cache", Lfm2ToyModel.L_CACHE)
            .addFloat32("lfm2moe.attention.layer_norm_rms_epsilon", 1.0e-5f)
            .addFloat32("lfm2moe.rope.freq_base", 1_000_000.0f)
            .addUint32("lfm2moe.expert_count", EXPERTS)
            .addUint32("lfm2moe.expert_used_count", EXPERTS_USED)
            .addUint32("lfm2moe.expert_feed_forward_length", HIDDEN)
            .addUint32("lfm2moe.leading_dense_block_count", LEADING_DENSE)
            .addUint32("lfm2moe.expert_gating_func", 2);

    List<Integer> kvHeads = new ArrayList<>();
    for (int layer = 0; layer < LAYERS; layer++) {
      kvHeads.add(Lfm2ToyModel.ATTENTION_LAYERS.contains(layer) ? Lfm2ToyModel.KV_HEADS : 0);
    }
    builder.addInt32Array("lfm2moe.attention.head_count_kv", kvHeads);

    builder.addTensor(
        "token_embd.weight",
        GgufTensorType.F32,
        new long[] {DIM, VOCAB},
        Lfm2ToyModel.f32(Lfm2ToyModel.matrix(VOCAB, DIM, 5)));
    builder.addTensor(
        "token_embd_norm.weight",
        GgufTensorType.F32,
        new long[] {DIM},
        Lfm2ToyModel.f32(Lfm2ToyModel.norm(9)));

    for (int layer = 0; layer < LAYERS; layer++) {
      String prefix = "blk." + layer + ".";
      builder.addTensor(
          prefix + "attn_norm.weight",
          GgufTensorType.F32,
          new long[] {DIM},
          Lfm2ToyModel.f32(Lfm2ToyModel.norm(11 + layer)));
      builder.addTensor(
          prefix + "ffn_norm.weight",
          GgufTensorType.F32,
          new long[] {DIM},
          Lfm2ToyModel.f32(Lfm2ToyModel.norm(21 + layer)));

      float[] gate = Lfm2ToyModel.matrix(HIDDEN, DIM, 31 + layer);
      float[] up = Lfm2ToyModel.matrix(HIDDEN, DIM, 41 + layer);
      float[] down = Lfm2ToyModel.matrix(DIM, HIDDEN, 51 + layer);

      if (layer < LEADING_DENSE) {
        builder.addTensor(
            prefix + "ffn_gate.weight",
            GgufTensorType.F32,
            new long[] {DIM, HIDDEN},
            Lfm2ToyModel.f32(gate));
        builder.addTensor(
            prefix + "ffn_up.weight",
            GgufTensorType.F32,
            new long[] {DIM, HIDDEN},
            Lfm2ToyModel.f32(up));
        builder.addTensor(
            prefix + "ffn_down.weight",
            GgufTensorType.F32,
            new long[] {HIDDEN, DIM},
            Lfm2ToyModel.f32(down));
      } else {
        // A router whose rows are all equal gives every expert the same logit, so the top-k is
        // decided by the tie-break and the renormalised weights are 1/EXPERTS_USED each. With
        // identical experts the mixture is the dense function whichever pair wins.
        builder.addTensor(
            prefix + "ffn_gate_inp.weight",
            GgufTensorType.F32,
            new long[] {DIM, EXPERTS},
            Lfm2ToyModel.f32(new float[EXPERTS * DIM]));
        builder.addTensor(
            prefix + "ffn_gate_exps.weight",
            GgufTensorType.F32,
            new long[] {DIM, HIDDEN, EXPERTS},
            Lfm2ToyModel.f32(repeat(gate, EXPERTS, perturbOneExpert && layer == LEADING_DENSE)));
        builder.addTensor(
            prefix + "ffn_up_exps.weight",
            GgufTensorType.F32,
            new long[] {DIM, HIDDEN, EXPERTS},
            Lfm2ToyModel.f32(repeat(up, EXPERTS, false)));
        builder.addTensor(
            prefix + "ffn_down_exps.weight",
            GgufTensorType.F32,
            new long[] {HIDDEN, DIM, EXPERTS},
            Lfm2ToyModel.f32(repeat(down, EXPERTS, false)));
      }
      Lfm2ToyModel.addMixerTensors(builder, layer, prefix, convKernelSeed, attentionKeySeed);
    }
    return builder.build();
  }

  /** One matrix repeated per expert, optionally perturbing expert 0 for the negative control. */
  private static float[] repeat(float[] matrix, int experts, boolean perturbFirst) {
    float[] all = new float[matrix.length * experts];
    for (int expert = 0; expert < experts; expert++) {
      System.arraycopy(matrix, 0, all, expert * matrix.length, matrix.length);
      if (perturbFirst && expert == 0) {
        for (int index = 0; index < matrix.length; index++) {
          all[index] = matrix[index] + 0.25f;
        }
      }
    }
    return all;
  }
}
