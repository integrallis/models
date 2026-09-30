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
package com.integrallis.models.backend.purejava.ops;

/**
 * Expert selection shared by every mixture-of-experts decoder in this backend.
 *
 * <p>One implementation on purpose. Qwen3-MoE on the Llama path and Qwen3.5-MoE on the hybrid path
 * route identically -- llama.cpp's {@code build_moe_ffn} with softmax gating and {@code norm_w} --
 * and a second copy would be a second thing to get subtly wrong.
 */
public final class ExpertRouting {

  private ExpertRouting() {}

  /**
   * Picks the highest-scoring experts, weighting them by a softmax over <b>all</b> experts.
   *
   * <p>The difference from {@link #selectExperts} is whether the chosen weights are renormalised,
   * and it is not cosmetic. Renormalised, they sum to one and the routed output is a weighted
   * average. Unnormalised, they sum to whatever mass the chosen experts held -- typically well
   * under one -- so the routed branch contributes proportionally less when the router is unsure.
   * That is a deliberate property of the architecture, not an oversight.
   *
   * <p>llama.cpp selects between the two on {@code expert_weights_norm}, whose default is false.
   * Qwen3.5-MoE and Qwen3-MoE set it; DeepSeek-V2 publishes no such key and therefore does not.
   * Using the renormalised form here would scale every routed contribution up by a factor that
   * varies per token, which reads as a plausible model that is subtly wrong rather than as a
   * failure.
   *
   * <p>Note the softmax is over the full width here, so it cannot be folded into the selection the
   * way {@link #selectExperts} folds it.
   *
   * @param routerLogits one logit per expert. <b>Overwritten</b> with the softmax over all of them
   * @param experts how many experts the layer holds
   * @param used how many of them a token selects
   * @param selected receives the chosen expert indices, highest-scoring first
   * @param routingWeights receives their weights, which do <b>not</b> sum to one
   */
  public static void selectExpertsWithoutRenormalisation(
      float[] routerLogits, int experts, int used, int[] selected, float[] routingWeights) {
    TensorOps.softmax(routerLogits, 0, experts);
    // Over the probabilities rather than the logits: softmax is monotone, so the ranking is the
    // same,
    // but the weights that come out are the full-width ones and not a renormalised subset.
    int filled = 0;
    for (int expert = 0; expert < experts; expert++) {
      float probability = routerLogits[expert];
      if (filled == used && probability <= routingWeights[used - 1]) {
        continue;
      }
      int slot = Math.min(filled, used - 1);
      while (slot > 0 && routingWeights[slot - 1] < probability) {
        routingWeights[slot] = routingWeights[slot - 1];
        selected[slot] = selected[slot - 1];
        slot--;
      }
      routingWeights[slot] = probability;
      selected[slot] = expert;
      if (filled < used) {
        filled++;
      }
    }
  }

  /**
   * Picks the highest-scoring experts and turns their logits into routing weights summing to one.
   *
   * <p>The reference softmaxes over all experts, takes the top {@code used}, then renormalises
   * those. Softmaxing over only the selected logits is algebraically identical -- the full-width
   * denominator is a constant that cancels -- so that is what this does: one softmax over {@code
   * used} values rather than over every expert and a division. Omitting the renormalisation would
   * leave the weights summing to whatever mass the chosen experts held, a silent per-token scale
   * error.
   *
   * <p>Insertion into a descending list rather than a sort of all experts: eight of 256 costs at
   * most a few thousand comparisons and touches none of the rest. Ties keep the lower expert index,
   * which is what {@code torch.topk} yields; exact ties between float logits do not arise in
   * practice, but the tie-break has to be decided somewhere or the output is unreproducible for the
   * input that hits it.
   *
   * @param routerLogits one logit per expert
   * @param experts how many experts the layer holds
   * @param used how many of them a token selects; must not exceed {@code experts}
   * @param selected receives the chosen expert indices, highest-scoring first
   * @param routingWeights receives their weights, summing to one
   */
  public static void selectExperts(
      float[] routerLogits, int experts, int used, int[] selected, float[] routingWeights) {
    int filled = 0;
    for (int expert = 0; expert < experts; expert++) {
      float logit = routerLogits[expert];
      if (filled == used && logit <= routingWeights[used - 1]) {
        continue;
      }
      // With the list full this overwrites the last entry, which is the eviction.
      int slot = Math.min(filled, used - 1);
      while (slot > 0 && routingWeights[slot - 1] < logit) {
        routingWeights[slot] = routingWeights[slot - 1];
        selected[slot] = selected[slot - 1];
        slot--;
      }
      routingWeights[slot] = logit;
      selected[slot] = expert;
      if (filled < used) {
        filled++;
      }
    }
    TensorOps.softmax(routingWeights, 0, used);
  }
}
