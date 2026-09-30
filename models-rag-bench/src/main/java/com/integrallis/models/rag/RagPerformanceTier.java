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
package com.integrallis.models.rag;

/** Project-defined RAG service tier. */
public enum RagPerformanceTier {
  PRODUCTION_READY,
  USABLE,

  /**
   * Quality gates all pass and the model runs correctly, but its latency is over the USABLE SLO.
   *
   * <p>Split out of {@link #OFFLINE} because that tier was conflating "too slow for this SLO" with
   * "cannot be used at all", and the difference is most of the catalogue. Measured on the
   * small/medium campaign: of 28 models tiered OFFLINE, every one passed retrieval recall, citation
   * recall, abstention accuracy and correct-answer rate -- none of them were broken. Five were
   * rejected on the tail alone, with a ttft median under the 2000ms bound at 95% confidence and
   * only 3 to 7 of 27 requests over it; `gemma_3_4b_it` missed by 90ms (4.5%) while holding 2x
   * margin on every other metric, and landed in the same tier as a model 3x over.
   *
   * <p>A model here is shippable with a slower latency expectation, which for a small/medium-first
   * catalogue is a product category rather than a failure.
   */
  PERFORMANCE_REDUCED,

  /**
   * Retained for records written before {@link #PERFORMANCE_REDUCED} existed, and no longer
   * produced by {@link RagPerformancePolicy#classify}.
   *
   * <p>Not removed, because 316 certified records carry it and a published qualification cannot be
   * retracted. Those records are re-readable without re-running anything: an OFFLINE record whose
   * quality gates passed is a PERFORMANCE_REDUCED record under the current taxonomy, and the stored
   * summary has every number needed to say so.
   */
  OFFLINE,
  FAILED_QUALITY,
  FAILED_RUNTIME
}
