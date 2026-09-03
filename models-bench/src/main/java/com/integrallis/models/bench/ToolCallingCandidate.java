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
package com.integrallis.models.bench;

import com.integrallis.models.runtime.chat.ChatTemplate;
import java.util.Arrays;
import java.util.Locale;

/** Exact artifacts admitted to the small-model tool-calling qualification round. */
enum ToolCallingCandidate {
  NEEDLE2(
      "needle2",
      "cactus_compute_needle2_cact_cq2_mixed",
      "Cactus Compute Needle 2 CACT CQ2 Mixed",
      "b43aabfcaf1a6db6acf488076eab71d823c08697c7af4521fc1d174b60ede5ba",
      ChatTemplate.NEEDLE2),
  QWEN3_06B(
      "qwen3-0.6b",
      "qwen3_0_6b_q4_0",
      "Qwen3 0.6B GGUF Q4_0",
      "da2572f16c06133561ce56accaa822216f2391ef4d37fba427801cd6736417d4",
      ChatTemplate.CHATML_NO_THINK),
  QWEN3_17B(
      "qwen3-1.7b",
      "qwen3_1_7b_q8_0",
      "Qwen3 1.7B GGUF Q8_0",
      "061b54daade076b5d3362dac252678d17da8c68f07560be70818cace6590cb1a",
      ChatTemplate.CHATML_NO_THINK),
  SMOLLM3_3B(
      "smollm3-3b",
      "smollm3_3b_q4_k_m",
      "SmolLM3 3B GGUF Q4_K_M",
      "8334b850b7bd46238c16b0c550df2138f0889bf433809008cc17a8b05761863e",
      ChatTemplate.SMOLLM3_NO_THINK),
  MINICPM5_1B(
      "minicpm5-1b",
      "minicpm5_1b_q4_k_m",
      "MiniCPM5 1B GGUF Q4_K_M",
      "81b64d05a23b17b34c475f42b3e72fbde62d4b92cc34541f7a8031d0752deafa",
      ChatTemplate.MINICPM5_NO_THINK),
  LLAMA32_3B(
      "llama3.2-3b",
      "bartowski_llama_3_2_3b_instruct_gguf_q4_k_m",
      "Llama 3.2 3B Instruct GGUF Q4_K_M",
      "6c1a2b41161032677be168d354123594c0e6e67d2b9227c84f296ad037c728ff",
      ChatTemplate.LLAMA3);

  private final String key;
  private final String modelId;
  private final String modelName;
  private final String artifactSha256;
  private final ChatTemplate template;

  ToolCallingCandidate(
      String key, String modelId, String modelName, String artifactSha256, ChatTemplate template) {
    this.key = key;
    this.modelId = modelId;
    this.modelName = modelName;
    this.artifactSha256 = artifactSha256;
    this.template = template;
  }

  String key() {
    return key;
  }

  String modelId() {
    return modelId;
  }

  String modelName() {
    return modelName;
  }

  String artifactSha256() {
    return artifactSha256;
  }

  ChatTemplate template() {
    return template;
  }

  static ToolCallingCandidate parse(String value) {
    String normalized = value == null ? "" : value.strip().toLowerCase(Locale.ROOT);
    return Arrays.stream(values())
        .filter(candidate -> candidate.key.equals(normalized))
        .findFirst()
        .orElseThrow(
            () ->
                new IllegalArgumentException(
                    "unknown tool-calling candidate '"
                        + value
                        + "'; expected one of "
                        + Arrays.stream(values()).map(ToolCallingCandidate::key).toList()));
  }
}
