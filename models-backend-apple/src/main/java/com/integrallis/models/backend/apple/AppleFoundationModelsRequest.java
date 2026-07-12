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
package com.integrallis.models.backend.apple;

import java.util.Objects;

/**
 * Prompt request for Apple Foundation Models.
 *
 * @param prompt user prompt
 * @param instructions optional session instructions
 * @param maxOutputTokens best-effort output token limit passed to the native bridge
 */
public record AppleFoundationModelsRequest(
    String prompt, String instructions, int maxOutputTokens) {

  public static final int DEFAULT_MAX_OUTPUT_TOKENS = 256;

  public AppleFoundationModelsRequest {
    Objects.requireNonNull(prompt, "prompt");
    instructions = instructions == null ? "" : instructions;
    if (prompt.isBlank()) {
      throw new IllegalArgumentException("prompt must not be blank");
    }
    if (maxOutputTokens <= 0) {
      throw new IllegalArgumentException("maxOutputTokens must be > 0: " + maxOutputTokens);
    }
  }

  /** Creates a request builder for the supplied prompt. */
  public static Builder builder(String prompt) {
    return new Builder(prompt);
  }

  /** Builder for {@link AppleFoundationModelsRequest}. */
  public static final class Builder {
    private final String prompt;
    private String instructions = "";
    private int maxOutputTokens = DEFAULT_MAX_OUTPUT_TOKENS;

    private Builder(String prompt) {
      this.prompt = prompt;
    }

    public Builder instructions(String instructions) {
      this.instructions = instructions;
      return this;
    }

    public Builder maxOutputTokens(int maxOutputTokens) {
      this.maxOutputTokens = maxOutputTokens;
      return this;
    }

    public AppleFoundationModelsRequest build() {
      return new AppleFoundationModelsRequest(prompt, instructions, maxOutputTokens);
    }
  }
}
