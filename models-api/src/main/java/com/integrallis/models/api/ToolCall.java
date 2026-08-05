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
package com.integrallis.models.api;

/**
 * A tool invocation produced by the model.
 *
 * <p>{@code argumentsJson} is opaque text. The runtime locates its extent with a brace-aware scan
 * but never parses it: Spring AI's {@code ToolCall.arguments} and LangChain4j's {@code
 * ToolExecutionRequest.arguments()} are both strings, so parsing here would add a JSON dependency
 * to a module that has none and buy nothing.
 *
 * <p>Open-weight models rarely emit call identifiers, so {@link #syntheticId(int)} supplies one.
 */
public record ToolCall(String id, String name, String argumentsJson) {

  /** Width required by Mistral chat templates, which reject identifiers of any other length. */
  private static final int IDENTIFIER_WIDTH = 9;

  public ToolCall {
    if (name == null || name.isBlank()) {
      throw new IllegalArgumentException("name must not be blank");
    }
    if (id == null || id.isBlank()) {
      throw new IllegalArgumentException("id must not be blank");
    }
    argumentsJson = argumentsJson == null || argumentsJson.isBlank() ? "{}" : argumentsJson;
  }

  /** Creates a call with a synthesised identifier derived from its position in the response. */
  public static ToolCall of(int index, String name, String argumentsJson) {
    return new ToolCall(syntheticId(index), name, argumentsJson);
  }

  /**
   * Returns a nine-digit identifier for the call at {@code index}.
   *
   * <p>Mistral's templates raise unless the identifier is exactly nine characters. Nine digits
   * satisfies that and is inert everywhere else, so one scheme serves every family.
   */
  public static String syntheticId(int index) {
    if (index < 0) {
      throw new IllegalArgumentException("index must be >= 0, got: " + index);
    }
    return String.format("%0" + IDENTIFIER_WIDTH + "d", index);
  }
}
