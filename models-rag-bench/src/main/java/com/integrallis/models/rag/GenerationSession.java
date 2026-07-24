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

final class GenerationSession {
  private final GenerationClient client;
  private final ThreadLocal<Integer> maxTokens = new ThreadLocal<>();
  private final ThreadLocal<String> prompt = new ThreadLocal<>();
  private final ThreadLocal<GenerationResult> result = new ThreadLocal<>();

  GenerationSession(GenerationClient client) {
    this.client = client;
  }

  void begin(int requestedMaxTokens) {
    maxTokens.set(requestedMaxTokens);
    prompt.remove();
    result.remove();
  }

  GenerationResult generate(String renderedPrompt) {
    Integer requestedMaxTokens = maxTokens.get();
    if (requestedMaxTokens == null) {
      throw new IllegalStateException("generation session was not started");
    }
    GenerationResult generated = client.generate(renderedPrompt, requestedMaxTokens);
    prompt.set(renderedPrompt);
    result.set(generated);
    return generated;
  }

  String prompt() {
    String value = prompt.get();
    if (value == null) {
      throw new IllegalStateException("framework did not invoke generation");
    }
    return value;
  }

  GenerationResult result() {
    GenerationResult value = result.get();
    if (value == null) {
      throw new IllegalStateException("framework did not invoke generation");
    }
    return value;
  }
}
