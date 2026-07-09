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

import java.util.Arrays;
import java.util.Locale;

final class StubAppleFoundationModelsBridge implements AppleFoundationModelsBridge {

  @Override
  public AppleFoundationModelsAvailability availability() {
    return new AppleFoundationModelsAvailability(
        true, true, "Apple Foundation Models stub mode is available");
  }

  @Override
  public AppleFoundationModelsResponse generate(AppleFoundationModelsRequest request) {
    return new AppleFoundationModelsResponse(
        limitWords(responseFor(request), request.maxOutputTokens()));
  }

  @Override
  public void close() {}

  private static String responseFor(AppleFoundationModelsRequest request) {
    String prompt = request.prompt().trim();
    String lowerPrompt = prompt.toLowerCase(Locale.ROOT);
    if (lowerPrompt.contains("single word hello")) {
      return "hello";
    }
    if (lowerPrompt.startsWith("summarize") || lowerPrompt.contains("summarize:")) {
      return "Stub summary: " + subject(prompt);
    }
    if (lowerPrompt.contains("json")) {
      return "{\"stub\":true,\"text\":\"" + jsonEscape(subject(prompt)) + "\"}";
    }
    if (request.instructions().isBlank()) {
      return "Stub response: " + prompt;
    }
    return "Stub response: " + prompt + " [" + request.instructions().trim() + "]";
  }

  private static String subject(String prompt) {
    int colon = prompt.indexOf(':');
    if (colon >= 0 && colon + 1 < prompt.length()) {
      return prompt.substring(colon + 1).trim();
    }
    return prompt;
  }

  private static String limitWords(String text, int maxWords) {
    String[] words = text.split("\\s+");
    if (words.length <= maxWords) {
      return text;
    }
    return String.join(" ", Arrays.copyOf(words, maxWords));
  }

  private static String jsonEscape(String text) {
    return text.replace("\\", "\\\\").replace("\"", "\\\"");
  }
}
