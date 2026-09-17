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
package com.integrallis.models.backend.purejava.tokenizer;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Resolves the token that ends an assistant turn in a GGUF's {@code tokenizer.chat_template}
 * without evaluating the Jinja.
 *
 * <p>The template is scanned for occurrences of CONTROL-typed vocabulary entries and of the {@code
 * eos_token} variable. The candidate is the last such occurrence before the final {@code
 * add_generation_prompt}, which in a turn-structured template is the marker that closes the
 * preceding turn. It is accepted only when it is not the token that opens the generation prompt and
 * when, somewhere in the template, it is the first marker after a {@code content} reference, i.e.
 * it closes message content. Anything else is unresolved, with the reason, so a template this
 * scanner cannot read leaves the stop set exactly as the other rules built it.
 *
 * <p>USER_DEFINED entries such as {@code <think>} and {@code <tool_call>} are never candidates.
 */
final class ChatTemplateEndOfTurn {

  static final String ABSENT = "absent";

  private static final int TOKEN_TYPE_CONTROL = 3;
  private static final String GENERATION_PROMPT = "add_generation_prompt";
  private static final Pattern CONTENT = Pattern.compile("\\bcontent\\b");
  private static final Pattern EOS_TOKEN = Pattern.compile("\\beos_token\\b");

  /** The outcome: {@code tokenId} is {@code -1} unless resolved. */
  record Resolution(int tokenId, String description) {

    static Resolution resolved(int tokenId) {
      return new Resolution(tokenId, "resolved:" + tokenId);
    }

    static Resolution unresolved(String reason) {
      return new Resolution(-1, "unresolved:" + reason);
    }

    boolean isResolved() {
      return tokenId >= 0;
    }
  }

  private record Marker(int start, int end, int tokenId, String text) {}

  private ChatTemplateEndOfTurn() {}

  static Resolution resolve(
      String template, String[] vocab, List<Integer> tokenTypes, int eosTokenId) {
    if (template == null || template.isBlank()) {
      return new Resolution(-1, ABSENT);
    }
    if (tokenTypes.isEmpty()) {
      return Resolution.unresolved("vocabulary declares no token types");
    }
    int generationPrompt = template.lastIndexOf(GENERATION_PROMPT);
    if (generationPrompt < 0) {
      return Resolution.unresolved("no add_generation_prompt block");
    }
    List<Marker> markers = markers(template, vocab, tokenTypes, eosTokenId);
    Marker candidate = null;
    Marker opener = null;
    for (Marker marker : markers) {
      if (marker.start() < generationPrompt) {
        candidate = marker;
      } else if (opener == null) {
        opener = marker;
      }
    }
    if (candidate == null) {
      return Resolution.unresolved("no control token before the generation prompt");
    }
    if (opener != null && opener.tokenId() == candidate.tokenId()) {
      return Resolution.unresolved(candidate.text() + " also opens the generation prompt");
    }
    if (!closesContent(template, markers, candidate.tokenId())) {
      return Resolution.unresolved(candidate.text() + " does not close message content");
    }
    return Resolution.resolved(candidate.tokenId());
  }

  private static List<Marker> markers(
      String template, String[] vocab, List<Integer> tokenTypes, int eosTokenId) {
    List<Marker> found = new ArrayList<>();
    int typed = Math.min(vocab.length, tokenTypes.size());
    for (int token = 0; token < typed; token++) {
      String text = vocab[token];
      if (tokenTypes.get(token) != TOKEN_TYPE_CONTROL || text == null || text.length() < 3) {
        continue;
      }
      for (int at = template.indexOf(text); at >= 0; at = template.indexOf(text, at + 1)) {
        found.add(new Marker(at, at + text.length(), token, text));
      }
    }
    if (eosTokenId >= 0 && eosTokenId < vocab.length) {
      Matcher eos = EOS_TOKEN.matcher(template);
      while (eos.find()) {
        found.add(new Marker(eos.start(), eos.end(), eosTokenId, "eos_token"));
      }
    }
    // Longest match wins at a position; overlapped shorter matches are dropped.
    found.sort(
        Comparator.comparingInt(Marker::start)
            .thenComparing(Comparator.comparingInt(Marker::end).reversed()));
    List<Marker> markers = new ArrayList<>(found.size());
    int covered = -1;
    for (Marker marker : found) {
      if (marker.start() >= covered) {
        markers.add(marker);
        covered = marker.end();
      }
    }
    return markers;
  }

  private static boolean closesContent(String template, List<Marker> markers, int tokenId) {
    Matcher content = CONTENT.matcher(template);
    while (content.find()) {
      for (Marker marker : markers) {
        if (marker.start() >= content.end()) {
          if (marker.tokenId() == tokenId) {
            return true;
          }
          break;
        }
      }
    }
    return false;
  }
}
