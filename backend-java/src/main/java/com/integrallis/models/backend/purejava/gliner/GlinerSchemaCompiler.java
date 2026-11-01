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
package com.integrallis.models.backend.purejava.gliner;

import com.integrallis.models.api.SchemaClassificationModel.Label;
import com.integrallis.models.api.SchemaClassificationModel.Request;
import com.integrallis.models.api.SchemaClassificationModel.Task;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.function.Function;
import java.util.regex.Pattern;

/** Publisher-compatible structural slots, independent of marker spellings in user text. */
public final class GlinerSchemaCompiler {
  private static final String WORD = "[\\p{L}\\p{N}\\p{M}_\\u200c\\u200d]";
  private static final Pattern SPLITTER =
      Pattern.compile(
          "https?://[^\\s]+|www\\.[^\\s]+|[a-z0-9._%+-]+@[a-z0-9.-]+\\.[a-z]{2,}"
              + "|@[a-z0-9_]+|"
              + WORD
              + "+(?:[-_]"
              + WORD
              + "+)*|\\S",
          Pattern.CASE_INSENSITIVE | Pattern.UNICODE_CASE | Pattern.UNICODE_CHARACTER_CLASS);
  private final Function<String, int[]> tokenizer;
  private final int maximumTokens;

  public GlinerSchemaCompiler(Function<String, int[]> tokenizer, int maximumTokens) {
    this.tokenizer = Objects.requireNonNull(tokenizer, "tokenizer");
    if (maximumTokens < 1) {
      throw new IllegalArgumentException("maximumTokens must be positive");
    }
    this.maximumTokens = maximumTokens;
  }

  /** Caller-owned token IDs and classification marker indices. */
  public record Encoded(int[] tokenIds, int[] markerPositions) {
    public Encoded {
      tokenIds = tokenIds.clone();
      markerPositions = markerPositions.clone();
    }

    @Override
    public int[] tokenIds() {
      return tokenIds.clone();
    }

    @Override
    public int[] markerPositions() {
      return markerPositions.clone();
    }
  }

  public Encoded compile(Request request) {
    Objects.requireNonNull(request, "request");
    List<Integer> ids = new ArrayList<>();
    List<Integer> markers = new ArrayList<>();
    Map<String, int[]> pieces = new HashMap<>();
    for (Task task : request.tasks()) {
      if (!ids.isEmpty()) {
        append(ids, pieces, "[SEP_STRUCT]");
      }
      StringBuilder description = new StringBuilder(task.name());
      for (Label label : task.labels()) {
        if (!label.description().isEmpty()) {
          description
              .append(" [DESCRIPTION] ")
              .append(label.name())
              .append(": ")
              .append(label.description());
        }
      }
      append(ids, pieces, "(");
      append(ids, pieces, "[P]");
      append(ids, pieces, description.toString());
      append(ids, pieces, "(");
      for (Label label : task.labels()) {
        markers.add(ids.size());
        append(ids, pieces, "[L]");
        append(ids, pieces, label.name());
      }
      append(ids, pieces, ")");
      append(ids, pieces, ")");
    }
    append(ids, pieces, "[SEP_TEXT]");
    String text = request.text();
    if (text.isEmpty() || !(text.endsWith(".") || text.endsWith("!") || text.endsWith("?"))) {
      text += ".";
    }
    var words = SPLITTER.matcher(text);
    while (words.find()) {
      append(ids, pieces, words.group().toLowerCase(Locale.ROOT));
    }
    return new Encoded(
        ids.stream().mapToInt(Integer::intValue).toArray(),
        markers.stream().mapToInt(Integer::intValue).toArray());
  }

  private void append(List<Integer> ids, Map<String, int[]> pieces, String piece) {
    int[] tokens = pieces.computeIfAbsent(piece, tokenizer);
    if ((long) ids.size() + tokens.length > maximumTokens) {
      throw new IllegalArgumentException(
          "schema plus text exceeds " + maximumTokens + " tokens; no truncation applied");
    }
    for (int token : tokens) {
      ids.add(token);
    }
  }
}
