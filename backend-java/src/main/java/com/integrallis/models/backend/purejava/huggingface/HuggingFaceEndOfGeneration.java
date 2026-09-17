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
package com.integrallis.models.backend.purejava.huggingface;

import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonToken;
import com.fasterxml.jackson.core.StreamReadFeature;
import com.fasterxml.jackson.core.exc.StreamReadException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/**
 * Reads the end-of-generation token IDs a Hugging Face checkpoint declares.
 *
 * <p>Both {@code config.json} and {@code generation_config.json} may carry a top-level {@code
 * eos_token_id}, either as one integer or as a list; chat checkpoints commonly list several (Qwen
 * 2.5 Instruct declares end-of-turn and end-of-text, GPT-OSS declares return, end-of-text and
 * call). Generation must stop on any of them, so the declarations are unioned.
 */
public final class HuggingFaceEndOfGeneration {

  private static final JsonFactory JSON =
      JsonFactory.builder().enable(StreamReadFeature.STRICT_DUPLICATE_DETECTION).build();

  private HuggingFaceEndOfGeneration() {}

  /**
   * Returns the union of the EOS IDs declared by a checkpoint directory's {@code config.json} and
   * {@code generation_config.json}, in declaration order without duplicates.
   *
   * <p>Absent files, absent keys, and {@code null} values contribute nothing.
   *
   * @param modelDirectory checkpoint directory
   * @return declared end-of-generation token IDs
   * @throws IOException if a present file cannot be read
   * @throws IllegalArgumentException if a declaration is not a non-negative integer or a list of
   *     them
   */
  public static List<Integer> tokenIds(Path modelDirectory) throws IOException {
    Objects.requireNonNull(modelDirectory, "modelDirectory");
    Set<Integer> ids = new LinkedHashSet<>();
    ids.addAll(read(modelDirectory.resolve("config.json")));
    ids.addAll(read(modelDirectory.resolve("generation_config.json")));
    return List.copyOf(ids);
  }

  /**
   * Returns the EOS IDs declared by one JSON file's top-level {@code eos_token_id}.
   *
   * @param file configuration file; absent files yield an empty list
   * @return declared IDs in order
   * @throws IOException if the file cannot be read
   */
  public static List<Integer> read(Path file) throws IOException {
    Objects.requireNonNull(file, "file");
    if (!Files.isRegularFile(file)) {
      return List.of();
    }
    List<Integer> ids = new ArrayList<>();
    try (JsonParser parser = JSON.createParser(file.toFile())) {
      if (parser.nextToken() != JsonToken.START_OBJECT) {
        throw new IllegalArgumentException(file.getFileName() + " must contain a JSON object");
      }
      while (parser.nextToken() != JsonToken.END_OBJECT) {
        String name = parser.currentName();
        JsonToken value = parser.nextToken();
        if ("eos_token_id".equals(name)) {
          ids.addAll(readTokenIds(parser, value, file.getFileName() + " eos_token_id"));
        } else {
          parser.skipChildren();
        }
      }
    } catch (StreamReadException malformed) {
      throw new IllegalArgumentException(
          "invalid or duplicate JSON in " + file.getFileName() + ": " + malformed.getMessage(),
          malformed);
    }
    return List.copyOf(ids);
  }

  /**
   * Reads a token-ID declaration that is either {@code null}, one integer, or a list of integers.
   *
   * @param parser parser positioned on {@code token}
   * @param token the value's first token
   * @param name field name used in error messages
   * @return declared IDs in order; empty for {@code null}
   * @throws IOException if the parser fails
   */
  public static List<Integer> readTokenIds(JsonParser parser, JsonToken token, String name)
      throws IOException {
    if (token == JsonToken.VALUE_NULL) {
      return List.of();
    }
    if (token == JsonToken.VALUE_NUMBER_INT) {
      return List.of(nonNegative(parser.getIntValue(), name));
    }
    if (token != JsonToken.START_ARRAY) {
      throw new IllegalArgumentException(name + " must be an integer or a list of integers");
    }
    List<Integer> ids = new ArrayList<>();
    while (parser.nextToken() != JsonToken.END_ARRAY) {
      if (parser.currentToken() != JsonToken.VALUE_NUMBER_INT) {
        throw new IllegalArgumentException(name + " list entries must be integers");
      }
      ids.add(nonNegative(parser.getIntValue(), name));
    }
    return List.copyOf(ids);
  }

  private static int nonNegative(int id, String name) {
    if (id < 0) {
      throw new IllegalArgumentException(name + " must not be negative, got: " + id);
    }
    return id;
  }
}
