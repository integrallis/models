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
package com.integrallis.models.semanticorder;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HexFormat;
import java.util.List;
import java.util.Objects;
import java.util.OptionalInt;

/** Pure-Java runtime for a Word Tour newline-delimited cyclic word order. */
public final class WordTour implements SemanticOrder {
  private static final char BYTE_ORDER_MARK = '\ufeff';

  private final String[] termsByRank;
  private final int[] ranksByTerm;
  private final List<String> terms;
  private final String fingerprint;

  private WordTour(String[] termsByRank, int[] ranksByTerm) {
    this.termsByRank = termsByRank;
    this.ranksByTerm = ranksByTerm;
    this.terms = List.copyOf(Arrays.asList(termsByRank));
    this.fingerprint = fingerprint(termsByRank);
  }

  /** Loads a UTF-8 newline-delimited tour and closes the file after reading it. */
  public static WordTour load(Path path) {
    Objects.requireNonNull(path, "path");
    try (InputStream input = Files.newInputStream(path)) {
      return load(input);
    } catch (IOException e) {
      throw new UncheckedIOException("Unable to load Word Tour: " + path, e);
    }
  }

  /**
   * Loads a UTF-8 newline-delimited tour. The caller retains ownership of {@code input}; this
   * method does not close it.
   */
  public static WordTour load(InputStream input) {
    Objects.requireNonNull(input, "input");
    BufferedReader reader =
        new BufferedReader(new InputStreamReader(input, StandardCharsets.UTF_8));
    List<String> values = new ArrayList<>();
    try {
      String line;
      int lineNumber = 0;
      while ((line = reader.readLine()) != null) {
        lineNumber++;
        if (lineNumber == 1 && !line.isEmpty() && line.charAt(0) == BYTE_ORDER_MARK) {
          line = line.substring(1);
        }
        if (line.isBlank()) {
          throw new IllegalArgumentException(
              "Word Tour contains a blank term at line " + lineNumber);
        }
        values.add(line);
      }
    } catch (IOException e) {
      throw new UncheckedIOException("Unable to read Word Tour", e);
    }
    return fromTerms(values);
  }

  /** Creates a tour from terms already arranged in cyclic rank order. */
  public static WordTour fromTerms(List<String> terms) {
    Objects.requireNonNull(terms, "terms");
    if (terms.isEmpty()) {
      throw new IllegalArgumentException("Word Tour must not be empty");
    }

    String[] termsByRank = new String[terms.size()];
    for (int rank = 0; rank < terms.size(); rank++) {
      String term = Objects.requireNonNull(terms.get(rank), "term at rank " + rank);
      if (term.isBlank()) {
        throw new IllegalArgumentException("Word Tour contains a blank term at rank " + rank);
      }
      termsByRank[rank] = term;
    }

    Integer[] sortedRanks = new Integer[termsByRank.length];
    for (int rank = 0; rank < sortedRanks.length; rank++) {
      sortedRanks[rank] = rank;
    }
    Arrays.sort(sortedRanks, Comparator.comparing(rank -> termsByRank[rank]));

    int[] ranksByTerm = new int[sortedRanks.length];
    for (int index = 0; index < sortedRanks.length; index++) {
      int rank = sortedRanks[index];
      if (index > 0 && termsByRank[sortedRanks[index - 1]].equals(termsByRank[rank])) {
        throw new IllegalArgumentException("Duplicate Word Tour term: " + termsByRank[rank]);
      }
      ranksByTerm[index] = rank;
    }
    return new WordTour(termsByRank, ranksByTerm);
  }

  @Override
  public int size() {
    return termsByRank.length;
  }

  @Override
  public OptionalInt rank(String term) {
    Objects.requireNonNull(term, "term");
    int low = 0;
    int high = ranksByTerm.length - 1;
    while (low <= high) {
      int middle = (low + high) >>> 1;
      int rank = ranksByTerm[middle];
      int comparison = termsByRank[rank].compareTo(term);
      if (comparison < 0) {
        low = middle + 1;
      } else if (comparison > 0) {
        high = middle - 1;
      } else {
        return OptionalInt.of(rank);
      }
    }
    return OptionalInt.empty();
  }

  @Override
  public String termAt(int rank) {
    return termsByRank[Objects.checkIndex(rank, termsByRank.length)];
  }

  @Override
  public List<String> terms() {
    return terms;
  }

  @Override
  public List<SemanticNeighbor> neighbors(String term, int radius) {
    if (radius < 0) {
      throw new IllegalArgumentException("radius must be >= 0");
    }
    int center = requireRank(term);
    int effectiveRadius = Math.min(radius, size() / 2);
    int capacity = Math.min(size() - 1, effectiveRadius * 2);
    List<SemanticNeighbor> result = new ArrayList<>(capacity);
    for (int distance = 1; distance <= effectiveRadius; distance++) {
      int predecessor = Math.floorMod(center - distance, size());
      int successor = Math.floorMod(center + distance, size());
      result.add(new SemanticNeighbor(termsByRank[predecessor], predecessor, -distance));
      if (successor != predecessor) {
        result.add(new SemanticNeighbor(termsByRank[successor], successor, distance));
      }
    }
    return List.copyOf(result);
  }

  @Override
  public int cyclicDistance(String left, String right) {
    int leftRank = requireRank(left);
    int rightRank = requireRank(right);
    int direct = Math.abs(leftRank - rightRank);
    return Math.min(direct, size() - direct);
  }

  @Override
  public String fingerprint() {
    return fingerprint;
  }

  private int requireRank(String term) {
    Objects.requireNonNull(term, "term");
    return rank(term)
        .orElseThrow(() -> new IllegalArgumentException("Unknown Word Tour term: " + term));
  }

  private static String fingerprint(String[] terms) {
    MessageDigest digest;
    try {
      digest = MessageDigest.getInstance("SHA-256");
    } catch (NoSuchAlgorithmException e) {
      throw new IllegalStateException("SHA-256 is unavailable", e);
    }
    byte[] separator = {0};
    for (String term : terms) {
      digest.update(term.getBytes(StandardCharsets.UTF_8));
      digest.update(separator);
    }
    return HexFormat.of().formatHex(digest.digest());
  }
}
