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
package com.integrallis.models.runtime;

import java.util.List;

/** Recent-history continuation lookup modeled after llama.cpp's ngram-simple strategy. */
final class NgramDraftStrategy {

  private static final int[] NO_DRAFT = new int[0];

  private final SpeculativeGenerationOptions options;
  private int windowAttempts;
  private int windowProposed;
  private int windowAccepted;
  private int suppressedUntilGenerated;

  NgramDraftStrategy(SpeculativeGenerationOptions options) {
    this.options = options;
  }

  boolean isSuppressed(int generatedTokens) {
    return generatedTokens < suppressedUntilGenerated;
  }

  int[] propose(List<Integer> history, int sampledToken, int remainingTokens) {
    int draftLimit = Math.min(options.maximumDraftTokens(), remainingTokens);
    if (draftLimit < options.minimumDraftTokens()) {
      return NO_DRAFT;
    }

    int ngramSize = options.ngramSize();
    int historySize = history.size();
    if (historySize < ngramSize + options.minimumDraftTokens()) {
      return NO_DRAFT;
    }

    int patternHistoryStart = historySize - (ngramSize - 1);
    int firstCandidate = Math.max(0, historySize - options.historyWindow());
    int lastCandidate = historySize - ngramSize - options.minimumDraftTokens();
    for (int candidate = lastCandidate; candidate >= firstCandidate; candidate--) {
      if (!matches(history, candidate, patternHistoryStart, sampledToken, ngramSize)) {
        continue;
      }
      int continuationStart = candidate + ngramSize;
      int available = historySize - continuationStart;
      int draftLength = Math.min(draftLimit, available);
      if (draftLength < options.minimumDraftTokens()) {
        continue;
      }
      int[] draft = new int[draftLength];
      for (int index = 0; index < draftLength; index++) {
        draft[index] = history.get(continuationStart + index);
      }
      return draft;
    }
    return NO_DRAFT;
  }

  void recordVerification(int proposed, int accepted, int generatedTokens) {
    windowAttempts++;
    windowProposed += proposed;
    windowAccepted += accepted;
    if (windowAttempts < options.adaptationWindow()) {
      return;
    }
    float acceptanceRate = windowProposed == 0 ? 0.0f : (float) windowAccepted / windowProposed;
    if (acceptanceRate < options.minimumAcceptanceRate()) {
      suppressedUntilGenerated = generatedTokens + options.cooldownTokens();
    }
    windowAttempts = 0;
    windowProposed = 0;
    windowAccepted = 0;
  }

  private static boolean matches(
      List<Integer> history,
      int candidate,
      int patternHistoryStart,
      int sampledToken,
      int ngramSize) {
    for (int index = 0; index < ngramSize - 1; index++) {
      if (!history.get(candidate + index).equals(history.get(patternHistoryStart + index))) {
        return false;
      }
    }
    return history.get(candidate + ngramSize - 1) == sampledToken;
  }
}
