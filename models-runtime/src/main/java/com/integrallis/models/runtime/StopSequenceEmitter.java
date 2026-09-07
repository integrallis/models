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

import com.integrallis.models.api.TokenStream;
import java.util.List;

/** Holds possible partial stop sequences until they can be emitted or suppressed. */
final class StopSequenceEmitter {
  private final TokenStream stream;
  private final List<String> stopSequences;
  private final StringBuilder pending = new StringBuilder();

  StopSequenceEmitter(TokenStream stream, List<String> stopSequences) {
    this.stream = stream;
    this.stopSequences = stopSequences;
  }

  boolean emit(String token) {
    if (stopSequences.isEmpty()) {
      stream.onToken(token);
      return false;
    }

    pending.append(token);
    int stopIndex = earliestStopIndex();
    if (stopIndex >= 0) {
      flush(stopIndex);
      pending.setLength(0);
      return true;
    }

    int retainedSuffix = longestPotentialStopPrefix();
    flush(pending.length() - retainedSuffix);
    return false;
  }

  void finish() {
    flush(pending.length());
  }

  private int earliestStopIndex() {
    int earliest = -1;
    for (String stopSequence : stopSequences) {
      int index = pending.indexOf(stopSequence);
      if (index >= 0 && (earliest < 0 || index < earliest)) {
        earliest = index;
      }
    }
    return earliest;
  }

  private int longestPotentialStopPrefix() {
    int retained = 0;
    for (String stopSequence : stopSequences) {
      int limit = Math.min(pending.length(), stopSequence.length() - 1);
      for (int length = limit; length > retained; length--) {
        if (pending
            .substring(pending.length() - length)
            .equals(stopSequence.substring(0, length))) {
          retained = length;
          break;
        }
      }
    }
    return retained;
  }

  private void flush(int length) {
    if (length <= 0) {
      return;
    }
    stream.onToken(pending.substring(0, length));
    pending.delete(0, length);
  }
}
