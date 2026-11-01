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

import java.time.Duration;
import java.util.List;
import java.util.Objects;

/**
 * One transcription result.
 *
 * <p>Carries the audio duration it covers as well as the text, because a word error rate is only
 * comparable when the reference and the hypothesis describe the same span, and because a
 * transcription that silently stopped early otherwise reads as a short utterance rather than as a
 * truncated one. {@link #truncated()} says which it was.
 */
public record Transcript(String text, List<Segment> segments, Duration audioDuration,
    boolean truncated) {

  /** A timed span of recognised text. */
  public record Segment(String text, Duration start, Duration end) {
    public Segment {
      Objects.requireNonNull(text, "text");
      Objects.requireNonNull(start, "start");
      Objects.requireNonNull(end, "end");
      if (start.isNegative() || end.isNegative()) {
        throw new IllegalArgumentException("segment bounds must not be negative");
      }
      if (end.compareTo(start) < 0) {
        throw new IllegalArgumentException("segment end precedes its start: " + start + " > " + end);
      }
    }
  }

  public Transcript {
    Objects.requireNonNull(text, "text");
    Objects.requireNonNull(audioDuration, "audioDuration");
    if (audioDuration.isNegative()) {
      throw new IllegalArgumentException("audioDuration must not be negative");
    }
    segments = segments == null ? List.of() : List.copyOf(segments);
  }

  /** A transcript with no segment timings. */
  public static Transcript of(String text, Duration audioDuration) {
    return new Transcript(text, List.of(), audioDuration, false);
  }
}
