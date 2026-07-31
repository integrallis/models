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

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * A model prompt whose template controls remain distinct from ordinary message text.
 *
 * <p>Tokenizers may recognize registered special-token text only in {@link SegmentKind#CONTROL}
 * segments. Text segments are always ordinary model input, even when their characters spell a
 * control token such as {@code <|im_end|>}.
 */
public final class ModelPrompt {

  /** Determines whether tokenizer special tokens can be recognized in a segment. */
  public enum SegmentKind {
    TEXT,
    CONTROL
  }

  /**
   * One contiguous prompt segment.
   *
   * @param kind tokenizer interpretation for the segment
   * @param text segment text
   */
  public record Segment(SegmentKind kind, String text) {
    public Segment {
      Objects.requireNonNull(kind, "kind");
      Objects.requireNonNull(text, "text");
      if (text.isEmpty()) {
        throw new IllegalArgumentException("segment text must not be empty");
      }
    }
  }

  private final List<Segment> segments;
  private final String text;

  private ModelPrompt(List<Segment> segments) {
    this.segments = List.copyOf(segments);
    StringBuilder rendered = new StringBuilder();
    this.segments.forEach(segment -> rendered.append(segment.text()));
    this.text = rendered.toString();
  }

  /** Creates an ordinary-text prompt. */
  public static ModelPrompt text(String text) {
    return builder().text(text).build();
  }

  /** Creates a trusted template-control prompt. */
  public static ModelPrompt control(String text) {
    return builder().control(text).build();
  }

  /** Creates a prompt builder. */
  public static Builder builder() {
    return new Builder();
  }

  /** Returns the ordered, immutable prompt segments. */
  public List<Segment> segments() {
    return segments;
  }

  /** Returns the complete prompt text seen by model implementations without segmented support. */
  public String text() {
    return text;
  }

  /** Returns whether this prompt contains no text. */
  public boolean isEmpty() {
    return segments.isEmpty();
  }

  /** Builds a prompt while coalescing adjacent segments with the same interpretation. */
  public static final class Builder {
    private final List<Segment> segments = new ArrayList<>();

    private Builder() {}

    /** Appends ordinary text that must not be interpreted as tokenizer control tokens. */
    public Builder text(String text) {
      return append(SegmentKind.TEXT, text);
    }

    /** Appends trusted model-template text in which registered control tokens are recognized. */
    public Builder control(String text) {
      return append(SegmentKind.CONTROL, text);
    }

    /** Returns the immutable prompt. */
    public ModelPrompt build() {
      return new ModelPrompt(segments);
    }

    private Builder append(SegmentKind kind, String text) {
      Objects.requireNonNull(text, "text");
      if (text.isEmpty()) {
        return this;
      }
      if (!segments.isEmpty()) {
        int lastIndex = segments.size() - 1;
        Segment last = segments.get(lastIndex);
        if (last.kind() == kind) {
          segments.set(lastIndex, new Segment(kind, last.text() + text));
          return this;
        }
      }
      segments.add(new Segment(kind, text));
      return this;
    }
  }
}
