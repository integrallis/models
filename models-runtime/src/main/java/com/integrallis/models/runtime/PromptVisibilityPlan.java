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

import com.integrallis.models.api.ModelPrompt;
import com.integrallis.models.api.Tokenizer;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.List;
import java.util.Objects;

/**
 * Causal visibility policy for bounded-context decoding with pinned prompt sinks.
 *
 * <p>The policy mirrors Needle's useful part: each query sees the recent window plus selected
 * earlier sink positions. In this runtime experiment, structured prompt control segments become
 * sinks because they usually carry chat delimiters, system instructions, and rendered tool schemas.
 */
public final class PromptVisibilityPlan {

  private final int tokenCount;
  private final int recentWindow;
  private final BitSet sinks;

  private PromptVisibilityPlan(int tokenCount, int recentWindow, BitSet sinks) {
    if (tokenCount < 0) {
      throw new IllegalArgumentException("tokenCount must be >= 0: " + tokenCount);
    }
    if (recentWindow <= 0) {
      throw new IllegalArgumentException("recentWindow must be > 0: " + recentWindow);
    }
    this.tokenCount = tokenCount;
    this.recentWindow = recentWindow;
    this.sinks = (BitSet) Objects.requireNonNull(sinks, "sinks").clone();
    if (this.sinks.length() > tokenCount) {
      throw new IllegalArgumentException("sink position outside tokenCount");
    }
  }

  /** Creates a visibility plan with explicit sink positions. */
  public static PromptVisibilityPlan of(int tokenCount, int recentWindow, int... sinkPositions) {
    Objects.requireNonNull(sinkPositions, "sinkPositions");
    BitSet sinks = new BitSet(tokenCount);
    for (int position : sinkPositions) {
      if (position < 0 || position >= tokenCount) {
        throw new IllegalArgumentException("sink position out of range: " + position);
      }
      sinks.set(position);
    }
    return new PromptVisibilityPlan(tokenCount, recentWindow, sinks);
  }

  /**
   * Builds a plan from a structured prompt, pinning tokens emitted by control segments as sinks.
   */
  public static PromptVisibilityPlan forPrompt(
      ModelPrompt prompt, Tokenizer tokenizer, int recentWindow) {
    Objects.requireNonNull(prompt, "prompt");
    Objects.requireNonNull(tokenizer, "tokenizer");

    int position = 0;
    BitSet sinks = new BitSet();
    for (ModelPrompt.Segment segment : prompt.segments()) {
      int[] tokens =
          segment.kind() == ModelPrompt.SegmentKind.CONTROL
              ? tokenizer.encodeControl(segment.text())
              : tokenizer.encode(segment.text());
      if (segment.kind() == ModelPrompt.SegmentKind.CONTROL) {
        sinks.set(position, position + tokens.length);
      }
      position += tokens.length;
    }
    return new PromptVisibilityPlan(position, recentWindow, sinks);
  }

  /** Returns the number of prompt tokens represented by the plan. */
  public int tokenCount() {
    return tokenCount;
  }

  /** Returns the recent causal window width. */
  public int recentWindow() {
    return recentWindow;
  }

  /** Returns whether {@code position} is pinned as a sink. */
  public boolean isSink(int position) {
    requirePosition(position, "position");
    return sinks.get(position);
  }

  /** Returns whether {@code keyPosition} is visible to {@code queryPosition}. */
  public boolean visible(int queryPosition, int keyPosition) {
    requirePosition(queryPosition, "queryPosition");
    requirePosition(keyPosition, "keyPosition");
    if (keyPosition > queryPosition) {
      return false;
    }
    int recentStart = Math.max(0, queryPosition - recentWindow + 1);
    return keyPosition >= recentStart || sinks.get(keyPosition);
  }

  /** Returns visible key positions for {@code queryPosition}, in ascending order. */
  public int[] visiblePositions(int queryPosition) {
    requirePosition(queryPosition, "queryPosition");
    List<Integer> positions = new ArrayList<>();
    for (int position = 0; position <= queryPosition; position++) {
      if (visible(queryPosition, position)) {
        positions.add(position);
      }
    }
    return positions.stream().mapToInt(Integer::intValue).toArray();
  }

  /** Returns pinned sink positions in ascending order. */
  public int[] sinkPositions() {
    return sinks.stream().toArray();
  }

  private void requirePosition(int position, String name) {
    if (position < 0 || position >= tokenCount) {
      throw new IllegalArgumentException(name + " out of range: " + position);
    }
  }
}
