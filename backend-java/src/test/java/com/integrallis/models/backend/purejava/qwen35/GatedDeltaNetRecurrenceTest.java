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
package com.integrallis.models.backend.purejava.qwen35;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

import java.util.Arrays;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Compatibility fixtures for FreeToken's Qwen3.5 Gated DeltaNet recurrence.
 *
 * <p>The expected output and final state were independently evaluated in float32 from {@code
 * recurrent_gated_delta_rule} in {@code freetoken/models/qwen3_5_moe/gdn_reference.py} at FreeToken
 * commit {@code bd372b630a028e3faa51f4ab0ef6a98c2f2de501}.
 */
@Tag("unit")
class GatedDeltaNetRecurrenceTest {

  private static final int TOKENS = 3;
  private static final int HEADS = 2;
  private static final int KEY_DIM = 2;
  private static final int VALUE_DIM = 2;

  private static final float[] QUERY = {
    0.5f, -1.0f, 1.5f, 0.25f,
    -0.75f, 0.5f, 0.1f, -0.4f,
    1.25f, 0.75f, -0.3f, 0.9f
  };

  private static final float[] KEY = {
    0.2f, 0.8f, -0.6f, 0.9f,
    1.0f, -0.5f, 0.7f, 0.2f,
    -0.4f, 0.3f, 0.25f, -0.75f
  };

  private static final float[] VALUE = {
    0.3f, -0.7f, 1.1f, 0.4f,
    -0.2f, 0.9f, 0.5f, -1.2f,
    0.8f, 0.1f, -0.6f, 0.7f
  };

  private static final float[] LOG_DECAY = {
    -0.1f, -0.3f,
    -0.5f, -0.2f,
    -0.05f, -0.7f
  };

  private static final float[] BETA = {
    0.7f, 0.2f,
    0.4f, 0.9f,
    1.0f, 0.6f
  };

  private static final float[] INITIAL_STATE = {
    0.1f, -0.2f, 0.3f, 0.4f,
    -0.5f, 0.2f, 0.1f, -0.3f
  };

  private static final float[] EXPECTED_OUTPUT = {
    -0.148594886f,
    0.0923976079f,
    -0.29807961f,
    0.0387914963f,
    0.0826374739f,
    -0.232684523f,
    -0.201746792f,
    0.0990339369f,
    -0.115048051f,
    -0.159083426f,
    0.29010731f,
    -0.305043638f
  };

  private static final float[] EXPECTED_FINAL_STATE = {
    -0.549856126f,
    -0.20131427f,
    0.600189984f,
    -0.101754844f,
    0.0727154538f,
    -0.372453094f,
    0.456705213f,
    -0.578883469f
  };

  @Test
  void matchesThePinnedFreeTokenFloat32Recurrence() {
    float[] initialState = INITIAL_STATE.clone();

    GatedDeltaNetRecurrence.Result actual =
        GatedDeltaNetRecurrence.forward(
            QUERY, KEY, VALUE, LOG_DECAY, BETA, initialState, TOKENS, HEADS, KEY_DIM, VALUE_DIM);

    assertThat(actual.output()).containsExactly(EXPECTED_OUTPUT, within(2.0e-6f));
    assertThat(actual.finalState()).containsExactly(EXPECTED_FINAL_STATE, within(2.0e-6f));
    assertThat(initialState).containsExactly(INITIAL_STATE);
  }

  @Test
  void continuationMatchesOneUninterruptedPass() {
    GatedDeltaNetRecurrence.Result expected =
        GatedDeltaNetRecurrence.forward(
            QUERY, KEY, VALUE, LOG_DECAY, BETA, INITIAL_STATE, TOKENS, HEADS, KEY_DIM, VALUE_DIM);

    GatedDeltaNetRecurrence.Result first =
        GatedDeltaNetRecurrence.forward(
            firstTokens(QUERY, KEY_DIM, 2),
            firstTokens(KEY, KEY_DIM, 2),
            firstTokens(VALUE, VALUE_DIM, 2),
            firstTokens(LOG_DECAY, 1, 2),
            firstTokens(BETA, 1, 2),
            INITIAL_STATE,
            2,
            HEADS,
            KEY_DIM,
            VALUE_DIM);
    GatedDeltaNetRecurrence.Result second =
        GatedDeltaNetRecurrence.forward(
            lastToken(QUERY, KEY_DIM),
            lastToken(KEY, KEY_DIM),
            lastToken(VALUE, VALUE_DIM),
            lastToken(LOG_DECAY, 1),
            lastToken(BETA, 1),
            first.finalState(),
            1,
            HEADS,
            KEY_DIM,
            VALUE_DIM);

    float[] continued = new float[first.output().length + second.output().length];
    System.arraycopy(first.output(), 0, continued, 0, first.output().length);
    System.arraycopy(second.output(), 0, continued, first.output().length, second.output().length);

    assertThat(continued).containsExactly(expected.output(), within(2.0e-6f));
    assertThat(second.finalState()).containsExactly(expected.finalState(), within(2.0e-6f));
  }

  @Test
  void inPlaceExecutionMatchesTheReferenceWithoutReplacingState() {
    float[] initialState = {0.2f, -0.1f, 0.05f, 0.3f, -0.2f, 0.4f, 0.1f, -0.05f};
    var expected =
        GatedDeltaNetRecurrence.forward(
            QUERY, KEY, VALUE, LOG_DECAY, BETA, initialState, 3, 2, 2, 2);
    float[] mutableState = initialState.clone();

    var actual =
        GatedDeltaNetRecurrence.forwardInPlace(
            QUERY, KEY, VALUE, LOG_DECAY, BETA, mutableState, 3, 2, 2, 2);

    assertThat(actual.output()).containsExactly(expected.output());
    assertThat(actual.finalState()).isSameAs(mutableState).containsExactly(expected.finalState());
  }

  @Test
  void usesTheTiledGgufHeadOrderForGroupedQueryAndKeyHeads() {
    float[] query = {1.0f, 0.0f, 1.0f, 0.0f};
    float[] key = {1.0f, 0.0f, 0.0f, 1.0f};
    float[] value = {1.0f, 2.0f, 3.0f, 4.0f};
    float[] logDecay = {0.0f, 0.0f, 0.0f, 0.0f};
    float[] beta = {1.0f, 1.0f, 1.0f, 1.0f};

    var actual =
        GatedDeltaNetRecurrence.forward(query, key, value, logDecay, beta, null, 1, 2, 4, 2, 1);

    assertThat(actual.output())
        .containsExactly(new float[] {0.70710605f, 0.0f, 2.121318f, 0.0f}, within(2.0e-6f));
    assertThat(actual.finalState())
        .containsExactly(
            new float[] {0.9999995f, 0.0f, 0.0f, 1.999999f, 2.9999986f, 0.0f, 0.0f, 3.999998f},
            within(2.0e-6f));
  }

  private static float[] firstTokens(float[] values, int dimension, int tokenCount) {
    return Arrays.copyOf(values, tokenCount * HEADS * dimension);
  }

  private static float[] lastToken(float[] values, int dimension) {
    int tokenWidth = HEADS * dimension;
    return Arrays.copyOfRange(values, values.length - tokenWidth, values.length);
  }
}
