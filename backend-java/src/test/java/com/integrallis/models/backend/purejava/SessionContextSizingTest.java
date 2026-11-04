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
package com.integrallis.models.backend.purejava;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.models.backend.purejava.plan.PureJavaPlanConfiguration;
import java.util.Map;
import org.junit.jupiter.api.Test;

/**
 * How many tokens a session is sized for when nobody says.
 *
 * <p>A session allocates its whole key and value cache at open, so this is a memory decision rather
 * than only a limit, and taking the model's declared ceiling made it badly: a 262,144-token context
 * is about 17 GB of fp32 cache, which exhausted the heap before the first answer on a machine with
 * room to spare for the weights. Asserted here and not through a real model because the policy is
 * arithmetic -- a multi-gigabyte download would make the check too expensive to keep.
 */
class SessionContextSizingTest {

  /** Qwen3.5's declared context, the one that produced the failure. */
  private static final int LONG_CONTEXT = 262_144;

  /**
   * A per-model recommendation, with no deployment override in play.
   *
   * <p>{@code fromSystemProperties} folds in any real deployment property, so the assertion would
   * read whatever the surrounding JVM happened to set. Checking that it is absent keeps a green run
   * from meaning "the ambient configuration agreed with us".
   */
  private static PureJavaPlanConfiguration recommending(int maxContextLength) {
    assertThat(System.getProperty(PureJavaPlanConfiguration.MAX_CONTEXT_LENGTH_PROPERTY))
        .as("a deployment override would mask the recommendation under test")
        .isNull();
    return PureJavaPlanConfiguration.fromSystemProperties(
        Map.of(
            PureJavaPlanConfiguration.MAX_CONTEXT_LENGTH_PROPERTY,
            Integer.toString(maxContextLength)));
  }

  @Test
  void anUnconfiguredLongContextModelIsBoundedRatherThanTakingItsDeclaredCeiling() {
    int capacity =
        PureJavaBackend.runtimeContextLength(LONG_CONTEXT, PureJavaPlanConfiguration.defaults());

    assertThat(capacity).isEqualTo(PureJavaBackend.DEFAULT_MAXIMUM_CONTEXT);
    assertThat(capacity).isLessThan(LONG_CONTEXT);
  }

  @Test
  void aModelAlreadyWithinTheDefaultIsUnaffected() {
    // Most of the catalog is here, and this is the assertion that says so: bounding the default
    // must not quietly shorten a model that never reached the bound.
    for (int declared : new int[] {512, 2048, 4096, 8192}) {
      assertThat(
              PureJavaBackend.runtimeContextLength(declared, PureJavaPlanConfiguration.defaults()))
          .as("declared context %s", declared)
          .isEqualTo(declared);
    }
  }

  @Test
  void anExplicitRequestWinsOverTheDefaultInBothDirections() {
    // Up: a caller that wants the long context asks for it and pays for it.
    assertThat(PureJavaBackend.runtimeContextLength(LONG_CONTEXT, recommending(131_072)))
        .isEqualTo(131_072);
    // Down: and a caller that knows its prompts are short can go below the default.
    assertThat(PureJavaBackend.runtimeContextLength(LONG_CONTEXT, recommending(2048)))
        .isEqualTo(2048);
  }

  @Test
  void anExplicitRequestNeverExceedsWhatTheModelCanAttendTo() {
    // Asking for more than the model declares is not a licence to allocate it: the cache would be
    // addressable beyond any position the model can represent.
    assertThat(PureJavaBackend.runtimeContextLength(4096, recommending(131_072))).isEqualTo(4096);
  }

  @Test
  void theDefaultKeepsALongContextCacheWithinAWorkableHeap() {
    // The arithmetic the default exists for, stated so a future change to it has to argue with a
    // number. Qwen3.5's full-attention cache measures about 17 GB over 262,144 tokens, so roughly
    // 68 KiB per token for keys and values together.
    long bytesPerToken = 17L * 1024 * 1024 * 1024 / LONG_CONTEXT;
    long atDefault = bytesPerToken * PureJavaBackend.DEFAULT_MAXIMUM_CONTEXT;

    // Comfortably inside a 1 GiB allowance, where the declared ceiling needed seventeen.
    assertThat(atDefault).isLessThan(1024L * 1024 * 1024);
  }
}
