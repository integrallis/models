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

import com.integrallis.models.backend.purejava.fixture.ModelFixtureDescriptor;
import com.integrallis.models.backend.purejava.fixture.ModelFixtureRegistry;
import com.integrallis.models.backend.purejava.fixture.ModelFixtureRequirement;
import com.integrallis.models.backend.purejava.gguf.GgufParser;
import java.lang.foreign.Arena;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * A session opened for a long declared context must not allocate that context.
 *
 * <p>This is the regression gate for #357, where opening Qwen3.5 4B under {@code -Xmx8g} died in
 * {@code SessionState} before reading a token: the key and value caches were allocated for the
 * model's declared 262,144 positions, about 17 GB of fp32 across the full-attention layers.
 *
 * <p>Asserted on allocated bytes rather than on heap usage, because a heap measurement is a
 * measurement of the garbage collector and would either flake or pass for the wrong reason. The
 * session reports what it holds, so the policy is observable -- and an allocation policy that
 * cannot be observed is one no test can hold to account, which is how this shipped in the first
 * place.
 *
 * <p>0.8B rather than 4B because the assertion is about the sizing policy, which does not vary with
 * the parameter count, and a 4B fixture would make the gate too expensive to keep.
 */
@Tag("integration")
class Qwen35SessionGrowthTest {

  private static final ModelFixtureRequirement QWEN35_08B_Q4_K_M =
      ModelFixtureRequirement.of("hf://unsloth/Qwen3.5-0.8B-GGUF")
          .version("[3.5.0,3.6.0)")
          .variant("q4_k_m")
          .backend("pure-java")
          .capability("chat");

  @Test
  void openingAtTheDeclaredCeilingDoesNotAllocateIt() throws Exception {
    ModelFixtureDescriptor descriptor =
        ModelFixtureRegistry.fromClasspath().resolve(QWEN35_08B_Q4_K_M).orElseThrow();

    try (Arena arena = Arena.ofConfined()) {
      Qwen35ForwardPass pass =
          Qwen35ForwardPass.fromGgufFile(
              GgufParser.parse(descriptor.localPath().orElseThrow(), arena));
      Qwen35Config config = pass.config();
      int declared = config.contextLength();

      // The ceiling under test has to be a long one, or this gate proves nothing.
      assertThat(declared)
          .as("fixture must declare a context worth growing into")
          .isGreaterThanOrEqualTo(32_768);

      Qwen35ForwardPass.Session session = pass.openSession(declared);
      assertThat(session.capacity()).isEqualTo(declared);

      long atCeiling = fullAttentionCacheBytes(config, declared);
      long atOpen = session.allocatedCacheBytes();

      // The actual regression: opening is cheap, and stays within a thousandth of the ceiling.
      assertThat(atOpen)
          .as("bytes held at open, against %s at the declared ceiling", atCeiling)
          .isLessThan(atCeiling / 1000);

      // And it grows when a sequence needs it, rather than refusing.
      pass.prefill(session, new int[] {760, 760, 760, 760}, 0);
      long afterFourTokens = session.allocatedCacheBytes();
      assertThat(afterFourTokens).isGreaterThan(0L).isLessThan(atCeiling / 1000);

      for (int step = 0; step < 64; step++) {
        pass.forward(session, 760, session.checkpoint());
      }
      assertThat(session.checkpoint()).isEqualTo(68);
      assertThat(session.allocatedCacheBytes())
          .as("68 positions must be held, and not the whole ceiling")
          .isGreaterThanOrEqualTo(fullAttentionCacheBytes(config, 68))
          .isLessThan(atCeiling / 100);
    }
  }

  /** Bytes the full-attention layers would hold for {@code positions} keys and values. */
  private static long fullAttentionCacheBytes(Qwen35Config config, int positions) {
    long layers = 0;
    for (int layer = 0; layer < config.numLayers(); layer++) {
      if (config.usesFullAttention(layer)) {
        layers++;
      }
    }
    return layers * 2L * positions * config.attentionKeyDim() * Float.BYTES;
  }
}
