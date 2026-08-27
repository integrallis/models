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

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.models.api.Tokenizer;
import com.integrallis.models.api.ToolSpec;
import com.integrallis.models.runtime.chat.ToolSyntax;
import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class ToolCallTokenConstraintsTest {

  @Test
  void doesNotApplySingleCallConstraintsToNeedle2ArrayWrappedGeneration() {
    var tool = new ToolSpec("set_mode", "Set the mode", "{}");

    assertThat(
            ToolCallTokenConstraints.compile(
                new UnusedTokenizer(),
                ToolSyntax.NEEDLE2,
                List.of(tool),
                ignored -> List.of("{\"mode\":\"cool\"}")))
        .isEmpty();
  }

  private static final class UnusedTokenizer implements Tokenizer {
    @Override
    public int[] encode(String text) {
      throw new AssertionError("array-wrapped syntax must fall back before tokenization");
    }

    @Override
    public String decode(int[] tokens) {
      throw new AssertionError("not used");
    }

    @Override
    public String decode(int token) {
      throw new AssertionError("not used");
    }

    @Override
    public int vocabSize() {
      return 1;
    }

    @Override
    public int bosToken() {
      return 0;
    }

    @Override
    public int eosToken() {
      return 0;
    }
  }
}
