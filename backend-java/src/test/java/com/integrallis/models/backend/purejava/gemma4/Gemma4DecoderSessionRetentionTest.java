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
package com.integrallis.models.backend.purejava.gemma4;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.withSettings;

import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.util.concurrent.atomic.AtomicBoolean;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

@Tag("unit")
class Gemma4DecoderSessionRetentionTest {
  @ParameterizedTest
  @ValueSource(booleans = {false, true})
  void releasesBorrowedSessionsOnSuccessValidationAndInferenceFailure(boolean transientOutput)
      throws Exception {
    AtomicBoolean fail = new AtomicBoolean();
    Gemma4ForwardPass graph =
        mock(
            Gemma4ForwardPass.class,
            withSettings()
                .defaultAnswer(
                    invocation -> {
                      if (invocation.getMethod().getName().equals("openSession"))
                        return mock(Gemma4ForwardPass.Session.class);
                      if (fail.get()) throw new IllegalStateException("inference failed");
                      return null;
                    }));
    Constructor<Gemma4Decoder> constructor =
        Gemma4Decoder.class.getDeclaredConstructor(
            Gemma4Config.class, Gemma4Experts.class, Gemma4ForwardPass.class);
    constructor.setAccessible(true);
    try (Gemma4Decoder decoder =
        constructor.newInstance(mock(Gemma4Config.class), mock(Gemma4Experts.class), graph)) {
      Gemma4Decoder.Session first = decoder.openSession();
      Gemma4Decoder.Session second = decoder.openSession();
      invoke(decoder, transientOutput, new Gemma4Decoder.Session[] {first, second});
      Object[] scratch = sessionScratch(decoder);
      assertThat(scratch).hasSize(2).containsOnlyNulls();
      assertThatThrownBy(
              () -> invoke(decoder, transientOutput, new Gemma4Decoder.Session[] {first, null}))
          .isInstanceOf(NullPointerException.class);
      assertThat(sessionScratch(decoder)).isSameAs(scratch).containsOnlyNulls();
      fail.set(true);
      assertThatThrownBy(
              () -> invoke(decoder, transientOutput, new Gemma4Decoder.Session[] {first, second}))
          .isInstanceOf(IllegalStateException.class)
          .hasMessage("inference failed");
      assertThat(sessionScratch(decoder)).isSameAs(scratch).containsOnlyNulls();
    }
  }

  private static void invoke(
      Gemma4Decoder decoder, boolean transientOutput, Gemma4Decoder.Session[] sessions) {
    if (transientOutput) decoder.forwardBatchTransient(sessions, new int[] {1, 2});
    else decoder.forwardBatch(sessions, new int[] {1, 2});
  }

  private static Object[] sessionScratch(Gemma4Decoder decoder) throws Exception {
    Field field = Gemma4Decoder.class.getDeclaredField("sessionBatch");
    field.setAccessible(true);
    return (Object[]) field.get(decoder);
  }
}
