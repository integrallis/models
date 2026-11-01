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
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.withSettings;

import com.integrallis.models.backend.purejava.gemma4.Gemma4Decoder;
import com.integrallis.models.backend.purejava.llama.LlamaForwardPass;
import java.lang.reflect.Field;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.stream.Stream;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.mockito.stubbing.Answer;

@Tag("unit")
class DecoderSessionRetentionTest {
  static Stream<Arguments> batchOperations() {
    return Stream.of(
        Arguments.of("llama", "prefill"),
        Arguments.of("llama", "hidden"),
        Arguments.of("llama", "prefillTransient"),
        Arguments.of("llama", "forward"),
        Arguments.of("llama", "forwardTransient"),
        Arguments.of("gemma4", "forward"),
        Arguments.of("gemma4", "forwardTransient"));
  }

  @ParameterizedTest
  @MethodSource("batchOperations")
  void releasesBorrowedSessionsAfterSuccessAndDelegateFailure(String family, String operation)
      throws Exception {
    AtomicBoolean fail = new AtomicBoolean();
    PureJavaDecoder decoder = decoder(family, fail);
    PureJavaDecoder.Session first = decoder.openSession();
    PureJavaDecoder.Session second = decoder.openSession();
    invoke(decoder, operation, new PureJavaDecoder.Session[] {first, second});
    Object[] scratch = sessionScratch(decoder);
    assertThat(scratch).hasSize(2).containsOnlyNulls();

    // Reuse storage, while making the delegate fail after borrowing both sessions.
    fail.set(true);
    assertThatThrownBy(
            () -> invoke(decoder, operation, new PureJavaDecoder.Session[] {first, second}))
        .isInstanceOf(IllegalStateException.class)
        .hasMessage("inference failed");
    assertThat(sessionScratch(decoder)).isSameAs(scratch).containsOnlyNulls();
  }

  @ParameterizedTest
  @MethodSource("batchOperations")
  void releasesPartiallyUnwrappedBatchWhenValidationFails(String family, String operation)
      throws Exception {
    PureJavaDecoder decoder = decoder(family, new AtomicBoolean());
    PureJavaDecoder.Session first = decoder.openSession();
    assertThatThrownBy(
            () -> invoke(decoder, operation, new PureJavaDecoder.Session[] {first, null}))
        .isInstanceOf(NullPointerException.class);
    assertThat(sessionScratch(decoder)).hasSize(2).containsOnlyNulls();
  }

  private static PureJavaDecoder decoder(String family, AtomicBoolean fail) {
    Class<?> sessionType =
        family.equals("llama") ? LlamaForwardPass.Session.class : Gemma4Decoder.Session.class;
    Answer<Object> answer =
        invocation -> {
          if (invocation.getMethod().getName().equals("openSession")) return mock(sessionType);
          if (fail.get()) throw new IllegalStateException("inference failed");
          return null;
        };
    return family.equals("llama")
        ? new LlamaDecoder(mock(LlamaForwardPass.class, withSettings().defaultAnswer(answer)))
        : new Gemma4DecoderAdapter(mock(Gemma4Decoder.class, withSettings().defaultAnswer(answer)));
  }

  private static void invoke(
      PureJavaDecoder decoder, String operation, PureJavaDecoder.Session[] sessions) {
    switch (operation) {
      case "prefill" -> decoder.prefillBatch(sessions, new int[][] {{1}, {2}});
      case "hidden" -> decoder.prefillBatchHiddenStates(sessions, new int[][] {{1}, {2}});
      case "prefillTransient" -> decoder.prefillBatchTransient(sessions, new int[][] {{1}, {2}});
      case "forward" -> decoder.forwardBatch(sessions, new int[] {1, 2});
      case "forwardTransient" -> decoder.forwardBatchTransient(sessions, new int[] {1, 2});
      default -> throw new IllegalArgumentException(operation);
    }
  }

  private static Object[] sessionScratch(PureJavaDecoder decoder) throws Exception {
    // Inspect strong references directly: a GC-based leak assertion would be nondeterministic.
    Field field = decoder.getClass().getDeclaredField("sessionBatch");
    field.setAccessible(true);
    return (Object[]) field.get(decoder);
  }
}
