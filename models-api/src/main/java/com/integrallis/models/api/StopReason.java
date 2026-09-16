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

/**
 * Why a text generation ended normally.
 *
 * <p>A generation that fails reports its failure through {@link TokenStream#onError(Throwable)} and
 * has no stop reason. When several conditions hold on the same token, the producer reports the
 * first one in declaration order, so a model-chosen end (end of generation, a stop sequence, or a
 * completed constraint) is never reported as a truncation.
 */
public enum StopReason {
  /** The model sampled a token from its end-of-generation set (EOS, EOT, EOM and similar). */
  EOS,

  /** The decoded output reached a caller-supplied stop sequence. */
  STOP_SEQUENCE,

  /**
   * A token-level constraint, such as a grammar or tool-call schema, reached an accepting state.
   */
  CONSTRAINT_COMPLETE,

  /**
   * The runtime's repetition-loop detector found the output repeating a span and stopped it.
   *
   * <p>The output is truncated at the detected loop; callers should treat it as degenerate rather
   * than as a complete answer.
   */
  REPETITION_LOOP,

  /** The consumer cancelled the generation through {@link TokenStream#isCancelled()}. */
  CANCELLED,

  /** Generation produced the requested maximum number of tokens. */
  MAX_TOKENS
}
