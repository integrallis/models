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

import java.util.Objects;
import java.util.Optional;

/**
 * Per-request transcription settings.
 *
 * <p>Deliberately small. The TDT and CTC decoders a {@code parakeet}-class model carries are driven
 * by metadata the artifact publishes — {@code decoding.max_symbols}, {@code blank_id}, the mel front
 * end's window and hop — not by caller preference, so exposing those here would invite a caller to
 * contradict the file. What a caller legitimately chooses is the language hint, whether to ask for
 * the CTC head where a model ships one, and a cap on how long a transcript may run.
 */
public final class TranscriptionOptions {

  private final String language;
  private final Head head;
  private final int maximumTokens;

  private TranscriptionOptions(Builder builder) {
    this.language = builder.language;
    this.head = builder.head;
    this.maximumTokens = builder.maximumTokens;
  }

  /**
   * Which decoder head to read.
   *
   * <p>A hybrid artifact may carry more than one. {@code parakeet-tdt_ctc-110m} ships a
   * {@code ctc_decoder} tensor root and {@code parakeet-tdt-0.6b-v3} does not, although both declare
   * {@code arch = hybrid_tdt_ctc} — so the declared architecture does not say which heads exist, and
   * {@link #MODEL_DEFAULT} lets the loader answer from the tensor set rather than from the string.
   */
  public enum Head {
    /** Whatever the artifact's tensor set provides, preferring the transducer when both exist. */
    MODEL_DEFAULT,
    /** The token-and-duration transducer head. */
    TDT,
    /** The connectionist temporal classification head, where the artifact carries one. */
    CTC
  }

  /** A BCP-47 language hint, or empty to let the model decide. */
  public Optional<String> language() {
    return Optional.ofNullable(language);
  }

  /** Which decoder head to read. */
  public Head head() {
    return head;
  }

  /** Upper bound on emitted tokens; zero means the model's own limit. */
  public int maximumTokens() {
    return maximumTokens;
  }

  /** A new builder with model defaults. */
  public static Builder builder() {
    return new Builder();
  }

  /** Builds {@link TranscriptionOptions}. */
  public static final class Builder {
    private String language;
    private Head head = Head.MODEL_DEFAULT;
    private int maximumTokens;

    private Builder() {}

    /** Sets a BCP-47 language hint. */
    public Builder language(String value) {
      this.language = value;
      return this;
    }

    /** Selects a decoder head. */
    public Builder head(Head value) {
      this.head = Objects.requireNonNull(value, "head");
      return this;
    }

    /** Caps emitted tokens; zero keeps the model's own limit. */
    public Builder maximumTokens(int value) {
      if (value < 0) {
        throw new IllegalArgumentException("maximumTokens must not be negative: " + value);
      }
      this.maximumTokens = value;
      return this;
    }

    /** Builds the options. */
    public TranscriptionOptions build() {
      return new TranscriptionOptions(this);
    }
  }
}
