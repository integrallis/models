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

/** Portable controls shared by text-to-speech and voice-cloning models. */
public record SpeechSynthesisOptions(
    SamplingOptions sampling,
    String language,
    String voice,
    float speed,
    PcmAudio referenceAudio,
    String referenceText) {

  public SpeechSynthesisOptions {
    language = optionalText(language, "language");
    voice = optionalText(voice, "voice");
    referenceText = optionalText(referenceText, "referenceText");
    if (!(speed > 0.0f) || !Float.isFinite(speed)) {
      throw new IllegalArgumentException("speed must be finite and positive: " + speed);
    }
    if (referenceText != null && referenceAudio == null) {
      throw new IllegalArgumentException("referenceText requires referenceAudio");
    }
  }

  /** Creates a builder whose omitted values are resolved by the selected model. */
  public static Builder builder() {
    return new Builder();
  }

  private static String optionalText(String value, String name) {
    if (value == null) {
      return null;
    }
    if (value.isBlank()) {
      throw new IllegalArgumentException(name + " must not be blank");
    }
    return value.trim();
  }

  /** Builder for {@link SpeechSynthesisOptions}. */
  public static final class Builder {
    private SamplingOptions sampling;
    private String language;
    private String voice;
    private float speed = 1.0f;
    private PcmAudio referenceAudio;
    private String referenceText;

    Builder() {}

    public Builder sampling(SamplingOptions sampling) {
      this.sampling = Objects.requireNonNull(sampling, "sampling");
      return this;
    }

    public Builder language(String language) {
      this.language = language;
      return this;
    }

    public Builder voice(String voice) {
      this.voice = voice;
      return this;
    }

    public Builder speed(float speed) {
      this.speed = speed;
      return this;
    }

    public Builder referenceAudio(PcmAudio referenceAudio) {
      this.referenceAudio = Objects.requireNonNull(referenceAudio, "referenceAudio");
      return this;
    }

    public Builder referenceText(String referenceText) {
      this.referenceText = referenceText;
      return this;
    }

    public SpeechSynthesisOptions build() {
      return new SpeechSynthesisOptions(
          sampling, language, voice, speed, referenceAudio, referenceText);
    }
  }
}
