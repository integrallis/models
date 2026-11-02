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
package com.integrallis.models.audio.phonemizer;

/**
 * A mutable token in the phonemizer pipeline, mirroring misaki's {@code MToken} and the private
 * attributes it hangs off it. Package-private: the pipeline rewrites these in place.
 */
final class MToken {

  String text;
  String tag;
  String whitespace;
  String phonemes;
  boolean isHead = true;
  String alias;
  Double stress;
  String currency;
  String numFlags = "";
  boolean prespace;
  Integer rating;

  /** Set when the letter-to-phoneme fallback, not the lexicon, produced the phonemes. */
  boolean fallback;

  MToken(String text, String tag, String whitespace) {
    this.text = text;
    this.tag = tag;
    this.whitespace = whitespace;
  }

  MToken copy() {
    MToken copy = new MToken(text, tag, whitespace);
    copy.phonemes = phonemes;
    copy.isHead = isHead;
    copy.alias = alias;
    copy.stress = stress;
    copy.currency = currency;
    copy.numFlags = numFlags;
    copy.prespace = prespace;
    copy.rating = rating;
    return copy;
  }

  @Override
  public String toString() {
    return text + "/" + tag + (phonemes == null ? "" : "=" + phonemes);
  }
}
