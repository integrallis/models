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

/**
 * Grapheme-to-phoneme conversion for English in the spelling of <a
 * href="https://github.com/hexgrad/misaki">misaki</a>, the G2P library Kokoro and its distillations
 * were trained with.
 *
 * <p>{@link com.integrallis.models.audio.phonemizer.MisakiEnglishPhonemizer} is a Java port of
 * misaki's English lexicon path at revision {@code fba1236}: its gold and silver dictionaries, stem
 * rules, number reading, stress rules and compound handling. Two parts of the original are
 * deliberately absent. The spaCy tokenizer and part-of-speech tagger are replaced by {@link
 * com.integrallis.models.audio.phonemizer.EnglishTokenizer}, a rule-based approximation, and there
 * is no eSpeak fallback for words outside the lexicon, because eSpeak NG is GPL-licensed. Where the
 * two differ from the original is measured, not assumed: the qualification record for each model
 * that uses this package reports the phoneme-string agreement against the publisher's own pipeline.
 */
package com.integrallis.models.audio.phonemizer;
