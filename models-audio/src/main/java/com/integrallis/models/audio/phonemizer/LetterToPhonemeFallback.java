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

import com.integrallis.models.backend.onnx.OnnxGraphExecutor;
import com.integrallis.models.backend.onnx.OnnxModel;
import com.integrallis.models.backend.onnx.Tensor;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/**
 * Letter-to-phoneme model for words the misaki lexicon path cannot resolve: the permissively
 * licensed stand-in for the GPL eSpeak fallback the publisher's pipeline uses.
 *
 * <p>The model is ours, trained on misaki 0.9.4's gold and silver dictionaries (Apache-2.0) with a
 * CTC objective so that the exported ONNX graph has no decoder loop: one run per word, the argmax
 * path collapsed. Training recipe, held-out accuracy and the symbol tables live beside the graph
 * ({@code g2p-<accent>.json}); the graph runs on {@code models-backend-onnx}, never ONNX Runtime.
 * Letters the table does not know (digits, accents) are dropped before the run; a word with no
 * known letter yields {@code null}, as a lexicon miss does.
 */
public final class LetterToPhonemeFallback {

  /**
   * Classpath directory holding the bundled models, beside the dictionaries they were trained on.
   */
  public static final String BUNDLED_RESOURCE_ROOT = MisakiLexicon.BUNDLED_RESOURCE_ROOT;

  /** misaki's rating for a fallback pronunciation. */
  public static final int RATING = 2;

  private static final String SIBILANTS = "szʃʒʧʤ";
  private static final String VOICELESS = "ptkfθ";

  private final OnnxGraphExecutor executor;
  private final Map<Character, Long> letterIndex;
  private final String[] symbols;
  private final int upsample;
  private final boolean british;
  private final Map<String, Object> card;
  private final Map<String, String> dictionary;

  private LetterToPhonemeFallback(
      OnnxModel model, Map<String, Object> card, Map<String, String> dictionary) {
    this.executor = new OnnxGraphExecutor(model);
    this.card = Map.copyOf(card);
    this.dictionary = dictionary == null ? Map.of() : dictionary;
    List<String> letters = strings(card.get("letters"), "letters");
    this.letterIndex = new HashMap<>();
    for (int i = 0; i < letters.size(); i++) {
      letterIndex.put(letters.get(i).charAt(0), (long) (i + 1));
    }
    this.symbols = strings(card.get("symbols"), "symbols").toArray(new String[0]);
    this.upsample = ((Number) Objects.requireNonNull(card.get("upsample"), "upsample")).intValue();
    this.british = "gb".equals(card.get("accent"));
  }

  /**
   * Loads the bundled model for one accent, with the bundled dictionary tier when the accent has
   * one ({@code cmudict-us.json}: CMUdict's words that misaki's dictionaries lack, converted to
   * misaki's American spelling; BSD-2-Clause, see META-INF/NOTICE). A word found there is answered
   * from it; the model answers the rest.
   */
  public static LetterToPhonemeFallback bundled(boolean british) throws IOException {
    String accent = british ? "gb" : "us";
    byte[] dictionary = optionalResource("cmudict-" + accent + ".json");
    return new LetterToPhonemeFallback(
        OnnxModel.parse(resource("g2p-" + accent + ".onnx")),
        JsonAccess.object(new String(resource("g2p-" + accent + ".json"), StandardCharsets.UTF_8)),
        dictionary == null ? null : stringMap(new String(dictionary, StandardCharsets.UTF_8)));
  }

  /** Loads a model from an ONNX graph and its symbol-table card, with no dictionary tier. */
  public static LetterToPhonemeFallback load(Path onnx, Path card) throws IOException {
    return new LetterToPhonemeFallback(
        OnnxModel.load(onnx),
        JsonAccess.object(Files.readString(card, StandardCharsets.UTF_8)),
        null);
  }

  /** Loads a model with a dictionary tier ({@code word -> phonemes} JSON) consulted first. */
  public static LetterToPhonemeFallback load(Path onnx, Path card, Path dictionary)
      throws IOException {
    return new LetterToPhonemeFallback(
        OnnxModel.load(onnx),
        JsonAccess.object(Files.readString(card, StandardCharsets.UTF_8)),
        stringMap(Files.readString(dictionary, StandardCharsets.UTF_8)));
  }

  private static Map<String, String> stringMap(String json) {
    Map<String, Object> raw = JsonAccess.object(json);
    Map<String, String> out = new HashMap<>(raw.size() * 2);
    for (Map.Entry<String, Object> e : raw.entrySet()) {
      out.put(e.getKey(), String.valueOf(e.getValue()));
    }
    return out;
  }

  private static byte[] optionalResource(String name) throws IOException {
    String path = BUNDLED_RESOURCE_ROOT + name;
    try (InputStream in =
        LetterToPhonemeFallback.class.getClassLoader().getResourceAsStream(path)) {
      return in == null ? null : in.readAllBytes();
    }
  }

  /** The number of words the dictionary tier holds; zero when the accent has none. */
  public int dictionaryWords() {
    return dictionary.size();
  }

  /**
   * The dictionary tier's answer for {@code word}, or {@code null}: the word itself, or a
   * possessive or plural of a word it has, inflected by the suffix rule misaki's own stems use
   * ({@code ᵻz} after a sibilant, {@code s} after a voiceless consonant, {@code z} otherwise).
   */
  String lookup(String word) {
    String w = word.toLowerCase(Locale.ROOT);
    String direct = dictionary.get(w);
    if (direct != null) {
      return direct;
    }
    String base = null;
    if (w.endsWith("'s") && w.length() > 2) {
      base = dictionary.get(w.substring(0, w.length() - 2));
    } else if (w.endsWith("s'") && w.length() > 2) {
      base = dictionary.get(w.substring(0, w.length() - 2));
    } else if (w.endsWith("es") && w.length() > 2) {
      // "-es" after a sibilant (bushes); otherwise the base keeps its own "e" (names, cases).
      String stem = dictionary.get(w.substring(0, w.length() - 2));
      if (stem != null
          && !stem.isEmpty()
          && SIBILANTS.indexOf(stem.charAt(stem.length() - 1)) >= 0) {
        return stem + "ᵻz";
      }
      base = dictionary.get(w.substring(0, w.length() - 1));
    } else if (w.endsWith("s") && w.length() > 1) {
      base = dictionary.get(w.substring(0, w.length() - 1));
    }
    if (base == null || base.isEmpty()) {
      return null;
    }
    char last = base.charAt(base.length() - 1);
    if (SIBILANTS.indexOf(last) >= 0) {
      return base + "ᵻz";
    }
    return base + (VOICELESS.indexOf(last) >= 0 ? "s" : "z");
  }

  private static List<String> strings(Object value, String name) {
    if (!(value instanceof List<?> list)) {
      throw new IllegalArgumentException("model card field " + name + " must be an array");
    }
    List<String> out = new ArrayList<>(list.size());
    for (Object o : list) {
      out.add(String.valueOf(o));
    }
    return out;
  }

  private static byte[] resource(String name) throws IOException {
    String path = BUNDLED_RESOURCE_ROOT + name;
    try (InputStream in =
        LetterToPhonemeFallback.class.getClassLoader().getResourceAsStream(path)) {
      if (in == null) {
        throw new IOException(
            "Bundled letter-to-phoneme model missing from the classpath: " + path);
      }
      return in.readAllBytes();
    }
  }

  /** Whether this model spells British English. */
  public boolean british() {
    return british;
  }

  /** The model's card: training source, held-out word accuracy and phoneme error rate. */
  public Map<String, Object> card() {
    return card;
  }

  /**
   * The phoneme string for {@code word}: the dictionary tier's if it has the word, else the
   * model's; {@code null} when the word has no letter the model knows.
   */
  public synchronized String phonemize(String word) {
    Objects.requireNonNull(word, "word");
    String fromDictionary = lookup(word);
    if (fromDictionary != null) {
      return fromDictionary;
    }
    List<Long> ids = new ArrayList<>();
    for (char c : word.toLowerCase(Locale.ROOT).toCharArray()) {
      Long id = letterIndex.get(c);
      if (id != null) {
        ids.add(id);
      }
    }
    if (ids.isEmpty()) {
      return null;
    }
    long[] letters = new long[ids.size()];
    for (int i = 0; i < letters.length; i++) {
      letters[i] = ids.get(i);
    }
    Tensor logits =
        executor
            .run(Map.of("letters", Tensor.ofLongs(new long[] {1, letters.length}, letters)))
            .get("logits");
    long[] shape = logits.shape();
    int frames = (int) shape[1];
    int classes = (int) shape[2];
    float[] values = logits.floats();
    int[] path = new int[frames];
    for (int t = 0; t < frames; t++) {
      int best = 0;
      float bestValue = values[t * classes];
      for (int k = 1; k < classes; k++) {
        float v = values[t * classes + k];
        if (v > bestValue) {
          bestValue = v;
          best = k;
        }
      }
      path[t] = best;
    }
    String ps = collapse(path, symbols);
    return ps.isEmpty() ? null : ps;
  }

  /** Collapses a CTC argmax path: repeats merge, blanks (class 0) separate and vanish. */
  static String collapse(int[] path, String[] symbols) {
    StringBuilder out = new StringBuilder();
    int previous = 0;
    for (int k : path) {
      if (k != previous && k != 0) {
        out.append(symbols[k - 1]);
      }
      previous = k;
    }
    return out.toString();
  }

  int upsample() {
    return upsample;
  }
}
