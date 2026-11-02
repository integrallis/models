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

  private final OnnxGraphExecutor executor;
  private final Map<Character, Long> letterIndex;
  private final String[] symbols;
  private final int upsample;
  private final boolean british;
  private final Map<String, Object> card;

  private LetterToPhonemeFallback(OnnxModel model, Map<String, Object> card) {
    this.executor = new OnnxGraphExecutor(model);
    this.card = Map.copyOf(card);
    List<String> letters = strings(card.get("letters"), "letters");
    this.letterIndex = new HashMap<>();
    for (int i = 0; i < letters.size(); i++) {
      letterIndex.put(letters.get(i).charAt(0), (long) (i + 1));
    }
    this.symbols = strings(card.get("symbols"), "symbols").toArray(new String[0]);
    this.upsample = ((Number) Objects.requireNonNull(card.get("upsample"), "upsample")).intValue();
    this.british = "gb".equals(card.get("accent"));
  }

  /** Loads the bundled model for one accent. */
  public static LetterToPhonemeFallback bundled(boolean british) throws IOException {
    String accent = british ? "gb" : "us";
    return new LetterToPhonemeFallback(
        OnnxModel.parse(resource("g2p-" + accent + ".onnx")),
        JsonAccess.object(new String(resource("g2p-" + accent + ".json"), StandardCharsets.UTF_8)));
  }

  /** Loads a model from an ONNX graph and its symbol-table card. */
  public static LetterToPhonemeFallback load(Path onnx, Path card) throws IOException {
    return new LetterToPhonemeFallback(
        OnnxModel.load(onnx), JsonAccess.object(Files.readString(card, StandardCharsets.UTF_8)));
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

  /** The phoneme string for {@code word}, or {@code null} when it has no letter the model knows. */
  public synchronized String phonemize(String word) {
    Objects.requireNonNull(word, "word");
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
