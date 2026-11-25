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
package com.integrallis.models.runtime.fusion;

import com.integrallis.models.api.ModelMetadata;
import com.integrallis.models.api.RewindableInferenceBackend;
import com.integrallis.models.api.Tokenizer;
import java.util.ArrayList;
import java.util.List;

/**
 * A deterministic toy language model whose logits depend on its whole token history, so a member
 * that missed a token produces different logits. Mirrors the bench's {@code ToyMember}.
 */
class ToyBackend implements RewindableInferenceBackend {
  static final int VOCAB = 12;
  static final int EOS = 11;

  private final String name;
  private final double seed;
  private final double offset;
  private final List<Integer> history = new ArrayList<>();
  final List<int[]> prefillDeltas = new ArrayList<>();
  final List<Integer> prefillStarts = new ArrayList<>();
  int forwardCalls;
  int resets;
  boolean closed;

  ToyBackend(String name, double seed, double offset) {
    this.name = name;
    this.seed = seed;
    this.offset = offset;
  }

  @Override
  public String name() {
    return name;
  }

  @Override
  public ModelMetadata metadata() {
    return new ModelMetadata("toy", name, 256, VOCAB, 8, 1, 1, 1);
  }

  @Override
  public Tokenizer tokenizer() {
    return ToyTokenizer.INSTANCE;
  }

  @Override
  public float[] prefill(int[] tokens, int startPosition) {
    requirePosition(startPosition);
    prefillDeltas.add(tokens.clone());
    prefillStarts.add(startPosition);
    for (int token : tokens) {
      history.add(token);
    }
    return logits();
  }

  @Override
  public float[] forward(int token, int position) {
    forwardCalls++;
    requirePosition(position);
    history.add(token);
    return logits();
  }

  @Override
  public void reset() {
    resets++;
    history.clear();
  }

  @Override
  public int checkpoint() {
    return history.size();
  }

  @Override
  public void rewind(int checkpoint) {
    while (history.size() > checkpoint) {
      history.removeLast();
    }
  }

  @Override
  public void close() {
    closed = true;
  }

  /** Logits this toy would produce after exactly {@code tokens}, without touching its state. */
  static float[] logitsAfter(double seed, double offset, List<Integer> tokens) {
    ToyBackend fresh = new ToyBackend("fresh", seed, offset);
    fresh.history.addAll(tokens);
    return fresh.logits();
  }

  private void requirePosition(int position) {
    if (position != history.size()) {
      throw new IllegalStateException("position " + position + " != " + history.size());
    }
  }

  private float[] logits() {
    long sum = 0;
    for (int i = 0; i < history.size(); i++) {
      sum += (long) history.get(i) * (i + 3);
    }
    float[] logits = new float[VOCAB];
    for (int t = 0; t < VOCAB; t++) {
      logits[t] = (float) (offset + 3.0 * Math.sin(seed * (t + 1) + 0.37 * sum + history.size()));
    }
    logits[EOS] += (float) (0.35 * history.size() - 3.0);
    return logits;
  }

  /** Tokens decode to single characters: digits 0-9, a space, and an empty end token. */
  static final class ToyTokenizer implements Tokenizer {
    static final ToyTokenizer INSTANCE = new ToyTokenizer();

    @Override
    public int[] encode(String text) {
      int[] tokens = new int[text.length()];
      for (int i = 0; i < text.length(); i++) {
        char c = text.charAt(i);
        tokens[i] = Character.isDigit(c) ? c - '0' : 10;
      }
      return tokens;
    }

    @Override
    public String decode(int[] tokens) {
      StringBuilder text = new StringBuilder();
      for (int token : tokens) {
        text.append(decode(token));
      }
      return text.toString();
    }

    @Override
    public String decode(int token) {
      if (token == EOS) {
        return "";
      }
      return token == 10 ? " " : Integer.toString(token);
    }

    @Override
    public int vocabSize() {
      return VOCAB;
    }

    @Override
    public int bosToken() {
      return 10;
    }

    @Override
    public int eosToken() {
      return EOS;
    }
  }
}
