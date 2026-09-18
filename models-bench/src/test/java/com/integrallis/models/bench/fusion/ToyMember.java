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
package com.integrallis.models.bench.fusion;

import com.integrallis.models.api.Tokenizer;
import java.util.ArrayList;
import java.util.List;

/** A deterministic toy language model whose logits depend on its whole KV history. */
final class ToyMember implements FusionMember {
  static final int VOCAB = 12;
  static final int EOS = 11;

  private final String name;
  private final double seed;
  private final double offset;
  private final List<Integer> history = new ArrayList<>();
  int forwardCalls;
  int prefillCalls;

  ToyMember(String name, double seed, double offset) {
    this.name = name;
    this.seed = seed;
    this.offset = offset;
  }

  @Override
  public String name() {
    return name;
  }

  @Override
  public Tokenizer tokenizer() {
    return ToyTokenizer.INSTANCE;
  }

  @Override
  public void reset() {
    history.clear();
  }

  @Override
  public float[] prefill(int[] tokens, int startPosition) {
    prefillCalls++;
    requirePosition(startPosition);
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
  public int checkpoint() {
    return history.size();
  }

  @Override
  public void rewind(int checkpoint) {
    while (history.size() > checkpoint) {
      history.remove(history.size() - 1);
    }
  }

  @Override
  public float[][] verify(int[] tokens, int startPosition) {
    float[][] rows = new float[tokens.length][];
    for (int i = 0; i < tokens.length; i++) {
      rows[i] = forward(tokens[i], startPosition + i);
    }
    return rows;
  }

  @Override
  public void close() {}

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
    // Make the sequence end eventually.
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
        tokens[i] = c == ' ' ? 10 : Character.isDigit(c) ? c - '0' : 10;
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
