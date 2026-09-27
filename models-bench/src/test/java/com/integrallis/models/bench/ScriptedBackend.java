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
package com.integrallis.models.bench;

import com.integrallis.models.api.InferenceBackend;
import com.integrallis.models.api.ModelMetadata;
import com.integrallis.models.api.Tokenizer;
import java.util.List;
import java.util.Objects;

/**
 * A backend that emits a fixed token id sequence, so the CUDA gates can be exercised with no model
 * and no GPU.
 *
 * <p>{@code reset()} restarts the script, which is what makes two arms comparable: every prompt
 * replays the same sequence, so a divergence in a test is one the test put there.
 *
 * <p>{@code nanosPerForward} busy-waits rather than sleeping. G4's threshold is a ratio of measured
 * throughputs, and a test of that arithmetic wants a deterministic elapsed time; {@code
 * Thread.sleep} overshoots by whatever the scheduler feels like and would make the ratio, and so
 * the test, intermittent.
 */
final class ScriptedBackend implements InferenceBackend {

  private static final int VOCAB_SIZE = 128;
  private static final int PROMPT_TOKENS = 4;

  private final List<Integer> script;
  private final int eosToken;
  private final long nanosPerForward;
  private int cursor;
  private int forwardCalls;
  private int prefillCalls;
  private boolean closed;

  ScriptedBackend(List<Integer> script, int eosToken) {
    this(script, eosToken, 0L);
  }

  ScriptedBackend(List<Integer> script, int eosToken, long nanosPerForward) {
    this.script = List.copyOf(Objects.requireNonNull(script, "script"));
    this.eosToken = eosToken;
    this.nanosPerForward = nanosPerForward;
  }

  int forwardCalls() {
    return forwardCalls;
  }

  int prefillCalls() {
    return prefillCalls;
  }

  boolean closed() {
    return closed;
  }

  @Override
  public String name() {
    return "scripted";
  }

  @Override
  public ModelMetadata metadata() {
    return new ModelMetadata("scripted", "test", 4_096, VOCAB_SIZE, 8, 2, 2, 1);
  }

  @Override
  public Tokenizer tokenizer() {
    return new Tokenizer() {
      @Override
      public int[] encode(String text) {
        int[] tokens = new int[PROMPT_TOKENS];
        for (int index = 0; index < tokens.length; index++) {
          tokens[index] = 1 + index;
        }
        return tokens;
      }

      @Override
      public String decode(int[] tokens) {
        return Integer.toString(tokens.length);
      }

      @Override
      public String decode(int token) {
        return Integer.toString(token);
      }

      @Override
      public int bosToken() {
        return -1;
      }

      @Override
      public int eosToken() {
        return eosToken;
      }

      @Override
      public int vocabSize() {
        return VOCAB_SIZE;
      }
    };
  }

  @Override
  public float[] prefill(int[] tokens, int startPosition) {
    prefillCalls++;
    spin();
    return logitsFavouring(next());
  }

  @Override
  public float[] forward(int token, int position) {
    forwardCalls++;
    spin();
    return logitsFavouring(next());
  }

  @Override
  public void reset() {
    cursor = 0;
  }

  @Override
  public void close() {
    closed = true;
  }

  private int next() {
    int token = script.get(cursor % script.size());
    cursor++;
    return token;
  }

  private float[] logitsFavouring(int token) {
    float[] logits = new float[VOCAB_SIZE];
    logits[token] = 1.0f;
    return logits;
  }

  private void spin() {
    if (nanosPerForward <= 0) {
      return;
    }
    long deadline = System.nanoTime() + nanosPerForward;
    while (System.nanoTime() < deadline) {
      Thread.onSpinWait();
    }
  }
}
