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
package com.integrallis.models.decisions;

import com.integrallis.models.api.InferenceSession;
import com.integrallis.models.api.LogitBatch;
import com.integrallis.models.api.ModelMetadata;
import com.integrallis.models.api.SharedInferencePrefix;
import com.integrallis.models.api.SharedPrefixInferenceBackend;
import com.integrallis.models.api.Tokenizer;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * A backend whose logits are a pure function of every token the session has seen, with a
 * one-character-per-token tokenizer.
 *
 * <p>That makes a prompt observable: two paths return the same logits exactly when they fed the
 * model the same text, so a test can tell a shared prefix from a re-read, and a changed prompt from
 * an unchanged one, without a real model.
 */
final class ScriptedPrefixBackend implements SharedPrefixInferenceBackend {

  static final int VOCAB = 256;

  /** Every complete text a session was scored on, in order. */
  final List<String> scoredPrompts = new ArrayList<>();

  int prefillCalls;
  int freezes;
  boolean sharedPrefixes = true;
  boolean proveSharing = true;
  private final Set<InferenceSession> forks = new HashSet<>();

  /** The logits this backend returns after reading {@code text}. */
  static float[] logitsAfter(String text) {
    float[] logits = new float[VOCAB];
    long seed = 1125899906842597L;
    for (int index = 0; index < text.length(); index++) {
      seed = 31 * seed + text.charAt(index);
    }
    for (int token = 0; token < VOCAB; token++) {
      seed = seed * 6364136223846793005L + 1442695040888963407L;
      logits[token] = (float) ((seed >>> 40) % 2000) / 250.0f - 4.0f;
    }
    return logits;
  }

  private float[] read(InferenceSession session, int[] tokens) {
    Scripted scripted = (Scripted) session;
    for (int token : tokens) {
      scripted.text.append((char) token);
    }
    String all = scripted.text.toString();
    scoredPrompts.add(all);
    return logitsAfter(all);
  }

  @Override
  public String name() {
    return "scripted";
  }

  @Override
  public boolean supportsSharedPrefixes() {
    return sharedPrefixes;
  }

  @Override
  public boolean supportsActivatedBranch() {
    return false;
  }

  @Override
  public int maxBatchSize() {
    return 8;
  }

  @Override
  public InferenceSession openSession() {
    return new Scripted();
  }

  @Override
  public float[] prefill(InferenceSession session, int[] tokens, int startPosition) {
    prefillCalls++;
    if (startPosition != ((Scripted) session).text.length()) {
      throw new AssertionError(
          "prefill at "
              + startPosition
              + " but session holds "
              + ((Scripted) session).text.length());
    }
    return read(session, tokens);
  }

  @Override
  public float[] forward(InferenceSession session, int token, int position) {
    return read(session, new int[] {token});
  }

  @Override
  public SharedInferencePrefix freezePrefix(InferenceSession source) {
    freezes++;
    return new Frozen(((Scripted) source).text.toString());
  }

  @Override
  public InferenceSession fork(SharedInferencePrefix prefix, Branch branch) {
    Scripted fork = new Scripted();
    fork.text.append(((Frozen) prefix).text);
    forks.add(fork);
    return fork;
  }

  @Override
  public boolean sharesPrefixStorage(InferenceSession first, InferenceSession second) {
    return proveSharing && forks.contains(first) && forks.contains(second);
  }

  @Override
  public void reset(InferenceSession session) {
    ((Scripted) session).text.setLength(0);
  }

  @Override
  public void rewind(InferenceSession session, int checkpoint) {
    ((Scripted) session).text.setLength(checkpoint);
  }

  @Override
  public float[] forward(int token, int position) {
    throw new AssertionError("decisions never use the backend's own session");
  }

  @Override
  public LogitBatch forwardBatch(InferenceSession[] sessions, int[] tokens) {
    throw new AssertionError("not used");
  }

  @Override
  public ModelMetadata metadata() {
    throw new UnsupportedOperationException();
  }

  @Override
  public Tokenizer tokenizer() {
    return new Tokenizer() {
      @Override
      public int[] encode(String text) {
        return text.chars().toArray();
      }

      @Override
      public String decode(int[] tokens) {
        return new String(tokens, 0, tokens.length);
      }

      @Override
      public String decode(int token) {
        return String.valueOf((char) token);
      }

      @Override
      public int vocabSize() {
        return VOCAB;
      }

      @Override
      public int bosToken() {
        return 1;
      }

      @Override
      public int eosToken() {
        return 2;
      }
    };
  }

  @Override
  public void close() {}

  private static final class Scripted implements InferenceSession {
    final StringBuilder text = new StringBuilder();
    private boolean closed;

    @Override
    public int checkpoint() {
      return text.length();
    }

    @Override
    public boolean isClosed() {
      return closed;
    }

    @Override
    public void close() {
      closed = true;
    }
  }

  private record Frozen(String text) implements SharedInferencePrefix {
    @Override
    public int checkpoint() {
      return text.length();
    }

    @Override
    public long sharedBytes() {
      return text.length() * 512L;
    }
  }
}
