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
package com.integrallis.models.runtime;

import com.integrallis.models.api.InferenceBackend;
import com.integrallis.models.api.SamplingOptions;
import com.integrallis.models.api.TokenStream;
import com.integrallis.models.api.Tokenizer;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Autoregressive generation loop that drives an inference backend to generate text.
 *
 * <p>Generation resets request-specific backend state before prefill. Calls that share the same
 * backend instance are serialized because stateful backends may own a single key-value cache.
 */
public final class GenerationLoop {

  private final InferenceBackend backend;

  public GenerationLoop(InferenceBackend backend) {
    this.backend = Objects.requireNonNull(backend, "backend");
  }

  /** Generates text from a prompt, returning the complete generated string. */
  public String generate(String prompt, SamplingOptions options) {
    Objects.requireNonNull(prompt, "prompt");
    if (prompt.isEmpty()) {
      throw new IllegalArgumentException("prompt must not be empty");
    }
    Objects.requireNonNull(options, "options");

    StringBuilder result = new StringBuilder();
    generate(
        prompt,
        options,
        new TokenStream() {
          @Override
          public void onToken(String token) {
            result.append(token);
          }

          @Override
          public void onComplete() {}

          @Override
          public void onError(Throwable t) {
            throw new RuntimeException("Generation error", t);
          }
        });
    return result.toString();
  }

  /** Generates text with streaming output via a TokenStream callback. */
  public void generate(String prompt, SamplingOptions options, TokenStream stream) {
    Objects.requireNonNull(prompt, "prompt");
    if (prompt.isEmpty()) {
      throw new IllegalArgumentException("prompt must not be empty");
    }
    Objects.requireNonNull(options, "options");
    Objects.requireNonNull(stream, "stream");

    synchronized (backend) {
      backend.reset();
      Tokenizer tokenizer = backend.tokenizer();
      int[] promptTokens = tokenizer.encode(prompt);
      if (promptTokens.length == 0) {
        throw new IllegalArgumentException("prompt produced no tokens");
      }
      int eosToken = tokenizer.eosToken();

      Sampler sampler = new Sampler(options);
      List<Integer> allTokens = new ArrayList<>();

      try {
        float[] logits = backend.prefill(promptTokens, 0);
        for (int token : promptTokens) {
          allTokens.add(token);
        }
        int position = promptTokens.length;

        // Autoregressive decoding
        int generated = 0;
        while (generated < options.maxTokens()) {
          int nextToken = sampler.sample(logits, allTokens);

          if (nextToken == eosToken) {
            break;
          }

          String tokenStr = tokenizer.decode(nextToken);
          stream.onToken(tokenStr);
          allTokens.add(nextToken);

          logits = backend.forwardTransient(nextToken, position);
          position++;
          generated++;
        }

        stream.onComplete();
      } catch (Exception e) {
        stream.onError(e);
      }
    }
  }
}
