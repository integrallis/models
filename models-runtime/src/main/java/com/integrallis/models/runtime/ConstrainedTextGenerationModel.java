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

import com.integrallis.models.api.ModelPrompt;
import com.integrallis.models.api.SamplingOptions;
import com.integrallis.models.api.TextGenerationModel;
import com.integrallis.models.api.TokenStream;
import com.integrallis.models.api.Tokenizer;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;

/** Text generation model that can enforce token constraints during decoding. */
public interface ConstrainedTextGenerationModel extends TextGenerationModel {

  /** Returns the tokenizer whose token ids are passed to {@link TokenConstraint}. */
  Tokenizer tokenizer();

  /** Generates text while enforcing {@code constraint}. */
  default String generate(ModelPrompt prompt, SamplingOptions options, TokenConstraint constraint) {
    Objects.requireNonNull(prompt, "prompt");
    Objects.requireNonNull(options, "options");
    Objects.requireNonNull(constraint, "constraint");
    return collect(stream -> generate(prompt, options, stream, constraint));
  }

  /** Streams generated text while enforcing {@code constraint}. */
  void generate(
      ModelPrompt prompt, SamplingOptions options, TokenStream stream, TokenConstraint constraint);

  private static String collect(Consumer<TokenStream> generation) {
    StringBuilder output = new StringBuilder();
    AtomicReference<Throwable> failure = new AtomicReference<>();
    generation.accept(
        new TokenStream() {
          @Override
          public void onToken(String token) {
            output.append(token);
          }

          @Override
          public void onComplete() {}

          @Override
          public void onError(Throwable throwable) {
            failure.compareAndSet(null, throwable);
          }
        });
    Throwable generationFailure = failure.get();
    if (generationFailure instanceof RuntimeException runtimeFailure) {
      throw runtimeFailure;
    }
    if (generationFailure instanceof Error error) {
      throw error;
    }
    if (generationFailure != null) {
      throw new IllegalStateException("constrained generation failed", generationFailure);
    }
    return output.toString();
  }
}
