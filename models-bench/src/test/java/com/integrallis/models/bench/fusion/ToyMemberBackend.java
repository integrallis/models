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

import com.integrallis.models.api.ModelMetadata;
import com.integrallis.models.api.RewindableInferenceBackend;
import com.integrallis.models.api.Tokenizer;

/** Presents a {@link FusionMember} as the backend a {@code gated} arm composes. */
final class ToyMemberBackend implements RewindableInferenceBackend {
  private final FusionMember member;

  ToyMemberBackend(FusionMember member) {
    this.member = member;
  }

  @Override
  public String name() {
    return member.name();
  }

  @Override
  public ModelMetadata metadata() {
    return new ModelMetadata("toy", member.name(), 256, ToyMember.VOCAB, 8, 1, 1, 1);
  }

  @Override
  public Tokenizer tokenizer() {
    return member.tokenizer();
  }

  @Override
  public float[] prefill(int[] tokens, int startPosition) {
    return member.prefill(tokens, startPosition);
  }

  @Override
  public float[] forward(int token, int position) {
    return member.forward(token, position);
  }

  @Override
  public void reset() {
    member.reset();
  }

  @Override
  public int checkpoint() {
    return member.checkpoint();
  }

  @Override
  public void rewind(int checkpoint) {
    member.rewind(checkpoint);
  }

  @Override
  public void close() {
    member.close();
  }
}
