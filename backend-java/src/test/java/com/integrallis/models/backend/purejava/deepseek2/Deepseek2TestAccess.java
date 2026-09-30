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
package com.integrallis.models.backend.purejava.deepseek2;

/**
 * Hands the toy DeepSeek-V2 graph to tests in the backend package.
 *
 * <p>The weights loader stays package-private -- nothing outside should build one -- but the
 * decoder adapter lives one package up and needs a graph to adapt.
 */
public final class Deepseek2TestAccess {

  private Deepseek2TestAccess() {}

  /** The toy model as a ready graph, with room for sixteen positions. */
  public static Deepseek2ForwardPass toyForwardPass() {
    Deepseek2ToyModel model = Deepseek2ToyModel.create();
    Deepseek2Config config = Deepseek2ToyModel.config();
    return new Deepseek2ForwardPass(
        config, Deepseek2Weights.fromGgufFile(model.file(), config), 16);
  }
}
