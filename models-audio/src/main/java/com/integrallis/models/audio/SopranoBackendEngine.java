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
package com.integrallis.models.audio;

import com.integrallis.models.api.BackendDiagnostics;
import com.integrallis.models.backend.purejava.soprano.SopranoBackend;
import com.integrallis.models.backend.purejava.spi.GgufBatchedMatrixKernel;
import java.io.IOException;
import java.nio.file.Path;

final class SopranoBackendEngine implements SopranoEngine {

  private final SopranoBackend backend;
  private final String modelName;

  private SopranoBackendEngine(SopranoBackend backend, String modelName) {
    this.backend = backend;
    this.modelName = modelName;
  }

  static SopranoBackendEngine load(Path path) throws IOException {
    return new SopranoBackendEngine(SopranoBackend.load(path), modelName(path));
  }

  static SopranoBackendEngine load(Path path, GgufBatchedMatrixKernel matrixKernel)
      throws IOException {
    return new SopranoBackendEngine(SopranoBackend.load(path, matrixKernel), modelName(path));
  }

  @Override
  public int[] encodePrompt(String text) {
    return backend.encodePrompt(text);
  }

  @Override
  public Step begin(int[] prompt) {
    return step(backend.begin(prompt));
  }

  @Override
  public Step advance(int token) {
    return step(backend.advance(token));
  }

  @Override
  public float[] decode(float[] features, int frames) {
    return backend.decode(features, frames);
  }

  @Override
  public String modelName() {
    return modelName;
  }

  @Override
  public BackendDiagnostics diagnostics() {
    return backend.diagnostics();
  }

  @Override
  public int eosToken() {
    return backend.eosToken();
  }

  @Override
  public int vocabularySize() {
    return backend.vocabularySize();
  }

  @Override
  public int hiddenSize() {
    return backend.hiddenSize();
  }

  @Override
  public int contextLength() {
    return backend.contextLength();
  }

  @Override
  public int checkpoint() {
    return backend.checkpoint();
  }

  @Override
  public int sampleRate() {
    return backend.sampleRate();
  }

  @Override
  public int samplesPerToken() {
    return backend.samplesPerToken();
  }

  @Override
  public void close() {
    backend.close();
  }

  private static Step step(SopranoBackend.Step output) {
    return new Step(output.logits(), output.hiddenState());
  }

  private static String modelName(Path path) {
    Path fileName = path.getFileName();
    return fileName == null ? path.toString() : fileName.toString();
  }
}
