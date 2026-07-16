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

import com.integrallis.models.api.LogitBatch;
import com.integrallis.models.api.SpeculativeInferenceBackend;
import com.integrallis.models.backend.purejava.PureJavaBackend;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.TimeUnit;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.TearDown;
import org.openjdk.jmh.annotations.Warmup;

/** Compares sequential target evaluation with one causal speculative-verification batch. */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Thread)
@Fork(
    value = 1,
    jvmArgsPrepend = {"--add-modules", "jdk.incubator.vector"})
@Warmup(iterations = 2, time = 1)
@Measurement(iterations = 5, time = 1)
public class SpeculativeVerificationBenchmark {

  private static final String MODEL_PROPERTY = "models.bench.model";
  private static final String PROMPT =
      "Write a Java method that parses an integer and returns a useful validation error.";

  @Param({"2", "4", "8"})
  int draftLength;

  private SpeculativeInferenceBackend backend;
  private int[] promptTokens;
  private int[] proposedTokens;
  private int checkpoint;

  @Setup(Level.Trial)
  public void setUpTrial() {
    String configuredModel = System.getProperty(MODEL_PROPERTY);
    if (configuredModel == null || configuredModel.isBlank()) {
      throw new IllegalArgumentException("-D" + MODEL_PROPERTY + " must name a GGUF model");
    }
    Path model = Path.of(configuredModel).toAbsolutePath();
    if (!Files.isRegularFile(model)) {
      throw new IllegalArgumentException("model does not exist: " + model);
    }

    System.setProperty("models.purejava.maxContextLength", "256");
    PureJavaBackend pureJava = PureJavaBackend.load(model);
    backend = pureJava;
    promptTokens = pureJava.tokenizer().encode(PROMPT);
    proposedTokens = generateGreedyProposal(draftLength);
    assertExactVerification();
  }

  @Setup(Level.Iteration)
  public void setUpIteration() {
    backend.reset();
    backend.prefill(promptTokens, 0);
    checkpoint = backend.checkpoint();
  }

  @Benchmark
  public int sequential() {
    int checksum = 1;
    for (int index = 0; index < proposedTokens.length; index++) {
      float[] logits = backend.forwardTransient(proposedTokens[index], checkpoint + index);
      checksum = 31 * checksum + argmax(logits);
    }
    backend.rewind(checkpoint);
    return checksum;
  }

  @Benchmark
  public int batched() {
    LogitBatch logits = backend.verifyTransient(proposedTokens, checkpoint);
    int checksum = 1;
    for (int index = 0; index < proposedTokens.length; index++) {
      checksum = 31 * checksum + logits.argmax(index);
    }
    backend.rewind(checkpoint);
    return checksum;
  }

  @TearDown(Level.Trial)
  public void tearDown() {
    backend.close();
  }

  private int[] generateGreedyProposal(int length) {
    backend.reset();
    float[] logits = backend.prefill(promptTokens, 0);
    int[] proposal = new int[length];
    for (int index = 0; index < length; index++) {
      int token = argmax(logits);
      proposal[index] = token;
      logits = backend.forwardTransient(token, promptTokens.length + index);
    }
    return proposal;
  }

  private void assertExactVerification() {
    backend.reset();
    backend.prefill(promptTokens, 0);
    float[][] sequential = new float[proposedTokens.length][];
    for (int index = 0; index < proposedTokens.length; index++) {
      sequential[index] = backend.forward(proposedTokens[index], promptTokens.length + index);
    }

    backend.reset();
    backend.prefill(promptTokens, 0);
    LogitBatch batched = backend.verifyTransient(proposedTokens, promptTokens.length);
    for (int row = 0; row < proposedTokens.length; row++) {
      float[] actual = batched.copyRow(row);
      float[] expected = sequential[row];
      for (int column = 0; column < expected.length; column++) {
        if (Float.floatToRawIntBits(actual[column]) != Float.floatToRawIntBits(expected[column])) {
          throw new IllegalStateException(
              "verification mismatch at row " + row + ", vocabulary index " + column);
        }
      }
    }
  }

  private static int argmax(float[] logits) {
    int best = 0;
    for (int index = 1; index < logits.length; index++) {
      if (logits[index] > logits[best]) {
        best = index;
      }
    }
    return best;
  }
}
