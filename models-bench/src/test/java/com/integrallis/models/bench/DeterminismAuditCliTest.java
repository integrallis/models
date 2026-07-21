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

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.models.api.BackendDiagnostics;
import com.integrallis.models.api.InferenceBackend;
import com.integrallis.models.api.ModelMetadata;
import com.integrallis.models.api.Tokenizer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.modeljars.ModelJarRegistry;

class DeterminismAuditCliTest {

  @TempDir Path directory;

  @Test
  void parsesConservativeAuditDefaults() throws Exception {
    Path model = Files.write(directory.resolve("fixture.gguf"), new byte[] {1, 2, 3});

    DeterminismAuditCli.Configuration configuration =
        DeterminismAuditCli.parse(new String[] {"--model", model.toString()});

    assertThat(configuration.model().artifact()).isEqualTo(model);
    assertThat(configuration.model().descriptor()).isEmpty();
    assertThat(configuration.generatedTokens()).isEqualTo(4);
    assertThat(configuration.iterations()).isEqualTo(3);
    assertThat(configuration.contextLength()).isEqualTo(128);
    assertThat(configuration.promptMode()).isEqualTo(DeterminismAuditCli.PromptMode.SEQUENTIAL);
  }

  @Test
  void resolvesModelJarAliasWithoutDiscardingItsDescriptor() throws Exception {
    Path model = Files.write(directory.resolve("modeljar.gguf"), new byte[] {1, 2, 3});
    var descriptor = ModelJarTestFixtures.descriptor("fixture", model);

    DeterminismAuditCli.Configuration configuration =
        DeterminismAuditCli.parse(
            new String[] {"--modeljar", "fixture"}, ModelJarRegistry.of(List.of(descriptor)));

    assertThat(configuration.model().identity()).isEqualTo("fixture");
    assertThat(configuration.model().artifact()).isEqualTo(model);
    assertThat(configuration.model().descriptor()).contains(descriptor);
    assertThat(configuration.modelId()).isEqualTo("fixture");
  }

  @Test
  void recordsTheSelectedBackendPlanInDeterminismEvidence() throws Exception {
    Path model = Files.write(directory.resolve("fixture.gguf"), new byte[] {1, 2, 3});
    DeterminismAuditCli.Configuration configuration =
        DeterminismAuditCli.parse(new String[] {"--model", model.toString()});
    DeterminismAuditCli.AuditResult audit =
        DeterminismAuditCli.audit(
            new DeterministicBackend(),
            configuration.prompt(),
            configuration.generatedTokens(),
            configuration.iterations(),
            configuration.promptMode());
    BackendDiagnostics diagnostics =
        new BackendDiagnostics("pure-java", "fixture-plan", Map.of(), List.of());

    DeterminismAuditCli.Report report =
        DeterminismAuditCli.buildReport(configuration, 1.25, audit, diagnostics);

    assertThat(report.schemaVersion()).isEqualTo(2);
    assertThat(report.backendDiagnostics()).isEqualTo(diagnostics);
  }

  @Test
  void recordsExactRepeatabilityAndWinnerMargins() {
    DeterministicBackend backend = new DeterministicBackend();

    DeterminismAuditCli.AuditResult result =
        DeterminismAuditCli.audit(
            backend, "audit prompt", 2, 3, DeterminismAuditCli.PromptMode.SEQUENTIAL);

    assertThat(result.deterministic()).isTrue();
    assertThat(result.promptTokens()).containsExactly(1, 2);
    assertThat(result.trials()).hasSize(3);
    assertThat(result.trials().getFirst().steps())
        .extracting(DeterminismAuditCli.Step::winnerToken)
        .containsExactly(1, 2);
    assertThat(result.trials().getFirst().steps().getFirst().margin()).isEqualTo(1.0f);
    assertThat(result.trials())
        .extracting(DeterminismAuditCli.Trial::sequenceSha256)
        .containsOnly(result.trials().getFirst().sequenceSha256());
  }

  @Test
  void detectsLogitDriftEvenWhenTheWinningTokenDoesNotChange() {
    DriftingBackend backend = new DriftingBackend();

    DeterminismAuditCli.AuditResult result =
        DeterminismAuditCli.audit(
            backend, "audit prompt", 1, 2, DeterminismAuditCli.PromptMode.SEQUENTIAL);

    assertThat(result.trials())
        .extracting(trial -> trial.steps().getFirst().winnerToken())
        .containsOnly(1);
    assertThat(result.trials())
        .extracting(trial -> trial.steps().getFirst().logitsSha256())
        .doesNotHaveDuplicates();
    assertThat(result.deterministic()).isFalse();
  }

  @Test
  void hashesRawFloatBitsRatherThanNumericEquality() {
    assertThat(Hashing.sha256(new float[] {0.0f}))
        .isNotEqualTo(Hashing.sha256(new float[] {-0.0f}));
  }

  private static final class DeterministicBackend extends BaseBackend {
    @Override
    public float[] forward(int token, int position) {
      return position < 2 ? new float[] {0.0f, 4.0f, 3.0f} : new float[] {0.0f, 2.0f, 5.0f};
    }
  }

  private static final class DriftingBackend extends BaseBackend {
    private int trial;

    @Override
    public float[] forward(int token, int position) {
      return new float[] {trial * 0.25f, 10.0f, 9.0f};
    }

    @Override
    public void reset() {
      trial++;
    }
  }

  private abstract static class BaseBackend implements InferenceBackend {
    private final Tokenizer tokenizer =
        new Tokenizer() {
          @Override
          public int[] encode(String text) {
            return new int[] {1, 2};
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
          public int eosToken() {
            return -1;
          }

          @Override
          public int bosToken() {
            return -1;
          }

          @Override
          public int vocabSize() {
            return 3;
          }
        };

    @Override
    public String name() {
      return "fake";
    }

    @Override
    public ModelMetadata metadata() {
      return new ModelMetadata("fake", "audit", 128, 3, 1, 1, 1, 1);
    }

    @Override
    public Tokenizer tokenizer() {
      return tokenizer;
    }

    @Override
    public void reset() {}

    @Override
    public void close() {}
  }
}
