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
package com.integrallis.models.backend.nativekernel;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatNullPointerException;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import com.integrallis.models.api.BackendConfiguration;
import com.integrallis.models.api.BackendDiagnostics;
import com.integrallis.models.api.LogitBatch;
import com.integrallis.models.api.ModelMetadata;
import com.integrallis.models.api.OptimizationStatus;
import com.integrallis.models.api.Tokenizer;
import com.integrallis.models.backend.purejava.PureJavaBackend;
import com.integrallis.models.backend.purejava.plan.PureJavaExecutionPlan;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class RustFfmBackendTest {

  @Test
  void versionsAutomaticNativeProfileSelection() {
    assertThat(RustFfmBackend.PLAN_VERSION).isEqualTo("rust-ffm-v10");
  }

  @Test
  void resolvesNativeKernelSettingsFromModelProfileRecommendations() {
    NativeKernelSettings settings =
        NativeKernelSettings.resolve(
            Map.of(
                RustGgufBatchedMatrixKernel.NATIVE_DECODE_PROPERTY,
                "true",
                RustGgufBatchedMatrixKernel.Q5_0_GROUPED_PROPERTY,
                "true",
                NativeKernelLibrary.THREAD_COUNT_PROPERTY,
                "4",
                "models.purejava.prefillBatchSize",
                "64"),
            Map.of(),
            8);

    assertThat(settings.nativeDecode()).isTrue();
    assertThat(settings.q5_0Grouped()).isTrue();
    assertThat(settings.threadCount()).isEqualTo(4);
  }

  @Test
  void deploymentPropertiesOverrideModelProfileRecommendations() {
    NativeKernelSettings settings =
        NativeKernelSettings.resolve(
            Map.of(
                RustGgufBatchedMatrixKernel.NATIVE_DECODE_PROPERTY,
                "true",
                RustGgufBatchedMatrixKernel.Q5_0_GROUPED_PROPERTY,
                "true",
                NativeKernelLibrary.THREAD_COUNT_PROPERTY,
                "4"),
            Map.of(
                RustGgufBatchedMatrixKernel.NATIVE_DECODE_PROPERTY,
                "false",
                RustGgufBatchedMatrixKernel.Q5_0_GROUPED_PROPERTY,
                "false",
                NativeKernelLibrary.THREAD_COUNT_PROPERTY,
                "8"),
            16);

    assertThat(settings.nativeDecode()).isFalse();
    assertThat(settings.q5_0Grouped()).isFalse();
    assertThat(settings.threadCount()).isEqualTo(8);
  }

  @Test
  void rejectsMissingLoadInputsBeforeOpeningTheNativeLibrary() {
    Path modelPath = Path.of("model.gguf");
    Path libraryPath = Path.of("libjmodels_kernels.so");
    BackendConfiguration configuration = BackendConfiguration.empty();

    assertThatNullPointerException()
        .isThrownBy(() -> RustFfmBackend.load(null, libraryPath, configuration))
        .withMessage("modelPath");
    assertThatNullPointerException()
        .isThrownBy(() -> RustFfmBackend.load(modelPath, null, configuration))
        .withMessage("libraryPath");
    assertThatNullPointerException()
        .isThrownBy(() -> RustFfmBackend.load(modelPath, libraryPath, null))
        .withMessage("backendConfiguration");
  }

  @Test
  void delegatesTheCompleteInferenceContractToTheJavaTransformer() {
    PureJavaBackend delegate = mock(PureJavaBackend.class);
    BackendDiagnostics diagnostics =
        new BackendDiagnostics("rust-ffm", RustFfmBackend.PLAN_VERSION, Map.of(), List.of());
    ModelMetadata metadata = mock(ModelMetadata.class);
    PureJavaExecutionPlan executionPlan = mock(PureJavaExecutionPlan.class);
    Tokenizer tokenizer = mock(Tokenizer.class);
    float[] forward = {1.0f};
    float[] transientForward = {2.0f};
    float[] prefill = {3.0f};
    LogitBatch verification = mock(LogitBatch.class);
    LogitBatch transientVerification = mock(LogitBatch.class);
    when(delegate.metadata()).thenReturn(metadata);
    when(delegate.executionPlan()).thenReturn(executionPlan);
    when(delegate.tokenizer()).thenReturn(tokenizer);
    when(delegate.forward(7, 3)).thenReturn(forward);
    when(delegate.forwardTransient(11, 4)).thenReturn(transientForward);
    when(delegate.prefill(new int[] {1, 2}, 5)).thenReturn(prefill);
    when(delegate.checkpoint()).thenReturn(6);
    when(delegate.verify(new int[] {13}, 6)).thenReturn(verification);
    when(delegate.verifyTransient(new int[] {17}, 7)).thenReturn(transientVerification);

    RustFfmBackend backend = new RustFfmBackend(delegate, diagnostics);

    assertThat(backend.name()).isEqualTo("rust-ffm");
    assertThat(backend.metadata()).isSameAs(metadata);
    assertThat(backend.executionPlan()).isSameAs(executionPlan);
    assertThat(backend.diagnostics()).isSameAs(diagnostics);
    assertThat(backend.tokenizer()).isSameAs(tokenizer);
    assertThat(backend.forward(7, 3)).isSameAs(forward);
    assertThat(backend.forwardTransient(11, 4)).isSameAs(transientForward);
    assertThat(backend.prefill(new int[] {1, 2}, 5)).isSameAs(prefill);
    assertThat(backend.checkpoint()).isEqualTo(6);
    assertThat(backend.verify(new int[] {13}, 6)).isSameAs(verification);
    assertThat(backend.verifyTransient(new int[] {17}, 7)).isSameAs(transientVerification);

    backend.rewind(8);
    backend.reset();
    backend.close();

    verify(delegate).rewind(8);
    verify(delegate).reset();
    verify(delegate).close();
  }

  @Test
  void enrichesJavaDiagnosticsWithEnabledNativeKernelControls() {
    RustGgufBatchedMatrixKernel kernel = mock(RustGgufBatchedMatrixKernel.class);
    when(kernel.implementation()).thenReturn("rust-test");
    when(kernel.threadCount()).thenReturn(4);
    when(kernel.nativeDecodeEnabled()).thenReturn(true);
    when(kernel.q5_0GroupedEnabled()).thenReturn(true);
    BackendDiagnostics javaDiagnostics =
        new BackendDiagnostics("pure-java", "java-plan", Map.of("model", "nano"), List.of());

    BackendDiagnostics diagnostics = RustFfmBackend.diagnostics(javaDiagnostics, kernel);

    assertThat(diagnostics.backend()).isEqualTo("rust-ffm");
    assertThat(diagnostics.planVersion()).isEqualTo(RustFfmBackend.PLAN_VERSION);
    assertThat(diagnostics.environment())
        .containsEntry("model", "nano")
        .containsEntry("transformer-runtime", "java")
        .containsEntry("kernel-runtime", "rust-ffm")
        .containsEntry("kernel-implementation", "rust-test")
        .containsEntry("native-kernel-threads", "4")
        .containsEntry("native-quantized-decode", "true")
        .containsEntry("native-q5-0-grouped", "true");
    assertThat(diagnostics.optimization("rust-q5-0-grouped-matmul"))
        .hasValueSatisfying(
            decision -> assertThat(decision.status()).isEqualTo(OptimizationStatus.ENABLED));
    assertThat(diagnostics.optimization("rust-quantized-decode"))
        .hasValueSatisfying(
            decision -> assertThat(decision.status()).isEqualTo(OptimizationStatus.ENABLED));
    assertThat(diagnostics.optimization("rust-q6-k-batched-matmul"))
        .hasValueSatisfying(
            decision -> assertThat(decision.status()).isEqualTo(OptimizationStatus.ENABLED));
  }

  @Test
  void reportsDisabledOptionalNativeKernelControls() {
    RustGgufBatchedMatrixKernel kernel = mock(RustGgufBatchedMatrixKernel.class);
    when(kernel.implementation()).thenReturn("rust-test");
    when(kernel.threadCount()).thenReturn(2);

    BackendDiagnostics diagnostics =
        RustFfmBackend.diagnostics(
            new BackendDiagnostics("pure-java", "java-plan", Map.of(), List.of()), kernel);

    assertThat(diagnostics.optimization("rust-q5-0-grouped-matmul"))
        .hasValueSatisfying(
            decision -> {
              assertThat(decision.status()).isEqualTo(OptimizationStatus.DISABLED);
              assertThat(decision.reason()).contains("independent Q5_0 projections");
            });
    assertThat(diagnostics.optimization("rust-quantized-decode"))
        .hasValueSatisfying(
            decision -> {
              assertThat(decision.status()).isEqualTo(OptimizationStatus.DISABLED);
              assertThat(decision.reason())
                  .contains(RustGgufBatchedMatrixKernel.NATIVE_DECODE_PROPERTY);
            });
  }
}
