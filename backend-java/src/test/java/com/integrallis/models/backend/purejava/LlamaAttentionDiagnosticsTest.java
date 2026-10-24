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
package com.integrallis.models.backend.purejava;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.models.api.BackendDiagnostics;
import com.integrallis.models.api.OptimizationStatus;
import com.integrallis.models.backend.purejava.fixture.MapleTestHeaders;
import com.integrallis.models.backend.purejava.llama.LlamaConfig;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Bounded attention and the expert clamp are reported, not inferred.
 *
 * <p>Both failed silently before 2026-10-24 -- maple ran full attention on all 24 layers and never
 * clamped, with no exception and no non-finite logit. The point of these assertions is the
 * UNSUPPORTED case: a checkpoint that declares a window the graph does not apply must say so in the
 * record, because that is the one state no output inspection can distinguish from a model that is
 * merely a bit worse.
 */
@Tag("unit")
class LlamaAttentionDiagnosticsTest {

  private static BackendDiagnostics diagnose(LlamaConfig config) {
    return PureJavaBackend.llamaAttentionDiagnostics(
        BackendDiagnostics.unavailable("pure-java"), config);
  }

  @Test
  void reportsBoundedAttentionOnTheLayersThePatternMarksLocal() {
    BackendDiagnostics diagnostics =
        diagnose(LlamaConfig.fromMetadata(MapleTestHeaders.metadata(true, true)));

    assertThat(diagnostics.optimization("sliding-window-attention"))
        .get()
        .satisfies(
            decision -> {
              assertThat(decision.status()).isEqualTo(OptimizationStatus.ENABLED);
              assertThat(decision.settings())
                  .containsEntry("declared-window", "512")
                  .containsEntry(
                      "local-layers",
                      MapleTestHeaders.localLayers() + "/" + MapleTestHeaders.LAYERS)
                  .containsEntry("pattern-source", "declared-per-layer");
            });
  }

  @Test
  void reportsTheClampedLayerCount() {
    BackendDiagnostics diagnostics =
        diagnose(LlamaConfig.fromMetadata(MapleTestHeaders.metadata(true, true)));

    assertThat(diagnostics.optimization("expert-swiglu-clamp"))
        .get()
        .satisfies(
            decision -> {
              assertThat(decision.status()).isEqualTo(OptimizationStatus.ENABLED);
              assertThat(decision.settings())
                  .containsEntry(
                      "clamped-layers", MapleTestHeaders.LAYERS + "/" + MapleTestHeaders.LAYERS);
            });
  }

  @Test
  void callsADeclaredWindowThatNoLayerAppliesUnsupported() {
    // The exact pre-fix state: the window is declared, the graph applies it nowhere. This must
    // read as UNSUPPORTED rather than DISABLED -- DISABLED would say the checkpoint asked for
    // nothing, which is the misreading that let maple ship with full attention.
    BackendDiagnostics diagnostics =
        diagnose(LlamaConfig.fromMetadata(MapleTestHeaders.metadata(false, true)));

    assertThat(diagnostics.optimization("sliding-window-attention"))
        .get()
        .satisfies(
            decision -> {
              assertThat(decision.status()).isEqualTo(OptimizationStatus.UNSUPPORTED);
              assertThat(decision.reason()).contains("does not apply");
              assertThat(decision.settings())
                  .containsEntry("declared-window", "512")
                  .containsEntry("local-layers", "0/" + MapleTestHeaders.LAYERS)
                  .containsEntry("pattern-source", "architecture-stride");
            });
  }

  @Test
  void reportsNoClampWhereNoneIsDeclared() {
    BackendDiagnostics diagnostics =
        diagnose(LlamaConfig.fromMetadata(MapleTestHeaders.metadata(true, false)));

    assertThat(diagnostics.optimization("expert-swiglu-clamp"))
        .get()
        .satisfies(
            decision -> {
              assertThat(decision.status()).isEqualTo(OptimizationStatus.DISABLED);
              assertThat(decision.settings())
                  .containsEntry("clamped-layers", "0/" + MapleTestHeaders.LAYERS);
            });
  }
}
