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
package com.integrallis.models.backend.purejava.structure;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.models.backend.purejava.ops.TensorOps;
import java.net.URISyntaxException;
import java.nio.file.Path;
import java.util.List;
import java.util.Set;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Fails when a Vector API kernel in backend-java takes a {@link VectorSpecies} as a method or
 * constructor parameter, or holds one in a field that is not {@code static final}.
 *
 * <p>C2 constant-folds a species only when it is a static final constant. Another JVM inference
 * engine self-reported (not measured here) that turning the species into a parameter moved an 11 MB
 * allocation profile to 162 GB, because vector boxes stopped being eliminated.
 */
@Tag("unit")
class VectorSpeciesConstantStructuralTest {

  /**
   * Existing violations, each with the reason it is tolerated and the follow-up that removes it.
   * Empty as of 2026-09-16: backend-java's two Vector API classes ({@code TensorOps}, {@code
   * GroupedQueryAttentionKernel}) hold their species in static final fields only.
   */
  private static final Set<String> ALLOWED_VIOLATIONS = Set.of();

  @Test
  void backendJavaVectorKernelsHoldSpeciesOnlyInStaticFinalFields() throws URISyntaxException {
    Path classes =
        Path.of(TensorOps.class.getProtectionDomain().getCodeSource().getLocation().toURI());
    VectorSpeciesConstantRule.Scan scan =
        VectorSpeciesConstantRule.scanClassDirectory(classes, TensorOps.class.getClassLoader());

    // Non-vacuous: the scan must actually have reached the known Vector API classes.
    assertThat(scan.scannedClasses())
        .contains(
            "com.integrallis.models.backend.purejava.ops.TensorOps",
            "com.integrallis.models.backend.purejava.ops.GroupedQueryAttentionKernel");
    assertThat(scan.violations()).containsExactlyInAnyOrderElementsOf(ALLOWED_VIOLATIONS);
  }

  /**
   * Violations in the pinned vectors-core artifact, where most of the Vector API kernels
   * backend-java calls actually live. vectors-core is a separate project, so these are recorded,
   * not fixed here.
   */
  private static final Set<String> ALLOWED_VECTORS_CORE_VIOLATIONS =
      Set.of(
          // Follow-up (vectors project): inline into a static final initializer. Tolerated because
          // `javap -c` on vectors-core 0.1.22 shows all five call sites inside
          // PanamaVectorUtilSupport.<clinit>, so the species reaching kernels is still a static
          // final constant. Not measured: whether C2 treats it any differently.
          "com.integrallis.vectors.core.PanamaConstants#preferredSpecies(VectorSpecies) parameter 0");

  @Test
  void vectorsCoreKernelViolationsMatchTheRecordedAllowList() throws Exception {
    Class<?> vectorUtil = Class.forName("com.integrallis.vectors.core.VectorUtil");
    Path jar = Path.of(vectorUtil.getProtectionDomain().getCodeSource().getLocation().toURI());
    VectorSpeciesConstantRule.Scan scan =
        VectorSpeciesConstantRule.scanClassDirectory(jar, vectorUtil.getClassLoader());

    assertThat(scan.scannedClasses())
        .contains("com.integrallis.vectors.core.PanamaVectorUtilSupport");
    assertThat(scan.violations())
        .containsExactlyInAnyOrderElementsOf(ALLOWED_VECTORS_CORE_VIOLATIONS);
  }

  @Test
  void ruleFlagsSpeciesParameters() {
    assertThat(VectorSpeciesConstantRule.violations(SpeciesParameter.class))
        .containsExactlyInAnyOrder(
            SpeciesParameter.class.getName() + "#sum(VectorSpecies, float[]) parameter 0",
            SpeciesParameter.class.getName() + "#<init>(VectorSpecies) parameter 0");
  }

  @Test
  void ruleFlagsSpeciesFieldsThatAreNotStaticFinal() {
    assertThat(VectorSpeciesConstantRule.violations(SpeciesFields.class))
        .containsExactlyInAnyOrder(
            SpeciesFields.class.getName() + ".instanceFinal is not static final",
            SpeciesFields.class.getName() + ".staticMutable is not static final");
  }

  @Test
  void ruleAcceptsAStaticFinalSpeciesConstant() {
    assertThat(VectorSpeciesConstantRule.violations(SpeciesConstant.class)).isEmpty();
  }

  @Test
  void ruleFlagsSpeciesCapturedByALambdaParameter() {
    List<String> violations = VectorSpeciesConstantRule.violations(SpeciesLambda.class);

    assertThat(violations).anySatisfy(v -> assertThat(v).contains("lambda$").contains("parameter"));
  }

  static final class SpeciesParameter {
    SpeciesParameter(VectorSpecies<Float> species) {}

    static float sum(VectorSpecies<Float> species, float[] values) {
      return FloatVector.fromArray(species, values, 0)
          .reduceLanes(jdk.incubator.vector.VectorOperators.ADD);
    }
  }

  static final class SpeciesFields {
    private static final VectorSpecies<Float> CONSTANT = FloatVector.SPECIES_PREFERRED;
    private final VectorSpecies<Float> instanceFinal = FloatVector.SPECIES_PREFERRED;

    @SuppressWarnings("unused")
    private static VectorSpecies<Float> staticMutable = FloatVector.SPECIES_PREFERRED;
  }

  static final class SpeciesConstant {
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    static int lanes() {
      return SPECIES.length();
    }
  }

  static final class SpeciesLambda {
    static int lanes() {
      VectorSpecies<Float> local = FloatVector.SPECIES_PREFERRED;
      java.util.function.IntSupplier supplier = () -> local.length();
      return supplier.getAsInt();
    }
  }
}
