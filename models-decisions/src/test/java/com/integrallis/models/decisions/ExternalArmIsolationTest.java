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
package com.integrallis.models.decisions;

import static org.assertj.core.api.Assertions.assertThat;

import java.io.IOException;
import java.lang.classfile.ClassFile;
import java.lang.classfile.ClassModel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.stream.Stream;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * The training path must stay demonstrably free of any third-party service's output.
 *
 * <p>TypeSafe's Master Customer Agreement section 2.3(b) forbids using the Services or any Output
 * to "perform model distillation, train a model to imitate the output of the Services, or develop
 * ... a similar or competing product". Our head is trained on hidden states from our own Apache-2.0
 * base and on labels from published corpora, and it must remain provably so.
 *
 * <p>A comment saying that is worth nothing. This scans the compiled bytecode of the classes that
 * fit a head or a temperature and fails if any of them so much as mentions the external package. A
 * future contributor cannot cross the boundary by accident, and the provenance of every trained
 * weight stays demonstrable rather than asserted.
 */
@Tag("unit")
class ExternalArmIsolationTest {

  /** Classes that produce or shape a trained artefact. None may touch third-party output. */
  private static final Set<String> TRAINING_PATH =
      Set.of(
          "LogisticHeadTrainer.class",
          "TemperatureFitter.class",
          "LinearDecisionHead.class",
          "DecisionHead.class",
          "Calibration.class");

  private static final String EXTERNAL_PACKAGE = "com/integrallis/models/decisions/external";

  @Test
  void noClassOnTheTrainingPathReferencesThirdPartyVerdicts() throws IOException {
    Path classes = Path.of("build/classes/java/main/com/integrallis/models/decisions");
    org.junit.jupiter.api.Assumptions.assumeTrue(
        Files.isDirectory(classes), "compiled classes not present");

    List<String> offenders = new ArrayList<>();
    for (String name : TRAINING_PATH) {
      Path file = classes.resolve(name);
      if (!Files.isRegularFile(file)) {
        continue;
      }
      ClassModel model = ClassFile.of().parse(Files.readAllBytes(file));
      for (var entry : model.constantPool()) {
        String rendered =
            switch (entry) {
              case java.lang.classfile.constantpool.ClassEntry c -> c.asInternalName();
              case java.lang.classfile.constantpool.Utf8Entry u -> u.stringValue();
              default -> "";
            };
        if (rendered.contains(EXTERNAL_PACKAGE)) {
          offenders.add(name + " references " + rendered);
        }
      }
    }

    assertThat(offenders)
        .describedAs(
            "A class that fits a head or a temperature referenced third-party verdicts. That is the"
                + " boundary MCA 2.3(b) draws, and it must stay structural rather than advisory.")
        .isEmpty();
  }

  @Test
  void theExternalVerdictTypeCannotCarryATrainingSignal() {
    // It holds an id, a probability and a service name. No hidden state, no logits, no feature
    // vector: nothing a head could be fitted on even if someone tried.
    var fields =
        Stream.of(
                com.integrallis.models.decisions.external.ExternalVerdict.class.getDeclaredFields())
            .filter(f -> !f.isSynthetic())
            .map(java.lang.reflect.Field::getType)
            .toList();

    assertThat(fields).containsExactlyInAnyOrder(String.class, double.class, String.class);
    assertThat(fields).doesNotContain(float[].class, double[].class, int[].class);
  }

  @Test
  void theTrainerAcceptsOnlyCorpusLabelsAndBaseHiddenStates() throws NoSuchMethodException {
    var fit =
        LogisticHeadTrainer.class.getMethod("fit", AnswerSpace.class, float[][].class, int[].class);

    // float[][] comes from our own base's hidden states; int[] comes from CorpusItem.answerable().
    // There is no overload taking probabilities, so a third-party verdict cannot be passed as a
    // soft label, which is the shape model distillation would take.
    assertThat(fit.getParameterTypes())
        .containsExactly(AnswerSpace.class, float[][].class, int[].class);
    assertThat(LogisticHeadTrainer.class.getMethods())
        .filteredOn(m -> m.getName().equals("fit"))
        .hasSize(1);
  }
}
