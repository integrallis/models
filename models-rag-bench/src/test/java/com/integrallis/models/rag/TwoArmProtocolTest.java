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
package com.integrallis.models.rag;

import static org.assertj.core.api.Assertions.assertThat;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.junit.jupiter.api.Test;

/**
 * The two two-arm runners must agree on the protocol, and both must make their comparator
 * admissible.
 *
 * <p>The fleet worker bootstraps itself from S3 as a single file and so cannot source a shared
 * protocol, which is why this is checked rather than factored out. It is checked because the ad-hoc
 * pod runner drifted on both axes at once and nothing noticed: it omitted {@code --artifact} from
 * the comparator arm, which makes {@code artifactSha256} null and the comparison excludable as
 * "artifact SHA-256 differs", and it ran {@code --warmups 0 --iterations 1} where every comparator
 * arm in the certified catalog ran 1 and 3. Both faults are invisible in the reports, which is how
 * three runs' ratios were reported on 2026-10-31 without a gate having admitted any of them.
 */
class TwoArmProtocolTest {

  private static final Pattern WARMUPS = Pattern.compile("--warmups (\\d+)");
  private static final Pattern ITERATIONS = Pattern.compile("--iterations (\\d+)");

  @Test
  void theSharedProtocolIsTheOneTheCertifiedCatalogWasMeasuredUnder() throws IOException {
    String protocol = Files.readString(repositoryFile("scripts/fleet/two-arm-protocol.sh"));

    assertThat(protocol).contains("TWO_ARM_WARMUPS=1").contains("TWO_ARM_ITERATIONS=3");
  }

  @Test
  void theFleetWorkerAndThePodRunnerAgreeOnWarmupsAndIterations() throws IOException {
    String worker = Files.readString(repositoryFile("scripts/fleet/qual-worker-two-arm.sh"));
    String protocol = Files.readString(repositoryFile("scripts/fleet/two-arm-protocol.sh"));

    assertThat(firstGroup(WARMUPS, worker)).isEqualTo(valueOf(protocol, "TWO_ARM_WARMUPS"));
    assertThat(firstGroup(ITERATIONS, worker)).isEqualTo(valueOf(protocol, "TWO_ARM_ITERATIONS"));
  }

  @Test
  void thePodRunnerTakesItsProtocolFromTheSharedFileRatherThanRepeatingIt() throws IOException {
    String runner = Files.readString(repositoryFile("scripts/fleet/two-arm-pod.sh"));

    assertThat(runner)
        .contains(". \"$SCRIPT_DIR/two-arm-protocol.sh\"")
        .contains("two_arm_common_args");
    assertThat(WARMUPS.matcher(runner).find())
        .as("the pod runner must not hard-code --warmups; it comes from two-arm-protocol.sh")
        .isFalse();
  }

  @Test
  void everyComparatorArmPassesTheArtifactThatMakesItAdmissible() throws IOException {
    for (String script :
        new String[] {"scripts/fleet/two-arm-pod.sh", "scripts/fleet/qual-worker-two-arm.sh"}) {
      String body = Files.readString(repositoryFile(script));
      for (String engine : new String[] {"--backend llama.cpp", "--backend ollama"}) {
        int cursor = 0;
        while (true) {
          int arm = body.indexOf(engine, cursor);
          if (arm < 0) {
            break;
          }
          // The invocation ends at the first --output, which every arm writes. The pod runner
          // builds its arm into COMPARATOR_ARGS and passes --artifact beside it, so the window
          // searched runs from the engine to the end of the invocation that consumes it.
          int end = body.indexOf("--output", arm);
          assertThat(end).as("comparator arm in %s has no --output", script).isPositive();
          assertThat(body.substring(arm, end))
              .as("the %s comparator arm in %s must pass --artifact", engine, script)
              .contains("--artifact");
          cursor = end;
        }
      }
    }
  }

  /**
   * The pod runner defaults to the only comparator that can reach QUALIFIED.
   *
   * <p>{@code RagProductionQualificationPolicy.AUTHORITATIVE_ENGINES} is {@code {ollama,
   * transformers}}. A llama.cpp arm is admitted at 0.45/2.0 but is never the backend that qualifies
   * a model, so a runner defaulting to it produces a comparison that is real, admissible, and still
   * cannot return better than FAILED_RELATIVE_GATE -- a whole run that decides nothing. The
   * publisher-reference route that does let llama.cpp qualify a model is reviewed per artifact and
   * exists for publishers who ship their own llama.cpp, which is not the case for any of the
   * architectures this runner was written for.
   */
  @Test
  void thePodRunnerDefaultsToAnAuthoritativeComparator() throws IOException {
    String runner = Files.readString(repositoryFile("scripts/fleet/two-arm-pod.sh"));

    assertThat(runner).contains("COMPARATOR=${COMPARATOR:-ollama}");
  }

  private static int valueOf(String protocol, String name) {
    Matcher matcher = Pattern.compile(name + "=(\\d+)").matcher(protocol);
    assertThat(matcher.find()).as("%s is set in two-arm-protocol.sh", name).isTrue();
    return Integer.parseInt(matcher.group(1));
  }

  private static int firstGroup(Pattern pattern, String body) {
    Matcher matcher = pattern.matcher(body);
    assertThat(matcher.find()).as("pattern %s occurs", pattern).isTrue();
    return Integer.parseInt(matcher.group(1));
  }

  private static Path repositoryFile(String relative) {
    return Path.of(System.getProperty("models.repositoryRoot", "."))
        .toAbsolutePath()
        .normalize()
        .resolve(relative);
  }
}
