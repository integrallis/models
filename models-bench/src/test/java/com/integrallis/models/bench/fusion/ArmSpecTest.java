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

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

import java.util.List;
import org.junit.jupiter.api.Test;

class ArmSpecTest {

  @Test
  void parsesSingleMember() {
    ArmSpec arm = ArmSpec.parse("member:B");
    assertThat(arm.kind()).isEqualTo(ArmSpec.Kind.MEMBER);
    assertThat(arm.members()).containsExactly("B");
    assertThat(arm.requiresFrozen()).isFalse();
  }

  @Test
  void parsesFusionWithUniformExplicitAndTunedWeights() {
    ArmSpec uniform = ArmSpec.parse("fuse:A+B+C:poe:uniform");
    assertThat(uniform.kind()).isEqualTo(ArmSpec.Kind.FUSE);
    assertThat(uniform.rule()).isEqualTo(FusionRule.POE);
    assertThat(uniform.weightMode()).isEqualTo(ArmSpec.WeightMode.UNIFORM);
    assertThat(uniform.resolveWeights(null)[1]).isCloseTo(1.0 / 3, within(1e-12));

    ArmSpec article = ArmSpec.parse("fuse:A+B+C:article:0.5,0.2,0.3");
    assertThat(article.weightMode()).isEqualTo(ArmSpec.WeightMode.EXPLICIT);
    assertThat(article.resolveWeights(null)).containsExactly(0.5, 0.2, 0.3);

    ArmSpec tuned = ArmSpec.parse("fuse:A+C:mixture:tuned");
    assertThat(tuned.requiresFrozen()).isTrue();
    assertThatThrownBy(() -> tuned.resolveWeights(null))
        .isInstanceOf(IllegalStateException.class)
        .hasMessageContaining("frozen");
  }

  @Test
  void parsesVotingFamilySelfConsistencyAndRerank() {
    ArmSpec vote = ArmSpec.parse("vote:A+B+C:B");
    assertThat(vote.kind()).isEqualTo(ArmSpec.Kind.VOTE);
    assertThat(vote.tieBreakMember()).isEqualTo("B");
    assertThat(ArmSpec.parse("vote-consist:A+B+C:B").kind()).isEqualTo(ArmSpec.Kind.VOTE_CONSIST);
    ArmSpec best = ArmSpec.parse("vote:A+B+C:best");
    assertThat(best.requiresFrozen()).isTrue();
    assertThat(best.priority("C")).containsExactly("C", "A", "B");
    assertThatThrownBy(() -> best.resolveTieBreak(null)).hasMessageContaining("frozen");
    ArmSpec conf = ArmSpec.parse("vote-conf:A+B+C:B");
    assertThat(conf.requiresFrozen()).isTrue();

    ArmSpec sc = ArmSpec.parse("sc:C:5");
    assertThat(sc.kind()).isEqualTo(ArmSpec.Kind.SELF_CONSISTENCY);
    assertThat(sc.samples()).isEqualTo(5);
    assertThat(sc.members()).containsExactly("C");

    ArmSpec rerank = ArmSpec.parse("rerank:A+B+C:tuned");
    assertThat(rerank.kind()).isEqualTo(ArmSpec.Kind.RERANK);
    assertThat(rerank.requiresFrozen()).isTrue();
  }

  @Test
  void rejectsMalformedSpecs() {
    for (String bad :
        List.of(
            "",
            "member:",
            "fuse:A+B:poe",
            "fuse:A+B:sum:uniform",
            "fuse:A+B:poe:0.5,0.6",
            "fuse:A+B:poe:1.0",
            "fuse:A+A:poe:uniform",
            "vote:A+B+C:D",
            "sc:C:0",
            "sc:C:x",
            "oracle")) {
      assertThatThrownBy(() -> ArmSpec.parse(bad))
          .as(bad)
          .isInstanceOf(IllegalArgumentException.class);
    }
  }

  @Test
  void canonicalFormRoundTrips() {
    ArmSpec tuned = ArmSpec.parse("fuse:A+B+C:poe:tuned");
    assertThat(tuned.canonical()).isEqualTo("fuse:A+B+C:poe:tuned");
    assertThat(ArmSpec.parse("fuse:A+B:article:0.5,0.5").canonical())
        .isEqualTo("fuse:A+B:article:0.5,0.5");
  }
}
