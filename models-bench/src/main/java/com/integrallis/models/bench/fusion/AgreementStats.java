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

import java.util.Map;

/**
 * Gate G3 counters over every decode step of one fused output.
 *
 * @param steps decode steps taken
 * @param allMembersAgree steps where every member's argmax was the same token
 * @param fusedEqualsMember per member, steps where the fused token equals that member's argmax
 * @param fusedEqualsNoMember steps where the fused token equals no member's argmax
 * @param sumFusedEntropy sum over steps of the fused distribution's entropy (nats, T = 1)
 * @param sumKlFusedToMember per member, sum over steps of KL(fused || member)
 */
public record AgreementStats(
    int steps,
    int allMembersAgree,
    Map<String, Integer> fusedEqualsMember,
    int fusedEqualsNoMember,
    double sumFusedEntropy,
    Map<String, Double> sumKlFusedToMember) {

  public AgreementStats {
    fusedEqualsMember = Map.copyOf(fusedEqualsMember);
    sumKlFusedToMember = Map.copyOf(sumKlFusedToMember);
  }
}
