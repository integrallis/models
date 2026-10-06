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

import java.util.List;
import java.util.Map;

/**
 * One logged fused decode step.
 *
 * @param step zero-based step
 * @param token fused token
 * @param memberArgmax each member's argmax token
 * @param matchesMembers members whose argmax equals the fused token
 * @param fusedEntropy entropy of softmax(s) in nats
 * @param klFusedToMember KL(fused || member) in nats
 * @param fusedTokenLogProbability log p_fused(token) at T = 1
 */
public record TokenLogEntry(
    int step,
    int token,
    Map<String, Integer> memberArgmax,
    List<String> matchesMembers,
    double fusedEntropy,
    Map<String, Double> klFusedToMember,
    double fusedTokenLogProbability) {

  public TokenLogEntry {
    memberArgmax = Map.copyOf(memberArgmax);
    matchesMembers = List.copyOf(matchesMembers);
    klFusedToMember = Map.copyOf(klFusedToMember);
  }
}
