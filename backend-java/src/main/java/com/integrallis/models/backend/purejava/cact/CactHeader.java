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
package com.integrallis.models.backend.purejava.cact;

import java.util.List;

/** Architecture and runtime geometry serialized in the fixed `.cact` header. */
public record CactHeader(
    int tensorCount,
    int codebookLength,
    int kvWindow,
    int kvBits,
    int vocabularySize,
    int modelWidth,
    int queryHeadCount,
    int kvHeadCount,
    int layerCount,
    int headWidth,
    int maximumSequenceLength,
    int hadamardSize,
    int mhcLanes,
    int engramSlots,
    int engramSubDimension,
    int engramTableCount,
    int engramConvolutionTaps,
    int engramConvolutionDilation,
    List<Integer> engramOrders,
    List<Integer> engramSites,
    float ropeTheta) {

  public CactHeader {
    engramOrders = List.copyOf(engramOrders);
    engramSites = List.copyOf(engramSites);
  }
}
