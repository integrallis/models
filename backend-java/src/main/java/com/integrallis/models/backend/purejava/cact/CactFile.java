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

import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.Objects;

/** A parsed `.cact` artifact with zero-copy positional tensor access. */
public record CactFile(
    CactHeader header,
    float[] codebook,
    List<CactTensorInfo> tensorInfos,
    MemorySegment fileSegment) {

  public CactFile {
    Objects.requireNonNull(header, "header");
    codebook = codebook.clone();
    tensorInfos = List.copyOf(tensorInfos);
    Objects.requireNonNull(fileSegment, "fileSegment");
  }

  @Override
  public float[] codebook() {
    return codebook.clone();
  }

  /** Returns one positional tensor as a zero-copy slice. */
  public CactTensorData tensor(int index) {
    CactTensorInfo info = tensorInfos.get(index);
    return new CactTensorData(info, fileSegment.asSlice(info.offset(), info.byteSize()));
  }
}
