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

import java.util.Arrays;

/** Explicitly reviewed publisher references for formats unavailable in the ordinary comparators. */
public enum RagPublisherReference {
  /** Prism's exact PTQ1_0 artifact and its publisher-owned CPU implementation. */
  BONSAI2_PTQ1_0(
      "prism-bonsai2-ptq1_0",
      "prism_ml_ternary_bonsai_2_27b_gguf_ptq1_0",
      "53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3",
      5_946_648_928L,
      "https://github.com/PrismML-Eng/llama.cpp",
      "e8fd209c975c80aa7fb36a8423571ff3b1b7e955",
      "PrismML-Eng/llama.cpp",
      ""),

  /**
   * EdgeRock-4B Q4_K_M and the publisher's RMA patch, which upstream llama.cpp does not carry.
   *
   * <p>Pinned differently from Bonsai of necessity. Bonsai ships a fork, so its revision alone
   * identifies the implementation. EdgeRock ships {@code backend/llama-cpp-runtime.patch} against
   * an upstream commit, and that commit <b>without</b> the patch does not serve the architecture at
   * all -- so the patch is part of the reference and is pinned by digest. Verified on 2026-10-30:
   * all nine hunks apply cleanly at this revision and the CPU build succeeds.
   */
  EDGEROCK_4B_Q4_K_M(
      "edgerock-4b-q4_k_m",
      "hkust_aaron_edgerock_4b_gguf_q4_k_m",
      "67c354bcb2abe0cba407fbfe8e7f7cc100dd44458b6fbdfee95ae22f3aaa2c6a",
      2_936_478_592L,
      "https://github.com/ggml-org/llama.cpp",
      "62318cd57dc63900ecb153272c0b10f78550855d",
      "ggml-org/llama.cpp",
      "edgerock-rma@6a4c32359d88c84a8cee2a6920847220da61d7931d406dc2c0970f0da9dbf20f");

  private final String id;
  private final String modelId;
  private final String artifactSha256;
  private final long artifactSizeBytes;
  private final String repository;
  private final String revision;

  /** The slug the comparator's {@code backendVersion} is stamped with. */
  private final String backendSlug;

  /**
   * An out-of-tree patch that is part of the reference, or empty when the revision is sufficient.
   */
  private final String patchId;

  RagPublisherReference(
      String id,
      String modelId,
      String artifactSha256,
      long artifactSizeBytes,
      String repository,
      String revision,
      String backendSlug,
      String patchId) {
    this.id = id;
    this.modelId = modelId;
    this.artifactSha256 = artifactSha256;
    this.artifactSizeBytes = artifactSizeBytes;
    this.repository = repository;
    this.revision = revision;
    this.backendSlug = backendSlug;
    this.patchId = patchId;
  }

  public String id() {
    return id;
  }

  public String modelId() {
    return modelId;
  }

  public String artifactSha256() {
    return artifactSha256;
  }

  public long artifactSizeBytes() {
    return artifactSizeBytes;
  }

  public String repository() {
    return repository;
  }

  public String revision() {
    return revision;
  }

  /** The patch that is part of this reference, or empty when the revision alone identifies it. */
  public String patchId() {
    return patchId;
  }

  public String backendVersion() {
    String pinned = backendSlug + "@" + revision;
    return patchId.isEmpty() ? pinned : pinned + "+" + patchId;
  }

  boolean matchesArtifact(RagBenchmarkReport report) {
    return modelId.equals(report.modelId())
        && artifactSha256.equals(report.artifactSha256())
        && artifactSizeBytes == report.artifactSizeBytes();
  }

  boolean matchesComparator(RagBenchmarkReport report) {
    return matchesArtifact(report)
        && "llama.cpp".equals(report.backend())
        && backendVersion().equals(report.backendVersion());
  }

  public static RagPublisherReference fromId(String id) {
    return Arrays.stream(values())
        .filter(reference -> reference.id.equals(id))
        .findFirst()
        .orElseThrow(() -> new IllegalArgumentException("unknown publisher reference: " + id));
  }
}
