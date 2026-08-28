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

import java.util.Objects;
import java.util.Optional;

/** Executable weight views for the single Needle 2 architecture serialized by `.cact`. */
public final class Needle2Weights {

  record Layer(
      float[] normIn,
      CactCqMatrix query,
      CactCqMatrix key,
      CactCqMatrix value,
      float[] queryNorm,
      float[] keyNorm,
      CactCqMatrix attentionGateProjection,
      CactCqMatrix output,
      float[] postAttentionNorm,
      float attentionGate,
      float[] preHadamardNorm,
      float[] hadamardD1,
      float[] hadamardD2,
      float[] hadamardD3,
      CactCqMatrix mhcPhiPre,
      CactCqMatrix mhcPhiPost,
      CactCqMatrix mhcPhiResidual) {}

  record Engram(CactCqMatrix tables, CactCqMatrix key, CactCqMatrix value, float[] taps) {}

  final CactHeader header;
  final CactCqMatrix embedding;
  final Layer[] layers;
  final float[] mhcAPre;
  final float[] mhcAPost;
  final float[] mhcAResidual;
  final float[] mhcBPre;
  final float[] mhcBPost;
  final float[] mhcBResidual;
  final Engram[] engrams;
  final float[] finalNorm;
  final Needle2ProbeHead contrastiveHead;
  final Needle2ProbeHead confidenceHead;

  private Needle2Weights(
      CactHeader header,
      CactCqMatrix embedding,
      Layer[] layers,
      float[] mhcAPre,
      float[] mhcAPost,
      float[] mhcAResidual,
      float[] mhcBPre,
      float[] mhcBPost,
      float[] mhcBResidual,
      Engram[] engrams,
      float[] finalNorm,
      Needle2ProbeHead contrastiveHead,
      Needle2ProbeHead confidenceHead) {
    this.header = header;
    this.embedding = embedding;
    this.layers = layers;
    this.mhcAPre = mhcAPre;
    this.mhcAPost = mhcAPost;
    this.mhcAResidual = mhcAResidual;
    this.mhcBPre = mhcBPre;
    this.mhcBPost = mhcBPost;
    this.mhcBResidual = mhcBResidual;
    this.engrams = engrams;
    this.finalNorm = finalNorm;
    this.contrastiveHead = contrastiveHead;
    this.confidenceHead = confidenceHead;
  }

  public static Needle2Weights load(CactNeedle2Layout layout) {
    Objects.requireNonNull(layout, "layout");
    CactHeader header = layout.header();
    float[] codebook = layout.codebook();
    int layers = header.layerCount();
    int lanes = header.mhcLanes();
    CactCqMatrix phiPre = cq(layout, codebook, "mhc_phi_pre");
    CactCqMatrix phiPost = cq(layout, codebook, "mhc_phi_post");
    CactCqMatrix phiResidual = cq(layout, codebook, "mhc_phi_res");
    Layer[] layerWeights = new Layer[layers];
    for (int layer = 0; layer < layers; layer++) {
      String prefix = "layer%02d.".formatted(layer);
      layerWeights[layer] =
          new Layer(
              fp16(layout, prefix + "norm_in"),
              cq(layout, codebook, prefix + "q_proj"),
              cq(layout, codebook, prefix + "k_proj"),
              cq(layout, codebook, prefix + "v_proj"),
              fp16(layout, prefix + "q_norm"),
              fp16(layout, prefix + "k_norm"),
              cq(layout, codebook, prefix + "gate_proj"),
              cq(layout, codebook, prefix + "out_proj"),
              fp16(layout, prefix + "post_norm"),
              fp16(layout, prefix + "attn_gate")[0],
              fp16(layout, prefix + "pre_hada"),
              fp16(layout, prefix + "d1"),
              fp16(layout, prefix + "d2"),
              fp16(layout, prefix + "d3"),
              phiPre.rowSlice(layer * lanes, lanes),
              phiPost.rowSlice(layer * lanes, lanes),
              phiResidual.rowSlice(layer * lanes * lanes, lanes * lanes));
    }

    Engram[] engrams = new Engram[header.engramSites().size()];
    for (int site = 0; site < engrams.length; site++) {
      String prefix = "engram" + site + ".";
      engrams[site] =
          new Engram(
              cq(layout, codebook, prefix + "tables"),
              cq(layout, codebook, prefix + "key_proj"),
              cq(layout, codebook, prefix + "value_proj"),
              fp16(layout, prefix + "taps"));
    }

    return new Needle2Weights(
        header,
        cq(layout, codebook, "embedding"),
        layerWeights,
        fp16(layout, "mhc_a_pre"),
        fp16(layout, "mhc_a_post"),
        fp16(layout, "mhc_a_res"),
        fp16(layout, "mhc_b_pre"),
        fp16(layout, "mhc_b_post"),
        fp16(layout, "mhc_b_res"),
        engrams,
        fp16(layout, "final_norm"),
        probeHead(layout, "contrastive_head", 4, true),
        probeHead(layout, "confidence_head", 8, false));
  }

  /** Returns the loaded model geometry. */
  public CactHeader header() {
    return header;
  }

  Optional<Needle2ProbeHead> contrastiveHead() {
    return Optional.ofNullable(contrastiveHead);
  }

  Optional<Needle2ProbeHead> confidenceHead() {
    return Optional.ofNullable(confidenceHead);
  }

  /** Whether the artifact carries Needle's contrastive retrieval head. */
  public boolean supportsContrastiveHead() {
    return contrastiveHead != null;
  }

  /** Returns the contrastive output dimension. */
  public int contrastiveDimension() {
    if (contrastiveHead == null) {
      throw new UnsupportedOperationException("Needle artifact has no contrastive head");
    }
    return contrastiveHead.outputWidth();
  }

  /** Whether the artifact carries Needle's calibrated confidence head. */
  public boolean supportsConfidenceHead() {
    return confidenceHead != null;
  }

  private static Needle2ProbeHead probeHead(
      CactNeedle2Layout layout, String prefix, int probeCount, boolean normalize) {
    String probesName = prefix + ".probes";
    if (!layout.hasTensor(probesName)) {
      return null;
    }
    CactTensorData projectionTensor = layout.tensor(prefix + ".proj");
    int outputWidth = Math.toIntExact(projectionTensor.info().shape()[0]);
    return new Needle2ProbeHead(
        layout.header().modelWidth(),
        probeCount,
        outputWidth,
        fp16(layout, probesName),
        CactFp16Tensor.from(projectionTensor).readAll(),
        fp16(layout, prefix + ".bias"),
        normalize);
  }

  private static CactCqMatrix cq(CactNeedle2Layout layout, float[] codebook, String tensorName) {
    return CactCqMatrix.from(layout.tensor(tensorName), codebook);
  }

  private static float[] fp16(CactNeedle2Layout layout, String tensorName) {
    return CactFp16Tensor.from(layout.tensor(tensorName)).readAll();
  }
}
