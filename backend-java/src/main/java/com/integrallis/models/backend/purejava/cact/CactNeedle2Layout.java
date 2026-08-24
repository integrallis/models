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

import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/** Validated names over the positional tensor layout exported by Needle 2. */
public final class CactNeedle2Layout {

  private static final int TENSORS_PER_LAYER = 14;
  private static final int MHC_TENSORS = 9;
  private static final int TENSORS_PER_ENGRAM_SITE = 4;
  private static final int CONTRASTIVE_HEAD = 1;
  private static final int CONFIDENCE_HEAD = 2;
  private static final ValueLayout.OfShort LE_SHORT =
      ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

  private final CactFile file;
  private final List<String> tensorNames;
  private final Map<String, Integer> tensorIndices;

  private CactNeedle2Layout(
      CactFile file, List<String> tensorNames, Map<String, Integer> tensorIndices) {
    this.file = file;
    this.tensorNames = List.copyOf(tensorNames);
    this.tensorIndices = Map.copyOf(tensorIndices);
  }

  /** Validates and names every positional tensor in a parsed Needle 2 artifact. */
  public static CactNeedle2Layout from(CactFile file) {
    Objects.requireNonNull(file, "file");
    CactHeader header = file.header();
    Binder tensors = new Binder(file);
    long width = header.modelWidth();
    long attentionWidth = Math.multiplyExact(header.queryHeadCount(), header.headWidth());
    long kvWidth = Math.multiplyExact(header.kvHeadCount(), header.headWidth());
    long hadamardWidth = header.hadamardSize();
    long lanes = header.mhcLanes();
    long laneWidth = Math.multiplyExact(lanes, width);

    tensors.cq("embedding", header.vocabularySize(), width);
    for (int layer = 0; layer < header.layerCount(); layer++) {
      String prefix = "layer%02d.".formatted(layer);
      tensors.fp16(prefix + "norm_in", width);
      tensors.cq(prefix + "q_proj", attentionWidth, width);
      tensors.cq(prefix + "k_proj", kvWidth, width);
      tensors.cq(prefix + "v_proj", kvWidth, width);
      tensors.fp16(prefix + "q_norm", header.headWidth());
      tensors.fp16(prefix + "k_norm", header.headWidth());
      tensors.cq(prefix + "gate_proj", attentionWidth, width);
      tensors.cq(prefix + "out_proj", width, attentionWidth);
      tensors.fp16(prefix + "post_norm", width);
      tensors.fp16(prefix + "attn_gate", 1);
      tensors.fp16(prefix + "pre_hada", width);
      tensors.fp16(prefix + "d1", hadamardWidth);
      tensors.fp16(prefix + "d2", hadamardWidth);
      tensors.fp16(prefix + "d3", hadamardWidth);
    }

    tensors.fp16("mhc_a_pre", header.layerCount());
    tensors.fp16("mhc_a_post", header.layerCount());
    tensors.fp16("mhc_a_res", header.layerCount());
    tensors.fp16("mhc_b_pre", header.layerCount(), lanes);
    tensors.fp16("mhc_b_post", header.layerCount(), lanes);
    tensors.fp16("mhc_b_res", header.layerCount(), lanes, lanes);
    tensors.cq("mhc_phi_pre", Math.multiplyExact(header.layerCount(), lanes), laneWidth);
    tensors.cq("mhc_phi_post", Math.multiplyExact(header.layerCount(), lanes), laneWidth);
    tensors.cq(
        "mhc_phi_res",
        Math.multiplyExact(header.layerCount(), Math.multiplyExact(lanes, lanes)),
        laneWidth);

    long engramInputWidth =
        Math.multiplyExact(header.engramTableCount(), header.engramSubDimension());
    for (int site = 0; site < header.engramSites().size(); site++) {
      String prefix = "engram" + site + ".";
      tensors.cq(
          prefix + "tables",
          Math.multiplyExact(header.engramTableCount(), header.engramSlots()),
          header.engramSubDimension());
      tensors.cq(prefix + "key_proj", width, engramInputWidth);
      tensors.cq(prefix + "value_proj", width, engramInputWidth);
      tensors.fp16(prefix + "taps", header.engramConvolutionTaps(), width);
    }
    tensors.fp16("final_norm", width);

    int remaining = file.tensorInfos().size() - tensors.index();
    if (remaining < 1) {
      throw new MalformedCactException("Needle 2 layout has no tokenizer tensor");
    }
    int headPayload = remaining - 1;
    if (headPayload > 0) {
      if ((headPayload - 1) % 3 != 0) {
        throw new MalformedCactException(
            "probe-head tensor count must be one manifest plus three tensors per head");
      }
      int headCount = (headPayload - 1) / 3;
      if (headCount < 1 || headCount > 2) {
        throw new MalformedCactException("Needle 2 supports one or two serialized probe heads");
      }
      CactTensorData manifest = tensors.fp16("heads.manifest", headCount);
      int previousCode = 0;
      for (int head = 0; head < headCount; head++) {
        float encoded =
            Float.float16ToFloat(
                manifest.data().get(LE_SHORT, Math.multiplyExact((long) head, Short.BYTES)));
        int code = (int) encoded;
        if (encoded != code
            || (code != CONTRASTIVE_HEAD && code != CONFIDENCE_HEAD)
            || code <= previousCode) {
          throw new MalformedCactException(
              "probe-head manifest must contain unique canonical codes 1 then 2");
        }
        bindProbeHead(tensors, code, width);
        previousCode = code;
      }
    }
    tensors.raw("tokenizer");
    tensors.requireComplete();
    return new CactNeedle2Layout(file, tensors.names, tensors.indices);
  }

  private static void bindProbeHead(Binder tensors, int code, long width) {
    if (code == CONTRASTIVE_HEAD) {
      tensors.fp16("contrastive_head.probes", 4, width);
      CactTensorInfo projection = tensors.peek("contrastive_head.proj");
      long[] shape = projection.shape();
      if (projection.type() != CactTensorType.FP16
          || shape.length != 2
          || shape[0] <= 0
          || shape[1] != Math.multiplyExact(4, width)) {
        throw tensors.mismatch(
            "contrastive_head.proj", CactTensorType.FP16, -1, Math.multiplyExact(4, width));
      }
      long outputWidth = shape[0];
      tensors.fp16("contrastive_head.proj", outputWidth, Math.multiplyExact(4, width));
      tensors.fp16("contrastive_head.bias", outputWidth);
      return;
    }
    tensors.fp16("confidence_head.probes", 8, width);
    tensors.fp16("confidence_head.proj", 1, Math.multiplyExact(8, width));
    tensors.fp16("confidence_head.bias", 1);
  }

  /** Returns the canonical tensor names in serialized order. */
  public List<String> tensorNames() {
    return tensorNames;
  }

  /** Returns a named zero-copy tensor slice. */
  public CactTensorData tensor(String name) {
    Integer index = tensorIndices.get(Objects.requireNonNull(name, "name"));
    if (index == null) {
      throw new IllegalArgumentException("Unknown Needle 2 tensor: " + name);
    }
    return file.tensor(index);
  }

  /** Returns the artifact's embedded tokenizer. */
  public CactTokenizer tokenizer() {
    return CactTokenizer.from(file);
  }

  private static final class Binder {
    private final CactFile file;
    private final List<String> names = new ArrayList<>();
    private final Map<String, Integer> indices = new LinkedHashMap<>();
    private int index;

    private Binder(CactFile file) {
      this.file = file;
    }

    private int index() {
      return index;
    }

    private CactTensorData cq(String name, long... shape) {
      return bind(name, CactTensorType.CQ, shape);
    }

    private CactTensorData fp16(String name, long... shape) {
      return bind(name, CactTensorType.FP16, shape);
    }

    private CactTensorData raw(String name) {
      return bind(name, CactTensorType.RAW);
    }

    private CactTensorInfo peek(String name) {
      if (index >= file.tensorInfos().size()) {
        throw new MalformedCactException("missing positional tensor " + name);
      }
      return file.tensorInfos().get(index);
    }

    private CactTensorData bind(String name, CactTensorType type, long... shape) {
      CactTensorInfo actual = peek(name);
      if (actual.type() != type || !Arrays.equals(actual.shape(), shape)) {
        throw mismatch(name, type, shape);
      }
      if (indices.putIfAbsent(name, index) != null) {
        throw new IllegalStateException("duplicate canonical tensor name " + name);
      }
      names.add(name);
      return file.tensor(index++);
    }

    private MalformedCactException mismatch(
        String name, CactTensorType expectedType, long... expectedShape) {
      CactTensorInfo actual = peek(name);
      return new MalformedCactException(
          "tensor "
              + actual.index()
              + " ("
              + name
              + ") must be "
              + expectedType
              + " "
              + Arrays.toString(expectedShape)
              + "; got "
              + actual.type()
              + " "
              + Arrays.toString(actual.shape()));
    }

    private void requireComplete() {
      if (index != file.tensorInfos().size()) {
        throw new MalformedCactException(
            "Needle 2 layout leaves " + (file.tensorInfos().size() - index) + " unnamed tensors");
      }
    }
  }
}
