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
package com.integrallis.models.runtime.fusion;

import com.integrallis.models.api.InferenceBackend;
import com.integrallis.models.api.ModelMetadata;
import com.integrallis.models.api.RewindableInferenceBackend;
import com.integrallis.models.api.Tokenizer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.function.Consumer;

/**
 * Several same-tokenizer models presented as one backend: a lead that always steps, and auxiliaries
 * consulted only when an {@link EnsembleGate} says so.
 *
 * <p>Every member keeps its own weights and its own KV cache; nothing is shared physically. Per
 * step the lead runs, its distribution's entropy is read, and if the gate does not fire the lead's
 * logits are returned untouched, so a gate that never fires makes this backend the lead alone. If
 * the gate fires, each auxiliary catches up over the tokens it has not yet seen in one {@link
 * InferenceBackend#prefill(int[], int)} call, the batched prompt path, and the members'
 * distributions are combined under the recipe's {@link FusionRule}. The combined scores are
 * returned in place of logits: the sampler's {@code softmax(s / T)} over them is the fused
 * distribution.
 *
 * <p>This is the production form of the token-level fusion the {@code models-bench logit-fusion}
 * study measured every-token in September 2026, with the gate that study listed as its first
 * follow-on. It composes any backends; it does not require them to be loaded the same way.
 * Rewinding needs every member to be a {@link RewindableInferenceBackend}.
 *
 * <p>Single-threaded lifecycle, as for every {@link InferenceBackend}. The members' own thread
 * pools do their work; the backend calls them one after another.
 */
public final class FusedInferenceBackend implements RewindableInferenceBackend {

  private final List<InferenceBackend> members;
  private final InferenceBackend lead;
  private final List<InferenceBackend> auxiliaries;
  private final FusionRecipe recipe;
  private final double[] weights;
  private final Consumer<FusionStep> trace;
  private final String name;
  private final int vocabulary;

  /** Every token fed so far, prompt and generated, in order; the auxiliaries catch up from it. */
  private final List<Integer> fed = new ArrayList<>();

  /** Per auxiliary, how many of {@link #fed} it has consumed. */
  private final int[] synced;

  private FusionMetrics metrics = FusionMetrics.empty();
  private boolean closed;

  /**
   * Composes {@code members} under {@code recipe}.
   *
   * @param members the lead first, then the auxiliaries; at least two, all with one vocabulary
   * @param recipe rule, weights (lead first) and gate; its weight count must equal the member count
   * @param trace receives one {@link FusionStep} per step, or {@code null}
   */
  public FusedInferenceBackend(
      List<? extends InferenceBackend> members, FusionRecipe recipe, Consumer<FusionStep> trace) {
    Objects.requireNonNull(members, "members");
    this.recipe = Objects.requireNonNull(recipe, "recipe");
    if (members.size() < 2) {
      throw new IllegalArgumentException("a fusion needs at least two members");
    }
    if (recipe.members() != members.size()) {
      throw new IllegalArgumentException(
          "recipe has " + recipe.members() + " weights for " + members.size() + " members");
    }
    this.members = List.copyOf(members);
    this.lead = this.members.getFirst();
    this.auxiliaries = this.members.subList(1, this.members.size());
    this.weights = recipe.weights();
    this.trace = trace;
    this.vocabulary = lead.metadata().vocabSize();
    for (InferenceBackend member : this.members) {
      Objects.requireNonNull(member, "member");
      if (!(member instanceof RewindableInferenceBackend)) {
        throw new IllegalArgumentException(
            "every member must be rewindable; " + member.name() + " is not");
      }
      int size = member.metadata().vocabSize();
      if (size != vocabulary || member.tokenizer().vocabSize() != lead.tokenizer().vocabSize()) {
        throw new IllegalArgumentException(
            "members must share one vocabulary: "
                + lead.name()
                + " has "
                + vocabulary
                + ", "
                + member.name()
                + " has "
                + size);
      }
    }
    this.synced = new int[auxiliaries.size()];
    StringBuilder label = new StringBuilder("fused(");
    for (int i = 0; i < this.members.size(); i++) {
      label.append(i == 0 ? "" : "+").append(this.members.get(i).name());
    }
    this.name = label.append(')').toString();
  }

  /** Composes {@code members} under {@code recipe} without a step trace. */
  public FusedInferenceBackend(List<? extends InferenceBackend> members, FusionRecipe recipe) {
    this(members, recipe, null);
  }

  /** The recipe in force. */
  public FusionRecipe recipe() {
    return recipe;
  }

  /** The members, lead first. */
  public List<InferenceBackend> members() {
    return members;
  }

  /** Totals since the last {@link #reset()}. */
  public FusionMetrics metrics() {
    return metrics;
  }

  @Override
  public String name() {
    return name;
  }

  @Override
  public ModelMetadata metadata() {
    return lead.metadata();
  }

  @Override
  public int contextCapacity() {
    int capacity = Integer.MAX_VALUE;
    for (InferenceBackend member : members) {
      capacity = Math.min(capacity, member.contextCapacity());
    }
    return capacity;
  }

  @Override
  public Tokenizer tokenizer() {
    return lead.tokenizer();
  }

  @Override
  public float[] prefill(int[] tokens, int startPosition) {
    checkOpen();
    Objects.requireNonNull(tokens, "tokens");
    if (tokens.length == 0) {
      throw new IllegalArgumentException("tokens must not be empty");
    }
    requirePosition(startPosition);
    long t0 = System.nanoTime();
    float[] logits = lead.prefill(tokens, startPosition);
    long leadNanos = System.nanoTime() - t0;
    for (int token : tokens) {
      fed.add(token);
    }
    return gate(logits, fed.size() - 1, leadNanos);
  }

  @Override
  public float[] forward(int token, int position) {
    checkOpen();
    requirePosition(position);
    long t0 = System.nanoTime();
    float[] logits = lead.forward(token, position);
    long leadNanos = System.nanoTime() - t0;
    fed.add(token);
    return gate(logits, position, leadNanos);
  }

  private float[] gate(float[] leadLogits, int position, long leadNanos) {
    if (leadLogits.length != vocabulary) {
      throw new IllegalStateException(
          "lead returned " + leadLogits.length + " logits for a vocabulary of " + vocabulary);
    }
    long f0 = System.nanoTime();
    double entropy = FusionMath.entropyOfLogits(leadLogits);
    boolean consult = recipe.gate().consult(entropy, position);
    long fusionNanos = System.nanoTime() - f0;
    int[] catchUp = new int[auxiliaries.size()];
    if (!consult) {
      record(new FusionStep(position, false, entropy, catchUp, leadNanos, 0, fusionNanos));
      return leadLogits;
    }
    float[][] logits = new float[members.size()][];
    logits[0] = leadLogits;
    long auxiliaryNanos = 0;
    int target = fed.size();
    for (int i = 0; i < auxiliaries.size(); i++) {
      int from = synced[i];
      int delta = target - from;
      if (delta <= 0) {
        throw new IllegalStateException(
            "auxiliary " + auxiliaries.get(i).name() + " is ahead of the lead: " + from);
      }
      int[] unseen = new int[delta];
      for (int k = 0; k < delta; k++) {
        unseen[k] = fed.get(from + k);
      }
      long a0 = System.nanoTime();
      float[] row = auxiliaries.get(i).prefill(unseen, from);
      auxiliaryNanos += System.nanoTime() - a0;
      if (row.length != vocabulary) {
        throw new IllegalStateException(
            "auxiliary "
                + auxiliaries.get(i).name()
                + " returned "
                + row.length
                + " logits for a vocabulary of "
                + vocabulary);
      }
      logits[i + 1] = row;
      synced[i] = target;
      catchUp[i] = delta;
    }
    long c0 = System.nanoTime();
    double[][] logProbabilities = new double[members.size()][];
    for (int i = 0; i < members.size(); i++) {
      logProbabilities[i] = FusionMath.logSoftmax(logits[i]);
    }
    double[] scores = FusionMath.combine(recipe.rule(), logits, logProbabilities, weights);
    float[] fused = new float[vocabulary];
    for (int t = 0; t < vocabulary; t++) {
      fused[t] = (float) scores[t];
    }
    fusionNanos += System.nanoTime() - c0;
    record(
        new FusionStep(position, true, entropy, catchUp, leadNanos, auxiliaryNanos, fusionNanos));
    return fused;
  }

  private void record(FusionStep step) {
    metrics = metrics.plus(step);
    if (trace != null) {
      trace.accept(step);
    }
  }

  @Override
  public void reset() {
    checkOpen();
    for (InferenceBackend member : members) {
      member.reset();
    }
    fed.clear();
    Arrays.fill(synced, 0);
    metrics = FusionMetrics.empty();
  }

  @Override
  public int checkpoint() {
    checkOpen();
    int position = fed.size();
    int leadCheckpoint = ((RewindableInferenceBackend) lead).checkpoint();
    if (leadCheckpoint != position) {
      throw new IllegalStateException(
          "lead checkpoint " + leadCheckpoint + " disagrees with the fed token count " + position);
    }
    return position;
  }

  @Override
  public void rewind(int checkpoint) {
    checkOpen();
    if (checkpoint < 0 || checkpoint > fed.size()) {
      throw new IllegalArgumentException(
          "checkpoint " + checkpoint + " outside [0, " + fed.size() + "]");
    }
    ((RewindableInferenceBackend) lead).rewind(checkpoint);
    for (int i = 0; i < auxiliaries.size(); i++) {
      if (synced[i] > checkpoint) {
        ((RewindableInferenceBackend) auxiliaries.get(i)).rewind(checkpoint);
        synced[i] = checkpoint;
      }
    }
    while (fed.size() > checkpoint) {
      fed.removeLast();
    }
  }

  @Override
  public void close() {
    if (closed) {
      return;
    }
    closed = true;
    RuntimeException failure = null;
    for (InferenceBackend member : members) {
      try {
        member.close();
      } catch (RuntimeException e) {
        if (failure == null) {
          failure = e;
        } else {
          failure.addSuppressed(e);
        }
      }
    }
    if (failure != null) {
      throw failure;
    }
  }

  private void requirePosition(int position) {
    if (position != fed.size()) {
      throw new IllegalStateException(
          "position " + position + " is not the next position " + fed.size());
    }
  }

  private void checkOpen() {
    if (closed) {
      throw new IllegalStateException("backend is closed");
    }
  }
}
