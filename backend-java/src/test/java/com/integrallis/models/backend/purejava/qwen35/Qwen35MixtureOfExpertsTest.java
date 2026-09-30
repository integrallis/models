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
package com.integrallis.models.backend.purejava.qwen35;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

import com.integrallis.models.backend.purejava.PureJavaBackend;
import com.integrallis.models.backend.purejava.gguf.GgufParser;
import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.gguf.SyntheticGgufBuilder;
import java.lang.foreign.Arena;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Random;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * The routed feed-forward on the Qwen3.5 hybrid graph.
 *
 * <p>The load-bearing test here is {@link #clonedExpertsReproduceTheDenseEquivalentExactly()}. A
 * routed layer whose experts are all copies of one dense triple has to compute what the dense layer
 * computes, because the routing weights sum to one -- so the dense path, which is separately pinned
 * against eight qualified models, acts as the reference implementation for the routed one. That
 * single equality covers the expert slicing, the softmax renormalisation and the residual wiring at
 * once, and it does so without a hand-derived expected vector, which is the thing most likely to be
 * wrong in the same direction as the code.
 *
 * <p>What that test cannot see is which slice a router row is paired with, since every expert holds
 * the same numbers. {@link #permutingExpertsAndTheirRouterRowsLeavesTheOutputUnchanged()} covers
 * that separately, and the shared-expert tests cover the branch the dense equivalent deliberately
 * switches off.
 */
@Tag("unit")
class Qwen35MixtureOfExpertsTest {

  private static final int DIMENSION = 8;
  private static final int HEAD_DIMENSION = 4;
  private static final int VALUE_HEADS = 2;
  private static final int VALUE_DIMENSION = HEAD_DIMENSION * VALUE_HEADS;
  private static final int KEY_DIMENSION = HEAD_DIMENSION;
  private static final int CONVOLUTION_DIMENSION = 2 * KEY_DIMENSION + VALUE_DIMENSION;
  private static final int VOCABULARY_SIZE = 8;
  private static final int LAYERS = 2;

  /** The dense hidden width and the expert hidden width have to match for the clone equality. */
  private static final int EXPERT_HIDDEN = 8;

  private static final int SHARED_HIDDEN = 6;
  private static final int EXPERTS = 4;
  private static final int EXPERTS_USED = 2;

  private static final int[] NATURAL = {0, 1, 2, 3};

  /**
   * Two identical experts summed with weights that add to one are the one expert, so this must
   * reproduce the dense path. Not approximately for the wrong reason: the only licence for any
   * difference at all is that {@code w0 * y + w1 * y} is not bit-identical to {@code y} in float,
   * and that the softmax weights sum to one only to float precision.
   */
  @Test
  void clonedExpertsReproduceTheDenseEquivalentExactly(@TempDir Path directory) throws Exception {
    float[] dense = decode(write(directory, "dense.gguf", Options.dense()));
    float[] routed = decode(write(directory, "cloned.gguf", Options.cloned()));

    assertThat(routed).hasSameSizeAs(dense);
    for (int index = 0; index < dense.length; index++) {
      assertThat(routed[index])
          .describedAs("routed logit %s must equal the dense equivalent", index)
          .isEqualTo(dense[index], within(1.0e-5f));
    }
  }

  /**
   * The clone equality above holds only because that fixture zeroes {@code ffn_down_shexp}. If the
   * shared expert were never invoked, switching it on would change nothing and the equality would
   * be passing for the wrong reason.
   */
  @Test
  void theSharedExpertContributesToEveryToken(@TempDir Path directory) throws Exception {
    float[] withoutShared = decode(write(directory, "cloned.gguf", Options.cloned()));
    float[] withShared =
        decode(write(directory, "cloned-shared.gguf", Options.cloned().withLiveSharedDown()));

    assertThat(withShared)
        .describedAs("a live shared expert must reach the output")
        .isNotEqualTo(withoutShared);
  }

  /** {@code ffn_gate_inp_shexp} scales the shared contribution, so its weights must be read. */
  @Test
  void theSharedExpertGateIsRead(@TempDir Path directory) throws Exception {
    Options live = Options.distinct();
    float[] gated = decode(write(directory, "distinct.gguf", live));
    float[] ungated = decode(write(directory, "distinct-flat.gguf", live.withZeroSharedGate()));

    // A zeroed gate row is sigmoid(0) = 0.5 rather than nothing, so this asserts the gate is a
    // computed scalar and not a constant one.
    assertThat(gated)
        .describedAs("the shared expert's sigmoid gate must affect the output")
        .isNotEqualTo(ungated);
  }

  /**
   * Pairs each router row with its own expert slice.
   *
   * <p>GGUF stacks the experts into one tensor and {@code expertSlices} cuts it by byte offset. An
   * offset that is wrong by a whole expert still produces finite, plausible output, so nothing else
   * here would notice. Permuting the slices and their router rows together is a no-op only if that
   * pairing is right: under a shifted slicing the two fixtures pair different router rows with
   * different experts and disagree.
   */
  @Test
  void permutingExpertsAndTheirRouterRowsLeavesTheOutputUnchanged(@TempDir Path directory)
      throws Exception {
    float[] natural = decode(write(directory, "natural.gguf", Options.distinct()));
    float[] swapped =
        decode(write(directory, "swapped.gguf", Options.distinct().withOrder(1, 0, 2, 3)));

    assertThat(swapped).hasSameSizeAs(natural);
    for (int index = 0; index < natural.length; index++) {
      assertThat(swapped[index])
          .describedAs("permuted logit %s", index)
          .isEqualTo(natural[index], within(1.0e-6f));
    }
  }

  /** {@code expert_used_count} must actually restrict how many experts a token sums. */
  @Test
  void theUsedExpertCountIsHonoured(@TempDir Path directory) throws Exception {
    float[] twoOfFour = decode(write(directory, "two.gguf", Options.distinct()));
    float[] allFour = decode(write(directory, "all.gguf", Options.distinct().withUsed(EXPERTS)));

    // Distinct experts, so summing all four cannot coincide with summing the best two.
    assertThat(allFour)
        .describedAs("expert_used_count must change what is summed")
        .isNotEqualTo(twoOfFour);
  }

  /**
   * Pins the shared expert's arithmetic, not merely its presence.
   *
   * <p>{@link #theSharedExpertGateIsRead(Path)} only shows that the branch changes the output, so
   * it survives a gate/up swap inside the shared SwiGLU -- both of its fixtures would be wrong
   * identically. That swap is the likeliest mistake here: SwiGLU is {@code silu(gate) * up}, and
   * the reference's own {@code build_ffn} takes those two arguments the other way round.
   *
   * <p>So: silence every routed expert, give the shared expert a dense triple, and zero its gate
   * row so the sigmoid is exactly 0.5. The output must then be the dense feed-forward halved, which
   * a dense fixture with a halved {@code ffn_down} computes independently, {@code down} being
   * linear. Any misordering inside the shared SwiGLU breaks that equality.
   */
  @Test
  void theSharedExpertComputesAHalvedDenseFeedForward(@TempDir Path directory) throws Exception {
    float[] shared = decode(write(directory, "shared-only.gguf", Options.sharedOnly()));
    float[] halvedDense =
        decode(write(directory, "dense-half.gguf", Options.dense().withDenseDownScale(0.5f)));

    assertThat(shared).hasSameSizeAs(halvedDense);
    for (int index = 0; index < halvedDense.length; index++) {
      assertThat(shared[index])
          .describedAs("shared-expert logit %s must equal the halved dense feed-forward", index)
          .isEqualTo(halvedDense[index], within(1.0e-5f));
    }
  }

  /** The control: unhalved, that same comparison must fail, or the halving proves nothing. */
  @Test
  void theHalvingInThatEquivalenceIsLoadBearing(@TempDir Path directory) throws Exception {
    float[] shared = decode(write(directory, "shared-only.gguf", Options.sharedOnly()));
    float[] fullDense = decode(write(directory, "dense-full.gguf", Options.dense()));

    assertThat(shared)
        .describedAs(
            "the shared expert is gated at 0.5, so it cannot equal the unhalved dense path")
        .isNotEqualTo(fullDense);
  }

  /**
   * The route the fleet actually takes: a file whose {@code general.architecture} is {@code
   * qwen35moe} has to reach this decoder through the backend's own dispatch, and be reported under
   * its own family rather than folded into {@code qwen35}. The two share a decoder but are not the
   * same model, and qualification records results against what is reported here.
   */
  @Test
  void theBackendLoadsARoutedFileUnderItsOwnArchitecture(@TempDir Path directory) throws Exception {
    Path model = write(directory, "distinct.gguf", Options.distinct());

    try (PureJavaBackend backend = PureJavaBackend.load(model)) {
      assertThat(backend.metadata().modelFamily()).isEqualTo("qwen35moe");
      float[] logits = backend.prefill(new int[] {1, 2, 3}, 0);

      assertThat(logits).hasSize(VOCABULARY_SIZE);
      for (float logit : logits) {
        assertThat(Float.isFinite(logit)).isTrue();
      }
    }
  }

  /**
   * The published router shape: {@code ffn_gate_inp_shexp} is one row, so GGUF writes it as a 1-D
   * tensor with the trailing 1 dropped.
   *
   * <p>Read from the real files (Qwen3.5-35B-A3B and Qwen3-Coder-Next both carry it as {@code
   * [n_embd]}), not inferred. The loader required {@code [n_embd, 1]} and would have refused every
   * published Qwen3.5-MoE; the synthetic fixtures could not see it because a builder handed {@code
   * {columns, rows}} writes both dimensions whatever rows is.
   */
  @Test
  void aOneDimensionalSharedGateRowLoadsAndMeansTheSameThing(@TempDir Path directory)
      throws Exception {
    float[] twoDimensional = decode(write(directory, "2d.gguf", Options.distinct()));
    float[] asPublished =
        decode(write(directory, "1d.gguf", Options.distinct().asPublishedRouterShape()));

    assertThat(asPublished).hasSameSizeAs(twoDimensional);
    assertThat(asPublished)
        .describedAs("dropping a trailing dimension of 1 changes the encoding, not the weights")
        .containsExactly(twoDimensional);
  }

  /**
   * Tolerating a dropped trailing dimension must not become tolerating any 1-D tensor: a router row
   * of the wrong length is a malformed file, and reading it would take the sigmoid of a dot product
   * over the wrong number of weights.
   */
  @Test
  void aOneDimensionalSharedGateOfTheWrongWidthIsRefused(@TempDir Path directory) throws Exception {
    Path model =
        write(directory, "ragged.gguf", Options.distinct().withSharedGateWidth(DIMENSION + 1));

    try (Arena arena = Arena.ofConfined()) {
      assertThatThrownBy(() -> Qwen35ForwardPass.fromGgufFile(GgufParser.parse(model, arena)))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("ffn_gate_inp_shexp.weight")
          .hasMessageContaining("shape");
    }
  }

  /**
   * An untied {@code output.weight} must be the head.
   *
   * <p>Every published Qwen3.5-MoE and Qwen3-Next carries one; the dense Qwen3.5 files do not. The
   * loader tied unconditionally, and because the token embedding has exactly the head's shape that
   * is not a load failure -- it computes every logit from the wrong matrix, which reads as fluent,
   * confident, wrong text. Nothing else in this suite would have caught it.
   */
  @Test
  void anUntiedOutputHeadIsUsedInsteadOfTheTokenEmbedding(@TempDir Path directory)
      throws Exception {
    float[] tied = decode(write(directory, "tied.gguf", Options.distinct()));
    float[] untiedDifferent =
        decode(
            write(
                directory,
                "untied.gguf",
                Options.distinct().withOutputHead(OutputHead.UNTIED_DIFFERENT)));

    assertThat(untiedDifferent)
        .describedAs("a different output head must produce different logits")
        .isNotEqualTo(tied);
  }

  /**
   * And it must be applied with the same semantics as the tied head: an {@code output.weight}
   * holding exactly the token embedding has to reproduce the tied model bit for bit.
   *
   * <p>Without this, the test above is satisfied by any change at all -- reading the head
   * transposed, or scaling it -- so long as the numbers move.
   */
  @Test
  void anOutputHeadEqualToTheEmbeddingReproducesTheTiedModelExactly(@TempDir Path directory)
      throws Exception {
    float[] tied = decode(write(directory, "tied.gguf", Options.distinct()));
    float[] untiedSame =
        decode(
            write(
                directory,
                "untied-same.gguf",
                Options.distinct().withOutputHead(OutputHead.UNTIED_SAME_AS_EMBEDDING)));

    assertThat(untiedSame).containsExactly(tied);
  }

  @Test
  void aRoutedModelDecodesFiniteLogitsAtEveryPosition(@TempDir Path directory) throws Exception {
    Path model = write(directory, "distinct.gguf", Options.distinct());

    try (Arena arena = Arena.ofConfined()) {
      Qwen35ForwardPass graph = Qwen35ForwardPass.fromGgufFile(GgufParser.parse(model, arena));
      Qwen35ForwardPass.Session session = graph.openSession(4);
      for (int position = 0; position < 4; position++) {
        float[] logits = graph.forward(session, 2 + (position % 3), position);
        assertThat(logits).hasSize(VOCABULARY_SIZE);
        for (float logit : logits) {
          assertThat(Float.isFinite(logit)).describedAs("logit at position %s", position).isTrue();
        }
      }
    }
  }

  /**
   * Routing is per token, so the batched paths -- which run one weight matrix over many rows -- do
   * not apply. The graph has to clamp its own batch size rather than trust the caller's default.
   */
  @Test
  void aRoutedModelDecodesOneTokenAtATimeWhateverBatchSizeIsAsked(@TempDir Path directory)
      throws Exception {
    Path routed = write(directory, "distinct.gguf", Options.distinct());
    Path dense = write(directory, "dense.gguf", Options.dense());

    try (Arena arena = Arena.ofConfined()) {
      assertThat(
              Qwen35ForwardPass.fromGgufFile(GgufParser.parse(routed, arena), 8).prefillBatchSize())
          .describedAs("a routed graph must stay on the single-token path")
          .isEqualTo(1);
      // The clamp is specific to routing and must not have disarmed batched prefill generally.
      assertThat(
              Qwen35ForwardPass.fromGgufFile(GgufParser.parse(dense, arena), 8).prefillBatchSize())
          .isEqualTo(8);
    }
  }

  /** Whatever batch size is clamped to, prefill and token-at-a-time must agree. */
  @Test
  void prefillAgreesWithTokenByTokenDecoding(@TempDir Path directory) throws Exception {
    Path model = write(directory, "distinct.gguf", Options.distinct());
    int[] tokens = {1, 2, 3, 4};

    try (Arena arena = Arena.ofConfined()) {
      Qwen35ForwardPass graph = Qwen35ForwardPass.fromGgufFile(GgufParser.parse(model, arena), 8);
      Qwen35ForwardPass.Session prefilled = graph.openSession(8);
      float[] batched = graph.prefill(prefilled, tokens, 0);

      Qwen35ForwardPass.Session stepped = graph.openSession(8);
      float[] serial = null;
      for (int position = 0; position < tokens.length; position++) {
        serial = graph.forward(stepped, tokens[position], position);
      }

      assertThat(batched).containsExactly(serial);
    }
  }

  /**
   * Grouped decision batches rows through one projection, which a routed layer cannot do. It
   * refuses rather than falling back, because a caller reaches for it for throughput and a silent
   * per-question fallback would report that throughput without delivering it.
   */
  @Test
  void groupedDecisionRefusesOnARoutedModel(@TempDir Path directory) throws Exception {
    Path model = write(directory, "distinct.gguf", Options.distinct());

    try (Arena arena = Arena.ofConfined()) {
      Qwen35ForwardPass graph = Qwen35ForwardPass.fromGgufFile(GgufParser.parse(model, arena));
      Qwen35ForwardPass.Session session = graph.openSession(8);
      graph.forward(session, 1, 0);

      assertThatThrownBy(() -> graph.decideGrouped(session, new int[][] {{2}, {3}}))
          .isInstanceOf(UnsupportedOperationException.class)
          .hasMessageContaining("mixture-of-experts");
    }
  }

  /**
   * The topology report reads every layer's feed-forward tensor types. On a routed layer the dense
   * triple is null, so this used to be a NullPointerException from a diagnostics call.
   */
  @Test
  void theTopologyNamesTheRoutedArchitectureAndItsExpertTensorTypes(@TempDir Path directory)
      throws Exception {
    Path model = write(directory, "distinct.gguf", Options.distinct());

    try (Arena arena = Arena.ofConfined()) {
      Qwen35ForwardPass graph = Qwen35ForwardPass.fromGgufFile(GgufParser.parse(model, arena));

      assertThat(graph.topology().architecture()).isEqualTo("qwen35moe");
      assertThat(graph.topology().layers()).hasSize(LAYERS);
      for (var layer : graph.topology().layers()) {
        assertThat(layer.gate()).isEqualTo(GgufTensorType.F32);
        assertThat(layer.up()).isEqualTo(GgufTensorType.F32);
        assertThat(layer.down()).isEqualTo(GgufTensorType.F32);
      }
    }
  }

  /** A routed fixture publishes no feed_forward_length, so the loader must not require one. */
  @Test
  void aRoutedFixtureCarriesNoDenseFeedForwardLength(@TempDir Path directory) throws Exception {
    Path model = write(directory, "distinct.gguf", Options.distinct());

    try (Arena arena = Arena.ofConfined()) {
      Qwen35Config config = Qwen35Config.fromMetadata(GgufParser.parse(model, arena).metadata());

      assertThat(config.usesMixtureOfExperts()).isTrue();
      assertThat(config.hiddenDim()).isZero();
      assertThat(config.numExperts()).isEqualTo(EXPERTS);
      assertThat(config.numExpertsUsed()).isEqualTo(EXPERTS_USED);
      assertThat(config.expertHiddenDim()).isEqualTo(EXPERT_HIDDEN);
      assertThat(config.sharedExpertHiddenDim()).isEqualTo(SHARED_HIDDEN);
    }
  }

  private static float[] decode(Path model) throws Exception {
    try (Arena arena = Arena.ofConfined()) {
      Qwen35ForwardPass graph = Qwen35ForwardPass.fromGgufFile(GgufParser.parse(model, arena));
      Qwen35ForwardPass.Session session = graph.openSession(4);
      graph.forward(session, 1, 0);
      return graph.forward(session, 2, 1);
    }
  }

  /**
   * Whether a fixture carries its own {@code output.weight}.
   *
   * <p>Both are real in this family, which is the point: the dense Qwen3.5 files tie the head to
   * the token embedding and carry no {@code output.weight}, while every published Qwen3.5-MoE and
   * Qwen3-Next carries one. Reading the embedding on an untied model is not a load failure -- the
   * shape matches -- it silently computes every logit from the wrong matrix.
   */
  private enum OutputHead {
    /** No {@code output.weight}: the head is the token embedding. */
    TIED,
    /** An {@code output.weight} holding exactly the token embedding, so it must tie numerically. */
    UNTIED_SAME_AS_EMBEDDING,
    /** An {@code output.weight} holding different weights, so it must change the logits. */
    UNTIED_DIFFERENT
  }

  /** What a fixture's feed-forward layers hold. */
  private enum Feed {
    /** The dense {@code ffn_gate}/{@code ffn_up}/{@code ffn_down} triple. */
    DENSE,
    /** Routed, with every expert a copy of what {@link #DENSE} puts in that triple. */
    CLONED_EXPERTS,
    /** Routed, with each expert holding its own weights. */
    DISTINCT_EXPERTS,
    /** Routed, with every routed expert zeroed so only the shared expert speaks. */
    SILENT_EXPERTS
  }

  /**
   * One fixture's feed-forward shape.
   *
   * @param feed dense, cloned experts, or distinct experts
   * @param used {@code expert_used_count}
   * @param order which expert's weights land in each slot, and which router row goes with it
   * @param liveSharedDown false to zero {@code ffn_down_shexp}, silencing the shared expert
   * @param liveSharedGate false to zero {@code ffn_gate_inp_shexp}, pinning its sigmoid at 0.5
   * @param oneDimensionalSharedGate write {@code ffn_gate_inp_shexp} as a 1-D tensor, which is how
   *     the published files carry it: GGUF drops a trailing dimension of 1
   * @param outputHead whether the file carries its own {@code output.weight}, and if so whose
   *     weights
   * @param sharedHidden the shared expert's hidden width, normally unrelated to the expert width
   * @param denseDownScale a factor on a dense fixture's {@code ffn_down}, for the shared-expert
   *     equivalence, where the reference has to absorb the gate's constant 0.5
   */
  private record Options(
      Feed feed,
      int used,
      int[] order,
      boolean liveSharedDown,
      boolean liveSharedGate,
      int sharedHidden,
      float denseDownScale,
      boolean oneDimensionalSharedGate,
      int sharedGateWidth,
      OutputHead outputHead) {

    static Options dense() {
      return new Options(
          Feed.DENSE,
          0,
          NATURAL,
          false,
          false,
          SHARED_HIDDEN,
          1.0f,
          false,
          DIMENSION,
          OutputHead.TIED);
    }

    /** Routed, but arithmetically the dense equivalent: identical experts, silent shared expert. */
    static Options cloned() {
      return new Options(
          Feed.CLONED_EXPERTS,
          EXPERTS_USED,
          NATURAL,
          false,
          true,
          SHARED_HIDDEN,
          1.0f,
          false,
          DIMENSION,
          OutputHead.TIED);
    }

    static Options distinct() {
      return new Options(
          Feed.DISTINCT_EXPERTS,
          EXPERTS_USED,
          NATURAL,
          true,
          true,
          SHARED_HIDDEN,
          1.0f,
          false,
          DIMENSION,
          OutputHead.TIED);
    }

    /**
     * Only the shared expert speaks, and it holds a dense triple, with its gate row zeroed so the
     * sigmoid is exactly 0.5. Its hidden width is the dense width here so the comparison is
     * possible at all; every other routed fixture keeps the two widths different on purpose.
     */
    static Options sharedOnly() {
      return new Options(
          Feed.SILENT_EXPERTS,
          EXPERTS_USED,
          NATURAL,
          true,
          false,
          EXPERT_HIDDEN,
          1.0f,
          false,
          DIMENSION,
          OutputHead.TIED);
    }

    Options withOrder(int... value) {
      return new Options(
          feed,
          used,
          value,
          liveSharedDown,
          liveSharedGate,
          sharedHidden,
          denseDownScale,
          oneDimensionalSharedGate,
          sharedGateWidth,
          outputHead);
    }

    Options withUsed(int value) {
      return new Options(
          feed,
          value,
          order,
          liveSharedDown,
          liveSharedGate,
          sharedHidden,
          denseDownScale,
          oneDimensionalSharedGate,
          sharedGateWidth,
          outputHead);
    }

    Options withLiveSharedDown() {
      return new Options(
          feed,
          used,
          order,
          true,
          liveSharedGate,
          sharedHidden,
          denseDownScale,
          oneDimensionalSharedGate,
          sharedGateWidth,
          outputHead);
    }

    Options withZeroSharedGate() {
      return new Options(
          feed,
          used,
          order,
          liveSharedDown,
          false,
          sharedHidden,
          denseDownScale,
          oneDimensionalSharedGate,
          sharedGateWidth,
          outputHead);
    }

    Options asPublishedRouterShape() {
      return new Options(
          feed,
          used,
          order,
          liveSharedDown,
          liveSharedGate,
          sharedHidden,
          denseDownScale,
          true,
          sharedGateWidth,
          outputHead);
    }

    /** A router row of the wrong length, which has to be refused rather than read. */
    Options withSharedGateWidth(int value) {
      return new Options(
          feed,
          used,
          order,
          liveSharedDown,
          liveSharedGate,
          sharedHidden,
          denseDownScale,
          true,
          value,
          outputHead);
    }

    Options withOutputHead(OutputHead value) {
      return new Options(
          feed,
          used,
          order,
          liveSharedDown,
          liveSharedGate,
          sharedHidden,
          denseDownScale,
          oneDimensionalSharedGate,
          sharedGateWidth,
          value);
    }

    Options withDenseDownScale(float value) {
      return new Options(
          feed,
          used,
          order,
          liveSharedDown,
          liveSharedGate,
          sharedHidden,
          value,
          oneDimensionalSharedGate,
          sharedGateWidth,
          outputHead);
    }

    boolean routed() {
      return feed != Feed.DENSE;
    }
  }

  private static Path write(Path directory, String fileName, Options options) throws Exception {
    // A dedicated Random for everything outside the feed-forward, so the non-FFN half of every
    // fixture is identical whatever the feed-forward mode draws. Sharing one generator would make
    // the dense and cloned fixtures differ in their attention weights, and the equality they are
    // built to test would be comparing two different models.
    Random structure = new Random(35);
    String architecture = options.routed() ? "qwen35moe" : "qwen35";
    String prefix = architecture + ".";
    SyntheticGgufBuilder builder =
        new SyntheticGgufBuilder()
            .addString("general.architecture", architecture)
            .addString("general.name", "Toy Qwen 3.5 MoE")
            .addUint32(prefix + "embedding_length", DIMENSION)
            .addUint32(prefix + "block_count", LAYERS)
            .addUint32(prefix + "attention.head_count", 2)
            .addUint32(prefix + "attention.head_count_kv", 1)
            .addUint32(prefix + "attention.key_length", HEAD_DIMENSION)
            .addUint32(prefix + "vocab_size", VOCABULARY_SIZE)
            .addUint32(prefix + "context_length", 8)
            .addFloat32(prefix + "rope.freq_base", 10_000.0f)
            .addUint32(prefix + "rope.dimension_count", HEAD_DIMENSION)
            .addFloat32(prefix + "attention.layer_norm_rms_epsilon", 1.0e-6f)
            .addUint32(prefix + "ssm.conv_kernel", 3)
            .addUint32(prefix + "ssm.state_size", HEAD_DIMENSION)
            .addUint32(prefix + "ssm.group_count", 1)
            .addUint32(prefix + "ssm.time_step_rank", VALUE_HEADS)
            .addUint32(prefix + "ssm.inner_size", VALUE_DIMENSION)
            .addUint32(prefix + "full_attention_interval", 2)
            .addStringArray(
                "tokenizer.ggml.tokens", List.of("<s>", "</s>", "a", "b", "c", "d", "e", "f"))
            .addFloat32Array(
                "tokenizer.ggml.scores", List.of(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f))
            .addUint32("tokenizer.ggml.bos_token_id", 0)
            .addUint32("tokenizer.ggml.eos_token_id", 1);
    if (options.routed()) {
      // Deliberately no feed_forward_length: a fully routed file publishes none.
      builder
          .addUint32(prefix + "expert_count", EXPERTS)
          .addUint32(prefix + "expert_used_count", options.used())
          .addUint32(prefix + "expert_feed_forward_length", EXPERT_HIDDEN)
          .addUint32(prefix + "expert_shared_feed_forward_length", options.sharedHidden());
    } else {
      builder.addUint32(prefix + "feed_forward_length", EXPERT_HIDDEN);
    }
    float[] embedding = randomFloats(structure, DIMENSION * VOCABULARY_SIZE);
    builder
        .addTensor(
            "token_embd.weight",
            GgufTensorType.F32,
            new long[] {DIMENSION, VOCABULARY_SIZE},
            bytes(embedding))
        .addTensor(
            "output_norm.weight",
            GgufTensorType.F32,
            new long[] {DIMENSION},
            bytes(ones(DIMENSION)));

    if (options.outputHead() != OutputHead.TIED) {
      float[] head =
          options.outputHead() == OutputHead.UNTIED_SAME_AS_EMBEDDING
              ? embedding
              : deterministic(4000L, DIMENSION * VOCABULARY_SIZE);
      builder.addTensor(
          "output.weight",
          GgufTensorType.F32,
          new long[] {DIMENSION, VOCABULARY_SIZE},
          bytes(head));
    }

    addGatedDeltaNetLayer(builder, structure, 0);
    addFullAttentionLayer(builder, structure, 1);
    for (int layer = 0; layer < LAYERS; layer++) {
      addFeedForward(builder, "blk." + layer + ".", layer, options);
    }
    Path model = directory.resolve(fileName);
    Files.write(model, builder.build());
    return model;
  }

  private static void addFeedForward(
      SyntheticGgufBuilder builder, String prefix, int layer, Options options) {
    if (options.feed() == Feed.DENSE) {
      addMatrix(
          builder, prefix + "ffn_gate.weight", EXPERT_HIDDEN, DIMENSION, expertGate(layer, 0));
      addMatrix(builder, prefix + "ffn_up.weight", EXPERT_HIDDEN, DIMENSION, expertUp(layer, 0));
      addMatrix(
          builder,
          prefix + "ffn_down.weight",
          DIMENSION,
          EXPERT_HIDDEN,
          scaled(expertDown(layer, 0), options.denseDownScale()));
      return;
    }
    boolean silent = options.feed() == Feed.SILENT_EXPERTS;
    boolean cloned = options.feed() == Feed.CLONED_EXPERTS;
    float[] gate = new float[EXPERTS * EXPERT_HIDDEN * DIMENSION];
    float[] up = new float[EXPERTS * EXPERT_HIDDEN * DIMENSION];
    float[] down = new float[EXPERTS * DIMENSION * EXPERT_HIDDEN];
    float[] router = new float[EXPERTS * DIMENSION];
    for (int slot = 0; slot < EXPERTS; slot++) {
      int source = cloned ? 0 : options.order()[slot];
      // The router is written even for silent experts: routing still runs, and weights that sum to
      // one over contributions of zero are what make the shared expert observable on its own.
      System.arraycopy(routerRow(layer, source), 0, router, slot * DIMENSION, DIMENSION);
      if (silent) {
        continue;
      }
      System.arraycopy(
          expertGate(layer, source),
          0,
          gate,
          slot * EXPERT_HIDDEN * DIMENSION,
          EXPERT_HIDDEN * DIMENSION);
      System.arraycopy(
          expertUp(layer, source),
          0,
          up,
          slot * EXPERT_HIDDEN * DIMENSION,
          EXPERT_HIDDEN * DIMENSION);
      System.arraycopy(
          expertDown(layer, source),
          0,
          down,
          slot * DIMENSION * EXPERT_HIDDEN,
          DIMENSION * EXPERT_HIDDEN);
    }
    builder
        .addTensor(
            prefix + "ffn_gate_exps.weight",
            GgufTensorType.F32,
            new long[] {DIMENSION, EXPERT_HIDDEN, EXPERTS},
            bytes(gate))
        .addTensor(
            prefix + "ffn_up_exps.weight",
            GgufTensorType.F32,
            new long[] {DIMENSION, EXPERT_HIDDEN, EXPERTS},
            bytes(up))
        .addTensor(
            prefix + "ffn_down_exps.weight",
            GgufTensorType.F32,
            new long[] {EXPERT_HIDDEN, DIMENSION, EXPERTS},
            bytes(down));
    addMatrix(builder, prefix + "ffn_gate_inp.weight", EXPERTS, DIMENSION, router);
    float[] sharedGateRow = options.liveSharedGate() ? sharedRow(layer) : new float[DIMENSION];
    if (options.oneDimensionalSharedGate()) {
      // As published: one router row over the model width, written with its trailing 1 dropped.
      int width = options.sharedGateWidth();
      builder.addTensor(
          prefix + "ffn_gate_inp_shexp.weight",
          GgufTensorType.F32,
          new long[] {width},
          bytes(width == DIMENSION ? sharedGateRow : new float[width]));
    } else {
      addMatrix(builder, prefix + "ffn_gate_inp_shexp.weight", 1, DIMENSION, sharedGateRow);
    }
    int sharedHidden = options.sharedHidden();
    addMatrix(
        builder,
        prefix + "ffn_gate_shexp.weight",
        sharedHidden,
        DIMENSION,
        silent ? expertGate(layer, 0) : deterministic(3000L + layer, sharedHidden * DIMENSION));
    addMatrix(
        builder,
        prefix + "ffn_up_shexp.weight",
        sharedHidden,
        DIMENSION,
        silent ? expertUp(layer, 0) : deterministic(3100L + layer, sharedHidden * DIMENSION));
    addMatrix(
        builder,
        prefix + "ffn_down_shexp.weight",
        DIMENSION,
        sharedHidden,
        sharedDown(layer, sharedHidden, silent, options.liveSharedDown()));
  }

  private static float[] sharedDown(int layer, int sharedHidden, boolean silent, boolean live) {
    if (!live) {
      return new float[DIMENSION * sharedHidden];
    }
    return silent ? expertDown(layer, 0) : deterministic(3200L + layer, DIMENSION * sharedHidden);
  }

  private static float[] scaled(float[] values, float factor) {
    if (factor == 1.0f) {
      return values;
    }
    float[] result = new float[values.length];
    for (int index = 0; index < values.length; index++) {
      result[index] = values[index] * factor;
    }
    return result;
  }

  private static float[] expertGate(int layer, int expert) {
    return deterministic(1000L + layer * 31L + expert, EXPERT_HIDDEN * DIMENSION);
  }

  private static float[] expertUp(int layer, int expert) {
    return deterministic(1500L + layer * 31L + expert, EXPERT_HIDDEN * DIMENSION);
  }

  private static float[] expertDown(int layer, int expert) {
    return deterministic(2000L + layer * 31L + expert, DIMENSION * EXPERT_HIDDEN);
  }

  private static float[] routerRow(int layer, int expert) {
    return deterministic(2500L + layer * 31L + expert, DIMENSION);
  }

  private static float[] sharedRow(int layer) {
    return deterministic(2900L + layer, DIMENSION);
  }

  /** Weights keyed by what they are rather than by draw order, so a fixture can be permuted. */
  private static float[] deterministic(long seed, int length) {
    return randomFloats(new Random(seed), length);
  }

  private static void addGatedDeltaNetLayer(
      SyntheticGgufBuilder builder, Random random, int layer) {
    String prefix = "blk." + layer + ".";
    addNorms(builder, prefix);
    addRandomMatrix(builder, random, prefix + "attn_qkv.weight", CONVOLUTION_DIMENSION, DIMENSION);
    addRandomMatrix(builder, random, prefix + "attn_gate.weight", VALUE_DIMENSION, DIMENSION);
    addRandomMatrix(builder, random, prefix + "ssm_beta.weight", VALUE_HEADS, DIMENSION);
    addRandomMatrix(builder, random, prefix + "ssm_alpha.weight", VALUE_HEADS, DIMENSION);
    builder
        .addTensor(
            prefix + "ssm_conv1d.weight",
            GgufTensorType.F32,
            new long[] {3, CONVOLUTION_DIMENSION},
            bytes(randomFloats(random, 3 * CONVOLUTION_DIMENSION)))
        .addTensor(
            prefix + "ssm_dt.bias",
            GgufTensorType.F32,
            new long[] {VALUE_HEADS},
            bytes(new float[] {0.1f, -0.2f}))
        .addTensor(
            prefix + "ssm_a",
            GgufTensorType.F32,
            new long[] {VALUE_HEADS},
            bytes(new float[] {-0.5f, -0.75f}))
        .addTensor(
            prefix + "ssm_norm.weight",
            GgufTensorType.F32,
            new long[] {HEAD_DIMENSION},
            bytes(ones(HEAD_DIMENSION)));
    addRandomMatrix(builder, random, prefix + "ssm_out.weight", DIMENSION, VALUE_DIMENSION);
  }

  private static void addFullAttentionLayer(
      SyntheticGgufBuilder builder, Random random, int layer) {
    String prefix = "blk." + layer + ".";
    addNorms(builder, prefix);
    addRandomMatrix(builder, random, prefix + "attn_q.weight", 2 * DIMENSION, DIMENSION);
    addRandomMatrix(builder, random, prefix + "attn_k.weight", KEY_DIMENSION, DIMENSION);
    addRandomMatrix(builder, random, prefix + "attn_v.weight", KEY_DIMENSION, DIMENSION);
    addRandomMatrix(builder, random, prefix + "attn_output.weight", DIMENSION, DIMENSION);
    builder
        .addTensor(
            prefix + "attn_q_norm.weight",
            GgufTensorType.F32,
            new long[] {HEAD_DIMENSION},
            bytes(ones(HEAD_DIMENSION)))
        .addTensor(
            prefix + "attn_k_norm.weight",
            GgufTensorType.F32,
            new long[] {HEAD_DIMENSION},
            bytes(ones(HEAD_DIMENSION)));
  }

  private static void addNorms(SyntheticGgufBuilder builder, String prefix) {
    builder
        .addTensor(
            prefix + "attn_norm.weight",
            GgufTensorType.F32,
            new long[] {DIMENSION},
            bytes(ones(DIMENSION)))
        .addTensor(
            prefix + "post_attention_norm.weight",
            GgufTensorType.F32,
            new long[] {DIMENSION},
            bytes(ones(DIMENSION)));
  }

  private static void addRandomMatrix(
      SyntheticGgufBuilder builder, Random random, String name, int rows, int columns) {
    addMatrix(builder, name, rows, columns, randomFloats(random, rows * columns));
  }

  private static void addMatrix(
      SyntheticGgufBuilder builder, String name, int rows, int columns, float[] values) {
    builder.addTensor(name, GgufTensorType.F32, new long[] {columns, rows}, bytes(values));
  }

  private static float[] randomFloats(Random random, int length) {
    float[] values = new float[length];
    for (int index = 0; index < length; index++) {
      values[index] = (random.nextFloat() - 0.5f) * 0.2f;
    }
    return values;
  }

  private static float[] ones(int length) {
    float[] values = new float[length];
    java.util.Arrays.fill(values, 1.0f);
    return values;
  }

  private static byte[] bytes(float[] values) {
    ByteBuffer buffer =
        ByteBuffer.allocate(values.length * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    for (float value : values) {
      buffer.putFloat(value);
    }
    return buffer.array();
  }
}
