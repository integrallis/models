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

import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.stream.Collectors;

/**
 * One study arm, parsed from its command-line spec string.
 *
 * <p>Grammar: {@code member:<m>}, {@code fuse:<m1+m2..>:<rule>:<uniform|tuned|w1,w2..>}, {@code
 * vote|vote-consist|vote-conf:<m1+m2..>:<tieBreak>}, {@code sc:<m>:<k>}, {@code
 * rerank:<m1+m2..>:<uniform|tuned|w1,w2..>}.
 */
public record ArmSpec(
    Kind kind,
    List<String> members,
    FusionRule rule,
    WeightMode weightMode,
    double[] explicitWeights,
    String tieBreakMember,
    int samples) {

  /** Arm families. */
  public enum Kind {
    MEMBER("member"),
    FUSE("fuse"),
    VOTE("vote"),
    VOTE_CONSIST("vote-consist"),
    VOTE_CONF("vote-conf"),
    SELF_CONSISTENCY("sc"),
    RERANK("rerank");

    private final String id;

    Kind(String id) {
      this.id = id;
    }

    public String id() {
      return id;
    }

    boolean isVote() {
      return this == VOTE || this == VOTE_CONSIST || this == VOTE_CONF;
    }
  }

  /** Where fusion or rerank weights come from. */
  public enum WeightMode {
    NONE,
    UNIFORM,
    EXPLICIT,
    TUNED
  }

  /** Tie-break placeholder resolved to the frozen development-split best member. */
  public static final String BEST = "best";

  public ArmSpec {
    Objects.requireNonNull(kind, "kind");
    members = List.copyOf(members);
    explicitWeights = explicitWeights == null ? null : explicitWeights.clone();
  }

  @Override
  public double[] explicitWeights() {
    return explicitWeights == null ? null : explicitWeights.clone();
  }

  /** Parses a spec string, rejecting anything the grammar does not name exactly. */
  public static ArmSpec parse(String spec) {
    if (spec == null || spec.isBlank()) {
      throw new IllegalArgumentException("--arm is required");
    }
    String[] parts = spec.trim().split(":", -1);
    String head = parts[0].toLowerCase(Locale.ROOT);
    return switch (head) {
      case "member" -> {
        requireParts(spec, parts, 2);
        yield new ArmSpec(
            Kind.MEMBER, members(spec, parts[1], 1), null, WeightMode.NONE, null, null, 1);
      }
      case "fuse" -> {
        requireParts(spec, parts, 4);
        List<String> names = members(spec, parts[1], 2);
        FusionRule rule = FusionRule.parse(parts[2]);
        WeightChoice weights = weights(spec, parts[3], names.size());
        yield new ArmSpec(Kind.FUSE, names, rule, weights.mode(), weights.values(), null, 1);
      }
      case "vote", "vote-consist", "vote-conf" -> {
        requireParts(spec, parts, 3);
        List<String> names = members(spec, parts[1], 2);
        String tie = parts[2];
        if (!names.contains(tie) && !BEST.equals(tie)) {
          throw new IllegalArgumentException(
              "tie-break member must be one of the voters or 'best': " + spec);
        }
        Kind kind =
            switch (head) {
              case "vote" -> Kind.VOTE;
              case "vote-consist" -> Kind.VOTE_CONSIST;
              default -> Kind.VOTE_CONF;
            };
        yield new ArmSpec(kind, names, null, WeightMode.NONE, null, tie, 1);
      }
      case "sc" -> {
        requireParts(spec, parts, 3);
        int k;
        try {
          k = Integer.parseInt(parts[2]);
        } catch (NumberFormatException failure) {
          throw new IllegalArgumentException("sample count must be an integer: " + spec, failure);
        }
        if (k < 1) {
          throw new IllegalArgumentException("sample count must be positive: " + spec);
        }
        yield new ArmSpec(
            Kind.SELF_CONSISTENCY,
            members(spec, parts[1], 1),
            null,
            WeightMode.NONE,
            null,
            null,
            k);
      }
      case "rerank" -> {
        requireParts(spec, parts, 3);
        List<String> names = members(spec, parts[1], 2);
        WeightChoice weights = weights(spec, parts[2], names.size());
        yield new ArmSpec(
            Kind.RERANK, names, FusionRule.POE, weights.mode(), weights.values(), null, 1);
      }
      default -> throw new IllegalArgumentException("unknown arm kind: " + spec);
    };
  }

  /** Whether the arm must reference a frozen tuning/calibration record. */
  public boolean requiresFrozen() {
    return weightMode == WeightMode.TUNED || kind == Kind.VOTE_CONF || BEST.equals(tieBreakMember);
  }

  /** Resolves the tie-break member, reading {@code best} only from the frozen record. */
  public String resolveTieBreak(FrozenTuning frozen) {
    if (!BEST.equals(tieBreakMember)) {
      return tieBreakMember;
    }
    if (frozen == null) {
      throw new IllegalStateException("tie-break 'best' requires --frozen with its sha256");
    }
    String best = frozen.bestMember();
    if (!members.contains(best)) {
      throw new IllegalStateException(
          "frozen best member " + best + " is not a voter in " + canonical());
    }
    return best;
  }

  /** Voter order used for ties: the tie-break member first, then the arm's member order. */
  public List<String> priority(String tieBreak) {
    List<String> order = new java.util.ArrayList<>();
    order.add(tieBreak);
    members.stream().filter(m -> !m.equals(tieBreak)).forEach(order::add);
    return List.copyOf(order);
  }

  /** Resolves the weight vector, reading tuned weights only from the frozen record. */
  public double[] resolveWeights(FrozenTuning frozen) {
    return switch (weightMode) {
      case NONE -> {
        double[] single = new double[members.size()];
        Arrays.fill(single, 1.0 / members.size());
        yield single;
      }
      case UNIFORM -> {
        double[] uniform = new double[members.size()];
        Arrays.fill(uniform, 1.0 / members.size());
        yield uniform;
      }
      case EXPLICIT -> explicitWeights.clone();
      case TUNED -> {
        if (frozen == null) {
          throw new IllegalStateException(
              "arm " + canonical() + " uses tuned weights and requires --frozen with its sha256");
        }
        yield frozen.weightsFor(members, rule);
      }
    };
  }

  /** The spec string in canonical form. */
  public String canonical() {
    String names = String.join("+", members);
    return switch (kind) {
      case MEMBER -> "member:" + names;
      case FUSE -> "fuse:" + names + ":" + rule.id() + ":" + weightText();
      case VOTE, VOTE_CONSIST, VOTE_CONF -> kind.id() + ":" + names + ":" + tieBreakMember;
      case SELF_CONSISTENCY -> "sc:" + names + ":" + samples;
      case RERANK -> "rerank:" + names + ":" + weightText();
    };
  }

  private String weightText() {
    return switch (weightMode) {
      case UNIFORM -> "uniform";
      case TUNED -> "tuned";
      case EXPLICIT ->
          Arrays.stream(explicitWeights)
              .mapToObj(Double::toString)
              .collect(Collectors.joining(","));
      case NONE -> "";
    };
  }

  private record WeightChoice(WeightMode mode, double[] values) {}

  private static WeightChoice weights(String spec, String text, int count) {
    if ("uniform".equals(text)) {
      return new WeightChoice(WeightMode.UNIFORM, null);
    }
    if ("tuned".equals(text)) {
      return new WeightChoice(WeightMode.TUNED, null);
    }
    String[] tokens = text.split(",", -1);
    if (tokens.length != count) {
      throw new IllegalArgumentException(
          "expected " + count + " weights, uniform or tuned: " + spec);
    }
    double[] values = new double[count];
    for (int i = 0; i < count; i++) {
      try {
        values[i] = Double.parseDouble(tokens[i]);
      } catch (NumberFormatException failure) {
        throw new IllegalArgumentException("weights must be numbers: " + spec, failure);
      }
    }
    FusionMath.validateWeights(values);
    return new WeightChoice(WeightMode.EXPLICIT, values);
  }

  private static List<String> members(String spec, String text, int minimum) {
    if (text.isBlank()) {
      throw new IllegalArgumentException("arm names no members: " + spec);
    }
    List<String> names = List.of(text.split("\\+", -1));
    if (names.stream().anyMatch(String::isBlank)) {
      throw new IllegalArgumentException("blank member name: " + spec);
    }
    if (new HashSet<>(names).size() != names.size()) {
      throw new IllegalArgumentException("duplicate member name: " + spec);
    }
    if (minimum == 1 && names.size() != 1) {
      throw new IllegalArgumentException("arm takes exactly one member: " + spec);
    }
    if (names.size() < minimum) {
      throw new IllegalArgumentException("arm needs at least " + minimum + " members: " + spec);
    }
    return names;
  }

  private static void requireParts(String spec, String[] parts, int count) {
    if (parts.length != count) {
      throw new IllegalArgumentException("malformed arm spec: " + spec);
    }
  }
}
