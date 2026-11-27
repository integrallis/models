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
package com.integrallis.models.backend.purejava;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HexFormat;
import java.util.List;
import java.util.Objects;

/**
 * What a multimodal prefix holds, position by position: runs of token ids and images, in order.
 *
 * <p>An image is identified by the SHA-256 of its encoded bytes and the number of positions its
 * projected rows occupy. Placeholder tokens are not identity: two prompts with the same text and
 * different images differ here, and a prefix built from one is refused for the other (#425). The
 * projector, its preprocessing and the image token layout are bound separately, by the backend's
 * fingerprint.
 *
 * <p>Equality is over the canonical form, in which adjacent token runs are merged, so content built
 * from {@code text(1, 2).text(3)} equals {@code text(1, 2, 3)}.
 */
public final class MultimodalContent {

  /** One run of positions. */
  public sealed interface Segment permits Tokens, Image {
    /** Positions this run occupies. */
    int positions();
  }

  /**
   * A run of token ids.
   *
   * @param tokens the ids, in order
   */
  public record Tokens(int[] tokens) implements Segment {
    public Tokens {
      tokens = tokens.clone();
      if (tokens.length == 0) throw new IllegalArgumentException("empty token run");
    }

    @Override
    public int[] tokens() {
      return tokens.clone();
    }

    @Override
    public int positions() {
      return tokens.length;
    }

    @Override
    public boolean equals(Object other) {
      return other instanceof Tokens that && Arrays.equals(tokens, that.tokens);
    }

    @Override
    public int hashCode() {
      return Arrays.hashCode(tokens);
    }

    @Override
    public String toString() {
      return "Tokens" + Arrays.toString(tokens);
    }
  }

  /**
   * An image.
   *
   * @param sha256 SHA-256 of the encoded image bytes
   * @param rows positions its projected rows occupy
   */
  public record Image(byte[] sha256, int rows) implements Segment {
    public Image {
      sha256 = sha256.clone();
      if (sha256.length != 32) throw new IllegalArgumentException("image digest must be 32 bytes");
      if (rows <= 0) throw new IllegalArgumentException("image rows must be positive");
    }

    @Override
    public byte[] sha256() {
      return sha256.clone();
    }

    @Override
    public int positions() {
      return rows;
    }

    @Override
    public boolean equals(Object other) {
      return other instanceof Image that && rows == that.rows && Arrays.equals(sha256, that.sha256);
    }

    @Override
    public int hashCode() {
      return 31 * Arrays.hashCode(sha256) + rows;
    }

    @Override
    public String toString() {
      return "Image[" + HexFormat.of().formatHex(sha256, 0, 6) + "..., rows=" + rows + "]";
    }
  }

  private static final byte TOKENS = 1;
  private static final byte IMAGE = 2;

  private final List<Segment> segments;
  private final int positions;

  private MultimodalContent(List<Segment> segments) {
    this.segments = List.copyOf(segments);
    int total = 0;
    for (Segment segment : this.segments) total = Math.addExact(total, segment.positions());
    this.positions = total;
  }

  /** Empty content, to be extended. */
  public static Builder builder() {
    return new Builder();
  }

  /** A builder that starts from {@code start}, to extend it. */
  public static Builder builder(MultimodalContent start) {
    Builder builder = new Builder();
    for (Segment segment : start.segments) builder.add(segment);
    return builder;
  }

  /** The runs, canonical: no two adjacent token runs. */
  public List<Segment> segments() {
    return segments;
  }

  /** Total positions. */
  public int positions() {
    return positions;
  }

  /** The first {@code count} positions; refused if that would cut through an image. */
  public MultimodalContent prefix(int count) {
    if (count < 0 || count > positions) throw new IllegalArgumentException("invalid prefix length");
    Builder builder = new Builder();
    int remaining = count;
    for (Segment segment : segments) {
      if (remaining == 0) break;
      if (segment instanceof Tokens run) {
        int take = Math.min(remaining, run.positions());
        builder.tokens(Arrays.copyOf(run.tokens, take));
        remaining -= take;
      } else {
        if (remaining < segment.positions()) {
          throw new IllegalArgumentException("a prefix may not end inside an image");
        }
        builder.add(segment);
        remaining -= segment.positions();
      }
    }
    return builder.build();
  }

  /** A description of the first difference from {@code other}, or {@code null} if equal. */
  public String firstDifference(MultimodalContent other) {
    Objects.requireNonNull(other, "other");
    int position = 0;
    int shared = Math.min(segments.size(), other.segments.size());
    for (int index = 0; index < shared; index++) {
      Segment mine = segments.get(index);
      Segment theirs = other.segments.get(index);
      if (!mine.equals(theirs)) {
        return "segment " + index + " at position " + position + ": " + mine + " vs " + theirs;
      }
      position += mine.positions();
    }
    if (segments.size() != other.segments.size()) {
      return "segment count " + segments.size() + " vs " + other.segments.size();
    }
    return null;
  }

  @Override
  public boolean equals(Object other) {
    return other instanceof MultimodalContent that && segments.equals(that.segments);
  }

  @Override
  public int hashCode() {
    return segments.hashCode();
  }

  @Override
  public String toString() {
    return "MultimodalContent" + segments;
  }

  /** Writes the canonical form: segment count, then each run. */
  void writeTo(DataOutput output) throws IOException {
    output.writeInt(segments.size());
    for (Segment segment : segments) {
      if (segment instanceof Tokens run) {
        output.writeByte(TOKENS);
        output.writeInt(run.tokens.length);
        for (int token : run.tokens) output.writeInt(token);
      } else {
        Image image = (Image) segment;
        output.writeByte(IMAGE);
        output.writeInt(image.rows);
        output.write(image.sha256);
      }
    }
  }

  /** Exact bytes {@link #writeTo} produces. */
  long serializedBytes() {
    long bytes = Integer.BYTES;
    for (Segment segment : segments) {
      bytes += 1 + Integer.BYTES;
      bytes += segment instanceof Tokens run ? (long) Integer.BYTES * run.tokens.length : 32;
    }
    return bytes;
  }

  /**
   * Reads content written by {@link #writeTo}, refusing anything outside the given bounds before it
   * allocates for it.
   *
   * @param maxPositions most positions the receiver can hold
   * @param vocabSize token ids must lie in {@code [0, vocabSize)}
   * @param imageRows positions every image must occupy on the receiver
   * @param maxBytes most bytes this content may occupy on the wire
   */
  static MultimodalContent readFrom(
      DataInput input, int maxPositions, int vocabSize, int imageRows, long maxBytes)
      throws IOException {
    long budget = maxBytes - Integer.BYTES;
    int count = input.readInt();
    if (count <= 0 || count > maxPositions || budget < 0) {
      throw new IOException("invalid multimodal segment count " + count);
    }
    Builder builder = new Builder();
    int positions = 0;
    for (int index = 0; index < count; index++) {
      budget -= 1 + Integer.BYTES;
      if (budget < 0) throw new IOException("multimodal content exceeds import budget");
      byte kind = input.readByte();
      int length = input.readInt();
      if (length <= 0 || length > maxPositions - positions) {
        throw new IOException("invalid multimodal segment length " + length);
      }
      if (kind == TOKENS) {
        budget -= (long) Integer.BYTES * length;
        if (budget < 0) throw new IOException("multimodal content exceeds import budget");
        int[] tokens = new int[length];
        for (int i = 0; i < length; i++) {
          tokens[i] = input.readInt();
          if (tokens[i] < 0 || tokens[i] >= vocabSize)
            throw new IOException("invalid prefix token");
        }
        builder.tokens(tokens);
      } else if (kind == IMAGE) {
        if (length != imageRows) {
          throw new IOException(
              "image occupies " + length + " positions; this projector uses " + imageRows);
        }
        budget -= 32;
        if (budget < 0) throw new IOException("multimodal content exceeds import budget");
        byte[] digest = new byte[32];
        input.readFully(digest);
        builder.add(new Image(digest, length));
      } else {
        throw new IOException("unknown multimodal segment kind " + kind);
      }
      positions += length;
    }
    return builder.build();
  }

  /** Accumulates segments in order, merging adjacent token runs. */
  public static final class Builder {
    private final List<Segment> segments = new ArrayList<>();

    private Builder() {}

    /** Appends a run of token ids. */
    public Builder tokens(int... tokens) {
      Objects.requireNonNull(tokens, "tokens");
      if (tokens.length == 0) return this;
      return add(new Tokens(tokens));
    }

    /** Appends an image by its encoded bytes, occupying {@code rows} positions. */
    public Builder image(byte[] encodedImage, int rows) {
      Objects.requireNonNull(encodedImage, "encodedImage");
      return add(new Image(sha256(encodedImage), rows));
    }

    Builder add(Segment segment) {
      if (segment instanceof Tokens run
          && !segments.isEmpty()
          && segments.getLast() instanceof Tokens previous) {
        int[] merged = Arrays.copyOf(previous.tokens, previous.tokens.length + run.tokens.length);
        System.arraycopy(run.tokens, 0, merged, previous.tokens.length, run.tokens.length);
        segments.set(segments.size() - 1, new Tokens(merged));
      } else {
        segments.add(segment);
      }
      return this;
    }

    /** The content so far. */
    public MultimodalContent build() {
      return new MultimodalContent(segments);
    }
  }

  static byte[] sha256(byte[] bytes) {
    try {
      return MessageDigest.getInstance("SHA-256").digest(bytes);
    } catch (NoSuchAlgorithmException impossible) {
      throw new IllegalStateException(impossible);
    }
  }
}
