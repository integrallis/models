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

import com.integrallis.models.api.ModelPrompt;
import com.integrallis.models.api.Tokenizer;
import java.io.ByteArrayOutputStream;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.CharBuffer;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.CodingErrorAction;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/** SentencePiece tokenizer stored in the raw tensor of a `.cact` artifact. */
public final class CactTokenizer implements Tokenizer {

  private static final int TOKENIZER_HEADER_BYTES = 24;
  private static final char META_SPACE = '\u2581';
  private static final ValueLayout.OfInt LE_INT =
      ValueLayout.JAVA_INT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
  private static final ValueLayout.OfFloat LE_FLOAT =
      ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

  private final String[] pieces;
  private final float[] scores;
  private final PieceType[] types;
  private final Map<String, Integer> pieceIds;
  private final List<Marker> markers;
  private final int[] byteIds;
  private final int[] byteValues;
  private final int padToken;
  private final int eosToken;
  private final int bosToken;
  private final int unknownToken;
  private final boolean addDummyPrefix;
  private final boolean byteFallback;

  private CactTokenizer(
      String[] pieces,
      float[] scores,
      PieceType[] types,
      Map<String, Integer> pieceIds,
      List<Marker> markers,
      int[] byteIds,
      int[] byteValues,
      int padToken,
      int eosToken,
      int bosToken,
      int unknownToken,
      boolean addDummyPrefix,
      boolean byteFallback) {
    this.pieces = pieces;
    this.scores = scores;
    this.types = types;
    this.pieceIds = Map.copyOf(pieceIds);
    this.markers = List.copyOf(markers);
    this.byteIds = byteIds;
    this.byteValues = byteValues;
    this.padToken = padToken;
    this.eosToken = eosToken;
    this.bosToken = bosToken;
    this.unknownToken = unknownToken;
    this.addDummyPrefix = addDummyPrefix;
    this.byteFallback = byteFallback;
  }

  /** Loads the single embedded tokenizer from a parsed `.cact` artifact. */
  public static CactTokenizer from(CactFile file) {
    Objects.requireNonNull(file, "file");
    List<CactTensorInfo> rawTensors =
        file.tensorInfos().stream().filter(info -> info.type() == CactTensorType.RAW).toList();
    if (rawTensors.size() != 1) {
      throw new MalformedCactException(
          "expected exactly one raw tokenizer tensor; got " + rawTensors.size());
    }
    CactTensorInfo tokenizer = rawTensors.getFirst();
    return parse(file.tensor(tokenizer.index()).data(), file.header().vocabularySize());
  }

  static CactTokenizer parse(MemorySegment data, int expectedVocabularySize) {
    Objects.requireNonNull(data, "data");
    if (data.byteSize() < TOKENIZER_HEADER_BYTES) {
      throw new MalformedCactException(
          "tokenizer header requires " + TOKENIZER_HEADER_BYTES + " bytes; got " + data.byteSize());
    }
    try {
      Cursor cursor = new Cursor(data);
      int pieceCount = positiveInt(cursor.readU32(), "tokenizer piece count");
      if (pieceCount != expectedVocabularySize) {
        throw new MalformedCactException(
            "tokenizer vocabulary has "
                + pieceCount
                + " pieces; model header declares "
                + expectedVocabularySize);
      }
      int pad = tokenId(cursor.readU32(), pieceCount, "pad token");
      int eos = tokenId(cursor.readU32(), pieceCount, "EOS token");
      int bos = tokenId(cursor.readU32(), pieceCount, "BOS token");
      int unknown = tokenId(cursor.readU32(), pieceCount, "unknown token");
      boolean addDummy = cursor.readFlag("add-dummy-prefix");
      boolean fallback = cursor.readFlag("byte-fallback");
      if (cursor.readU16() != 0) {
        throw new MalformedCactException("tokenizer header padding must be zero");
      }

      String[] pieces = new String[pieceCount];
      float[] scores = new float[pieceCount];
      PieceType[] types = new PieceType[pieceCount];
      Map<String, Integer> pieceIds = new HashMap<>(pieceCount * 2);
      List<Marker> markers = new ArrayList<>();
      int[] byteIds = new int[256];
      int[] byteValues = new int[pieceCount];
      Arrays.fill(byteIds, -1);
      Arrays.fill(byteValues, -1);
      for (int id = 0; id < pieceCount; id++) {
        float score = cursor.readFloat();
        if (!Float.isFinite(score)) {
          throw new MalformedCactException("tokenizer piece " + id + " score must be finite");
        }
        PieceType type = PieceType.fromId(cursor.readU8());
        int byteLength = cursor.readU16();
        if (byteLength == 0) {
          throw new MalformedCactException("tokenizer piece " + id + " must not be empty");
        }
        String piece = cursor.readUtf8(byteLength, id);
        if (pieceIds.putIfAbsent(piece, id) != null) {
          throw new MalformedCactException("duplicate tokenizer piece " + piece);
        }
        pieces[id] = piece;
        scores[id] = score;
        types[id] = type;
        if (type == PieceType.USER_DEFINED) {
          markers.add(new Marker(piece, id));
        } else if (type == PieceType.BYTE) {
          int value = parseBytePiece(piece, id);
          if (byteIds[value] >= 0) {
            throw new MalformedCactException("duplicate tokenizer byte piece " + piece);
          }
          byteIds[value] = id;
          byteValues[id] = value;
        }
      }
      if (cursor.offset != data.byteSize()) {
        throw new MalformedCactException(
            "tokenizer has " + (data.byteSize() - cursor.offset) + " trailing bytes");
      }
      requireType(types, pad, PieceType.CONTROL, "pad token");
      requireType(types, eos, PieceType.CONTROL, "EOS token");
      requireType(types, bos, PieceType.CONTROL, "BOS token");
      requireType(types, unknown, PieceType.UNKNOWN, "unknown token");
      markers.sort(Comparator.comparingInt((Marker marker) -> marker.text().length()).reversed());
      return new CactTokenizer(
          pieces,
          scores,
          types,
          pieceIds,
          markers,
          byteIds,
          byteValues,
          pad,
          eos,
          bos,
          unknown,
          addDummy,
          fallback);
    } catch (MalformedCactException malformed) {
      throw malformed;
    } catch (IndexOutOfBoundsException malformed) {
      throw new MalformedCactException("truncated tokenizer blob", malformed);
    }
  }

  @Override
  public int[] encode(String text) {
    return encode(ModelPrompt.text(Objects.requireNonNull(text, "text")));
  }

  @Override
  public int[] encode(ModelPrompt prompt) {
    Objects.requireNonNull(prompt, "prompt");
    if (prompt.isEmpty()) {
      return new int[0];
    }
    List<Integer> encoded = new ArrayList<>();
    StringBuilder ordinary = new StringBuilder();
    if (addDummyPrefix) {
      ordinary.append(META_SPACE);
    }
    for (ModelPrompt.Segment segment : prompt.segments()) {
      String normalized = segment.text().replace(' ', META_SPACE);
      if (segment.kind() == ModelPrompt.SegmentKind.CONTROL) {
        appendTrustedControl(encoded, ordinary, normalized);
      } else {
        ordinary.append(normalized);
      }
    }
    flushOrdinary(encoded, ordinary);
    return encoded.stream().mapToInt(Integer::intValue).toArray();
  }

  private void appendTrustedControl(List<Integer> encoded, StringBuilder ordinary, String control) {
    int position = 0;
    while (position < control.length()) {
      Marker next = null;
      int nextIndex = -1;
      for (Marker marker : markers) {
        int candidate = control.indexOf(marker.text(), position);
        if (candidate >= 0 && (nextIndex < 0 || candidate < nextIndex)) {
          next = marker;
          nextIndex = candidate;
        }
      }
      if (next == null) {
        ordinary.append(control, position, control.length());
        return;
      }
      ordinary.append(control, position, nextIndex);
      flushOrdinary(encoded, ordinary);
      encoded.add(next.id());
      position = nextIndex + next.text().length();
    }
  }

  private void flushOrdinary(List<Integer> encoded, StringBuilder ordinary) {
    if (ordinary.isEmpty()) {
      return;
    }
    List<String> symbols =
        new ArrayList<>(
            ordinary
                .toString()
                .codePoints()
                .mapToObj(codePoint -> new String(Character.toChars(codePoint)))
                .toList());
    while (symbols.size() > 1) {
      float bestScore = Float.NEGATIVE_INFINITY;
      int bestIndex = -1;
      for (int index = 0; index + 1 < symbols.size(); index++) {
        int token = ordinaryToken(symbols.get(index) + symbols.get(index + 1));
        if (token >= 0 && (bestIndex < 0 || scores[token] > bestScore)) {
          bestScore = scores[token];
          bestIndex = index;
        }
      }
      if (bestIndex < 0) {
        break;
      }
      symbols.set(bestIndex, symbols.get(bestIndex) + symbols.get(bestIndex + 1));
      symbols.remove(bestIndex + 1);
    }
    for (String symbol : symbols) {
      int token = ordinaryToken(symbol);
      if (token >= 0) {
        encoded.add(token);
      } else {
        appendFallback(encoded, symbol);
      }
    }
    ordinary.setLength(0);
  }

  private int ordinaryToken(String piece) {
    Integer token = pieceIds.get(piece);
    return token != null && types[token] == PieceType.NORMAL ? token : -1;
  }

  private void appendFallback(List<Integer> encoded, String symbol) {
    if (!byteFallback) {
      encoded.add(unknownToken);
      return;
    }
    for (byte value : symbol.getBytes(StandardCharsets.UTF_8)) {
      int token = byteIds[Byte.toUnsignedInt(value)];
      encoded.add(token >= 0 ? token : unknownToken);
    }
  }

  @Override
  public String decode(int[] tokens) {
    Objects.requireNonNull(tokens, "tokens");
    ByteArrayOutputStream bytes = new ByteArrayOutputStream();
    for (int token : tokens) {
      appendDecoded(bytes, token);
    }
    String decoded = bytes.toString(StandardCharsets.UTF_8).replace(META_SPACE, ' ');
    return addDummyPrefix && decoded.startsWith(" ") ? decoded.substring(1) : decoded;
  }

  @Override
  public String decode(int token) {
    ByteArrayOutputStream bytes = new ByteArrayOutputStream();
    appendDecoded(bytes, token);
    return bytes.toString(StandardCharsets.UTF_8).replace(META_SPACE, ' ');
  }

  private void appendDecoded(ByteArrayOutputStream destination, int token) {
    if (token < 0
        || token >= pieces.length
        || types[token] == PieceType.CONTROL
        || types[token] == PieceType.UNKNOWN) {
      return;
    }
    if (types[token] == PieceType.BYTE) {
      destination.write(byteValues[token]);
    } else {
      destination.writeBytes(pieces[token].getBytes(StandardCharsets.UTF_8));
    }
  }

  @Override
  public int tokenId(String text) {
    if (text == null) {
      return -1;
    }
    return pieceIds.getOrDefault(text, -1);
  }

  @Override
  public int vocabSize() {
    return pieces.length;
  }

  /** Returns the serialized padding token ID. */
  public int padToken() {
    return padToken;
  }

  @Override
  public int bosToken() {
    return bosToken;
  }

  @Override
  public int eosToken() {
    return eosToken;
  }

  /** Returns the serialized unknown-token ID. */
  public int unknownToken() {
    return unknownToken;
  }

  /** Returns whether encoding inserts the SentencePiece dummy prefix. */
  public boolean addDummyPrefix() {
    return addDummyPrefix;
  }

  /** Returns whether unknown symbols fall back to their UTF-8 bytes. */
  public boolean byteFallback() {
    return byteFallback;
  }

  private static int tokenId(long value, int vocabularySize, String name) {
    if (value >= vocabularySize) {
      throw new MalformedCactException(name + " " + value + " is outside the vocabulary");
    }
    return (int) value;
  }

  private static int positiveInt(long value, String name) {
    if (value == 0 || value > Integer.MAX_VALUE) {
      throw new MalformedCactException(name + " must be within the positive Java int range");
    }
    return (int) value;
  }

  private static void requireType(PieceType[] types, int token, PieceType expected, String name) {
    if (types[token] != expected) {
      throw new MalformedCactException(name + " must have piece type " + expected.serializedName);
    }
  }

  private static int parseBytePiece(String piece, int id) {
    if (piece.length() != 6 || !piece.startsWith("<0x") || piece.charAt(5) != '>') {
      throw new MalformedCactException("tokenizer byte piece " + id + " has invalid text " + piece);
    }
    try {
      return Integer.parseInt(piece.substring(3, 5), 16);
    } catch (NumberFormatException invalid) {
      throw new MalformedCactException(
          "tokenizer byte piece " + id + " has invalid text " + piece, invalid);
    }
  }

  private enum PieceType {
    NORMAL(0, "normal"),
    UNKNOWN(1, "unknown"),
    CONTROL(2, "control"),
    USER_DEFINED(3, "user-defined"),
    BYTE(4, "byte");

    private final int id;
    private final String serializedName;

    PieceType(int id, String serializedName) {
      this.id = id;
      this.serializedName = serializedName;
    }

    private static PieceType fromId(int id) {
      for (PieceType type : values()) {
        if (type.id == id) {
          return type;
        }
      }
      throw new MalformedCactException("tokenizer piece type " + id + " is unsupported");
    }
  }

  private record Marker(String text, int id) {}

  private static final class Cursor {
    private final MemorySegment data;
    private long offset;

    private Cursor(MemorySegment data) {
      this.data = data;
    }

    private int readU8() {
      int value = Byte.toUnsignedInt(data.get(ValueLayout.JAVA_BYTE, offset));
      offset++;
      return value;
    }

    private int readU16() {
      int low = readU8();
      return low | (readU8() << Byte.SIZE);
    }

    private long readU32() {
      long value = Integer.toUnsignedLong(data.get(LE_INT, offset));
      offset += Integer.BYTES;
      return value;
    }

    private float readFloat() {
      float value = data.get(LE_FLOAT, offset);
      offset += Float.BYTES;
      return value;
    }

    private boolean readFlag(String name) {
      int value = readU8();
      if (value != 0 && value != 1) {
        throw new MalformedCactException("tokenizer " + name + " flag must be zero or one");
      }
      return value == 1;
    }

    private String readUtf8(int length, int piece) {
      MemorySegment bytes = data.asSlice(offset, length);
      offset += length;
      try {
        CharBuffer decoded =
            StandardCharsets.UTF_8
                .newDecoder()
                .onMalformedInput(CodingErrorAction.REPORT)
                .onUnmappableCharacter(CodingErrorAction.REPORT)
                .decode(ByteBuffer.wrap(bytes.toArray(ValueLayout.JAVA_BYTE)));
        return decoded.toString();
      } catch (CharacterCodingException invalid) {
        throw new MalformedCactException(
            "tokenizer piece " + piece + " is not valid UTF-8", invalid);
      }
    }
  }
}
