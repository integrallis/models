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

import com.integrallis.models.api.BackendDiagnostics;
import com.integrallis.models.api.ImageInferenceBackend;
import com.integrallis.models.api.InferenceSession;
import com.integrallis.models.api.ModelMetadata;
import com.integrallis.models.api.PortablePrefixCapabilities;
import com.integrallis.models.api.SharedInferencePrefix;
import com.integrallis.models.api.Tokenizer;
import com.integrallis.models.backend.purejava.cache.PrefixFingerprint;
import com.integrallis.models.backend.purejava.plan.RuntimeFingerprint;
import com.integrallis.models.backend.purejava.spi.GgufMatrixKernelProvider;
import com.integrallis.models.backend.purejava.vision.ClipImagePreprocessor;
import com.integrallis.models.backend.purejava.vision.LlavaVisionProjector;
import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.HexFormat;
import java.util.Objects;
import java.util.ServiceLoader;

/**
 * Two-file LLaVA 1.5 inference, sharing the shipping Llama decoder and its KV cache.
 *
 * <p>Besides the single default sequence of {@link ImageInferenceBackend}, it offers sessions whose
 * content (token runs and images) is recorded, frozen into a {@link LlavaPrefix}, forked, and moved
 * between independently loaded backends (#425). An import must be given the content the receiver
 * itself expects ({@link #readPrefix(DataInput, long, MultimodalContent)}) and is refused on any
 * difference, an image included: placeholder tokens alone are not identity. The projector file, its
 * preprocessing code and the image token layout are bound by {@link #portablePrefixFingerprint()}.
 *
 * <p>This class deliberately does not implement the generic {@code PortablePrefixInferenceBackend}:
 * its {@code readPrefix} has no way to be told which image the receiver holds, so a prefix built
 * from one image could be continued under another.
 */
public final class LlavaBackend implements ImageInferenceBackend {
  private static final int PREFIX_MAGIC = 0x4c4c5631; // "LLV1"

  private final PureJavaBackend text;
  private final LlavaVisionProjector vision;
  private final Path projectorPath;
  private boolean closed;
  private int position;
  private String portableFingerprint;

  private LlavaBackend(PureJavaBackend text, LlavaVisionProjector vision, Path projectorPath) {
    this.text = text;
    this.vision = vision;
    this.projectorPath = projectorPath;
  }

  /** Loads a matched LLaVA text GGUF and converted 23-block CLIP MLP mmproj. */
  public static LlavaBackend load(Path textModel, Path mmproj) {
    LlavaVisionProjector vision = LlavaVisionProjector.load(mmproj);
    PureJavaBackend text = null;
    try {
      text = decoder(textModel);
      if (!text.supportsEmbeddingInput()
          || text.embeddingInputDimension() != vision.embeddingDimension()) {
        throw new IllegalArgumentException(
            "Text model and projector architectures/dimensions do not match");
      }
      if (text.contextCapacity() <= vision.tokenCount())
        throw new IllegalArgumentException("Context must fit image patches and text");
      return new LlavaBackend(text, vision, mmproj);
    } catch (RuntimeException | Error e) {
      var failedText = text;
      try (vision;
          failedText) {
        throw e;
      }
    }
  }

  private static PureJavaBackend decoder(Path model) {
    int threads = RuntimeFingerprint.capture().ggufThreads();
    for (GgufMatrixKernelProvider provider : ServiceLoader.load(GgufMatrixKernelProvider.class)) {
      var available = provider.open(threads);
      if (available.isPresent()) {
        var kernel = available.orElseThrow();
        com.integrallis.models.backend.purejava.spi.BatchedCausalAttentionKernel attention;
        try {
          attention = provider.openAttention(threads, kernel);
        } catch (RuntimeException | Error failure) {
          try (kernel) {
            throw failure;
          }
        }
        return PureJavaBackend.load(model, kernel, attention);
      }
    }
    return PureJavaBackend.load(model);
  }

  @Override
  public String name() {
    checkOpen();
    return "pure-java-llava";
  }

  @Override
  public ModelMetadata metadata() {
    checkOpen();
    return text.metadata();
  }

  @Override
  public BackendDiagnostics diagnostics() {
    checkOpen();
    var decoder = text.diagnostics();
    var environment = new java.util.LinkedHashMap<>(decoder.environment());
    environment.put("vision-projections", vision.implementation());
    environment.put("decoder-attention", text.attentionKernelImplementation());
    return new BackendDiagnostics(
        name(), "llava-v1/" + decoder.planVersion(), environment, decoder.optimizations());
  }

  @Override
  public Tokenizer tokenizer() {
    checkOpen();
    return text.tokenizer();
  }

  @Override
  public int contextCapacity() {
    checkOpen();
    return text.contextCapacity();
  }

  @Override
  public int imageTokenCount() {
    checkOpen();
    return vision.tokenCount();
  }

  @Override
  public float[] prefillImage(byte[] encodedImage, int startPosition) {
    requirePosition(startPosition, imageTokenCount());
    float[] result = text.prefillEmbeddings(vision.encode(encodedImage), startPosition);
    position += imageTokenCount();
    return result;
  }

  @Override
  public float[] forward(int token, int startPosition) {
    requirePosition(startPosition, 1);
    float[] result = text.forward(token, startPosition);
    position++;
    return result;
  }

  @Override
  public float[] prefill(int[] tokens, int startPosition) {
    requirePosition(startPosition, tokens.length);
    for (int token : tokens) {
      if (token < 0 || token >= metadata().vocabSize())
        throw new IllegalArgumentException("Invalid token");
    }
    float[] result = text.prefill(tokens, startPosition);
    position += tokens.length;
    return result;
  }

  @Override
  public void reset() {
    checkOpen();
    text.reset();
    position = 0;
  }

  @Override
  public void close() {
    if (!closed) {
      closed = true;
      try {
        text.close();
      } finally {
        vision.close();
      }
    }
  }

  // --- sessions, prefixes and their transfer (#425) --------------------------------------------

  /** A session whose content is recorded as it is read. */
  private static final class LlavaSession implements InferenceSession {
    private final LlavaBackend owner;
    private final InferenceSession text;
    private MultimodalContent content;

    private LlavaSession(LlavaBackend owner, InferenceSession text, MultimodalContent content) {
      this.owner = owner;
      this.text = text;
      this.content = content;
    }

    @Override
    public int checkpoint() {
      return text.checkpoint();
    }

    @Override
    public boolean isClosed() {
      return text.isClosed();
    }

    @Override
    public java.util.OptionalLong allocatedStateBytes() {
      return text.allocatedStateBytes();
    }

    @Override
    public void close() {
      text.close();
    }
  }

  /** A frozen session: the decoder's immutable prefix and the content it holds. */
  public static final class LlavaPrefix implements SharedInferencePrefix {
    private final LlavaBackend owner;
    private final SharedInferencePrefix text;
    private final MultimodalContent content;

    private LlavaPrefix(LlavaBackend owner, SharedInferencePrefix text, MultimodalContent content) {
      this.owner = owner;
      this.text = text;
      this.content = content;
      if (text.checkpoint() != content.positions()) {
        throw new IllegalStateException(
            "prefix holds " + text.checkpoint() + " positions but records " + content.positions());
      }
    }

    /** What the prefix holds, position by position. */
    public MultimodalContent content() {
      return content;
    }

    @Override
    public int checkpoint() {
      return text.checkpoint();
    }

    @Override
    public long sharedBytes() {
      return text.sharedBytes();
    }
  }

  /** Opens an empty session. */
  public InferenceSession openSession() {
    checkOpen();
    return new LlavaSession(this, text.openSession(), MultimodalContent.builder().build());
  }

  /** Reads tokens into a session and records them. */
  public float[] prefill(InferenceSession session, int[] tokens, int startPosition) {
    LlavaSession owned = requireSession(session);
    validateTokens(tokens);
    float[] result = text.prefill(owned.text, tokens, startPosition);
    owned.content = MultimodalContent.builder(owned.content).tokens(tokens).build();
    return result;
  }

  /** Reads one encoded image into a session and records its digest. */
  public float[] prefillImage(InferenceSession session, byte[] encodedImage, int startPosition) {
    LlavaSession owned = requireSession(session);
    Objects.requireNonNull(encodedImage, "encodedImage");
    float[] result = text.prefillEmbeddings(owned.text, vision.encode(encodedImage), startPosition);
    owned.content =
        MultimodalContent.builder(owned.content).image(encodedImage, imageTokenCount()).build();
    return result;
  }

  /** Runs one token in a session and records it. */
  public float[] forward(InferenceSession session, int token, int position) {
    LlavaSession owned = requireSession(session);
    validateTokens(new int[] {token});
    float[] result = text.forward(owned.text, token, position);
    owned.content = MultimodalContent.builder(owned.content).tokens(token).build();
    return result;
  }

  /** Freezes a session into an immutable prefix; the session is consumed. */
  public LlavaPrefix freezePrefix(InferenceSession session) {
    LlavaSession owned = requireSession(session);
    MultimodalContent content = owned.content;
    return new LlavaPrefix(this, text.freezePrefix(owned.text), content);
  }

  /** Opens an independent session continuing from {@code prefix}. */
  public InferenceSession fork(LlavaPrefix prefix) {
    LlavaPrefix owned = requirePrefix(prefix);
    return new LlavaSession(this, text.fork(owned.text), owned.content);
  }

  /** Whether prefixes of this configuration can be transferred, and if not, why. */
  public PortablePrefixCapabilities portablePrefixCapabilities() {
    checkOpen();
    return text.portablePrefixCapabilities();
  }

  /**
   * Digest binding the text model and its KV execution identity, the projector file, the image
   * preprocessing and projection code, and the image token layout.
   */
  public String portablePrefixFingerprint() throws IOException {
    checkOpen();
    if (!text.supportsPortablePrefixes()) {
      throw new UnsupportedOperationException(
          "prefixes cannot be transferred: " + text.portablePrefixCapabilities().status());
    }
    if (portableFingerprint == null) {
      String settings =
          "llava-v1;imageRows="
              + vision.tokenCount()
              + ";imageSize="
              + vision.imageSize()
              + ";embedding="
              + vision.embeddingDimension()
              + ";projection="
              + vision.implementation()
              + ";text="
              + text.portablePrefixFingerprint();
      portableFingerprint =
          PrefixFingerprint.of(
              projectorPath,
              settings,
              LlavaVisionProjector.class,
              ClipImagePreprocessor.class,
              LlavaBackend.class);
    }
    return portableFingerprint;
  }

  /** Exact bytes {@link #writePrefix} writes for {@code prefix}. */
  public long portablePrefixBytes(LlavaPrefix prefix) throws IOException {
    LlavaPrefix owned = requirePrefix(prefix);
    portablePrefixFingerprint();
    return Math.addExact(
        Integer.BYTES + 32L + owned.content.serializedBytes(),
        text.portablePrefixBytes(owned.content.positions()));
  }

  /** Writes {@code prefix}: format, identity, content, then the decoder's state. */
  public void writePrefix(LlavaPrefix prefix, DataOutput output) throws IOException {
    LlavaPrefix owned = requirePrefix(prefix);
    Objects.requireNonNull(output, "output");
    output.writeInt(PREFIX_MAGIC);
    output.write(HexFormat.of().parseHex(portablePrefixFingerprint()));
    owned.content.writeTo(output);
    text.writePrefix(owned.text, owned.content.positions(), output);
  }

  /**
   * Reads a prefix written by {@link #writePrefix}, provided it holds exactly {@code expected}: the
   * tokens and images the receiver itself is about to continue from. A different identity, a
   * different image, different tokens or more bytes than {@code maxBytes} are refused before any
   * decoder state is allocated, and leave this backend as it was.
   */
  public LlavaPrefix readPrefix(DataInput input, long maxBytes, MultimodalContent expected)
      throws IOException {
    checkOpen();
    Objects.requireNonNull(input, "input");
    Objects.requireNonNull(expected, "expected");
    long header = Integer.BYTES + 32L;
    if (maxBytes < header + Integer.BYTES) throw new IOException("prefix exceeds import budget");
    if (input.readInt() != PREFIX_MAGIC) throw new IOException("not a LLaVA prefix");
    byte[] identity = new byte[32];
    input.readFully(identity);
    if (!MessageDigest.isEqual(identity, HexFormat.of().parseHex(portablePrefixFingerprint()))) {
      throw new IOException("incompatible text model, projector, preprocessing or KV identity");
    }
    MultimodalContent content =
        MultimodalContent.readFrom(
            input,
            text.contextCapacity(),
            metadata().vocabSize(),
            imageTokenCount(),
            maxBytes - header);
    String difference = content.firstDifference(expected);
    if (difference != null) {
      throw new IOException("prefix content is not the expected content: " + difference);
    }
    SharedInferencePrefix decoded =
        text.readPrefix(input, content.positions(), maxBytes - header - content.serializedBytes());
    return new LlavaPrefix(this, decoded, content);
  }

  private LlavaSession requireSession(InferenceSession session) {
    checkOpen();
    if (!(session instanceof LlavaSession owned) || owned.owner != this) {
      throw new IllegalArgumentException("session belongs to another backend");
    }
    if (owned.isClosed()) throw new IllegalStateException("session is closed");
    return owned;
  }

  private LlavaPrefix requirePrefix(SharedInferencePrefix prefix) {
    checkOpen();
    if (!(prefix instanceof LlavaPrefix owned) || owned.owner != this) {
      throw new IllegalArgumentException("prefix belongs to another backend");
    }
    return owned;
  }

  private void validateTokens(int[] tokens) {
    Objects.requireNonNull(tokens, "tokens");
    for (int token : tokens) {
      if (token < 0 || token >= metadata().vocabSize())
        throw new IllegalArgumentException("Invalid token");
    }
  }

  private void checkOpen() {
    if (closed) throw new IllegalStateException("Backend is closed");
  }

  private void requirePosition(int start, int count) {
    checkOpen();
    if (start != position || count <= 0 || count > contextCapacity() - start) {
      throw new IllegalArgumentException("Input must be contiguous and fit the context");
    }
  }
}
