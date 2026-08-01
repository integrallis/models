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
package com.integrallis.models.backend.purejava.gemma4;

import com.integrallis.models.backend.purejava.gemma4.Gemma4TensorLayout.ExpertWeights;
import com.integrallis.models.backend.purejava.gemma4.Gemma4TensorLayout.TensorSlice;
import java.io.EOFException;
import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;

/** Loads the two discontiguous source ranges for one Gemma 4 expert into a bounded slot. */
final class Gemma4ExpertLoader implements AutoCloseable {

  private static final int MAX_CONSECUTIVE_EMPTY_READS = 16;

  @FunctionalInterface
  interface PositionalReader extends AutoCloseable {
    int read(ByteBuffer destination, long filePosition) throws IOException;

    @Override
    default void close() throws IOException {}
  }

  record LoadedExpert(
      ExpertWeights weights, MemorySegment storage, MemorySegment gateUp, MemorySegment down) {
    LoadedExpert {
      Objects.requireNonNull(weights, "weights");
      Objects.requireNonNull(storage, "storage");
      Objects.requireNonNull(gateUp, "gateUp");
      Objects.requireNonNull(down, "down");
    }
  }

  private final PositionalReader reader;
  private final AtomicBoolean closed = new AtomicBoolean();

  Gemma4ExpertLoader(PositionalReader reader) {
    this.reader = Objects.requireNonNull(reader, "reader");
  }

  static Gemma4ExpertLoader open(Path path) throws IOException {
    Objects.requireNonNull(path, "path");
    FileChannel channel = FileChannel.open(path, StandardOpenOption.READ);
    return new Gemma4ExpertLoader(
        new PositionalReader() {
          @Override
          public int read(ByteBuffer destination, long filePosition) throws IOException {
            return channel.read(destination, filePosition);
          }

          @Override
          public void close() throws IOException {
            channel.close();
          }
        });
  }

  LoadedExpert loadInto(ExpertWeights weights, MemorySegment destination) throws IOException {
    Objects.requireNonNull(weights, "weights");
    Objects.requireNonNull(destination, "destination");
    requireOpen();
    long requiredBytes = weights.totalBytes();
    if (destination.byteSize() < requiredBytes) {
      throw new IllegalArgumentException(
          "Expert slot must hold " + requiredBytes + " bytes, found " + destination.byteSize());
    }

    TensorSlice gateUpSource = weights.gateUp();
    MemorySegment gateUp = destination.asSlice(0, gateUpSource.byteSize());
    readFully(gateUpSource, gateUp);

    TensorSlice downSource = weights.down();
    MemorySegment down = destination.asSlice(gateUpSource.byteSize(), downSource.byteSize());
    readFully(downSource, down);
    return new LoadedExpert(weights, destination, gateUp, down);
  }

  private void readFully(TensorSlice source, MemorySegment destination) throws IOException {
    ByteBuffer buffer = destination.asByteBuffer();
    long filled = 0;
    int emptyReads = 0;
    while (buffer.hasRemaining()) {
      int remaining = buffer.remaining();
      int read = reader.read(buffer, Math.addExact(source.fileOffset(), filled));
      if (read < 0) {
        throw new EOFException(
            "Unexpected EOF reading "
                + source.tensorName()
                + ": required "
                + source.byteSize()
                + " bytes, read "
                + filled);
      }
      if (read == 0) {
        emptyReads++;
        if (emptyReads > MAX_CONSECUTIVE_EMPTY_READS) {
          throw new IOException(
              "No progress reading "
                  + source.tensorName()
                  + " at file offset "
                  + source.fileOffset());
        }
        Thread.onSpinWait();
        continue;
      }
      if (read > remaining) {
        throw new IOException(
            "Reader returned " + read + " bytes with only " + remaining + " bytes remaining");
      }
      emptyReads = 0;
      filled = Math.addExact(filled, read);
    }
  }

  private void requireOpen() {
    if (closed.get()) {
      throw new IllegalStateException("Gemma 4 expert loader is closed");
    }
  }

  @Override
  public void close() throws IOException {
    if (closed.compareAndSet(false, true)) {
      reader.close();
    }
  }
}
