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
package com.integrallis.models.api;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.Objects;

/** Encodes in-memory PCM audio into portable WAVE files. */
public final class WavEncoder {

  private static final int HEADER_BYTES = 44;

  private WavEncoder() {}

  /** Encodes normalized floating-point samples as little-endian signed 16-bit PCM. */
  public static byte[] pcm16(PcmAudio audio) {
    Objects.requireNonNull(audio, "audio");
    float[] samples = audio.samples();
    int dataBytes = Math.multiplyExact(samples.length, Short.BYTES);
    ByteBuffer output =
        ByteBuffer.allocate(Math.addExact(HEADER_BYTES, dataBytes)).order(ByteOrder.LITTLE_ENDIAN);

    ascii(output, "RIFF");
    output.putInt(36 + dataBytes);
    ascii(output, "WAVE");
    ascii(output, "fmt ");
    output.putInt(16);
    output.putShort((short) 1);
    output.putShort((short) audio.channels());
    output.putInt(audio.sampleRate());
    output.putInt(Math.multiplyExact(audio.sampleRate(), audio.channels() * Short.BYTES));
    output.putShort((short) (audio.channels() * Short.BYTES));
    output.putShort((short) 16);
    ascii(output, "data");
    output.putInt(dataBytes);
    for (float sample : samples) {
      output.putShort(toPcm16(sample));
    }
    return output.array();
  }

  private static short toPcm16(float sample) {
    if (sample <= -1.0f) {
      return Short.MIN_VALUE;
    }
    if (sample >= 1.0f) {
      return Short.MAX_VALUE;
    }
    return (short) Math.round(sample * (sample < 0.0f ? 32768.0f : 32767.0f));
  }

  private static void ascii(ByteBuffer output, String value) {
    output.put(value.getBytes(StandardCharsets.US_ASCII));
  }
}
