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

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/** The transcription contract: the streaming default, the terminal-signal rule, and the guards. */
@Tag("unit")
class TranscriptionModelTest {

  private static final PcmAudio ONE_SECOND = new PcmAudio(16_000, 1, new float[16_000]);

  private static TranscriptionModel model(Transcript result, RuntimeException failure) {
    return new TranscriptionModel() {
      @Override
      public String modelName() {
        return "stub";
      }

      @Override
      public BackendDiagnostics diagnostics() {
        return null;
      }

      @Override
      public int requiredSampleRate() {
        return 16_000;
      }

      @Override
      public Transcript transcribe(PcmAudio audio, TranscriptionOptions options) {
        if (failure != null) {
          throw failure;
        }
        return result;
      }
    };
  }

  @Test
  void theStreamingDefaultEmitsOneTranscriptThenCompletes() {
    Transcript expected = Transcript.of("hello world", Duration.ofSeconds(1));
    List<String> events = new ArrayList<>();
    model(expected, null)
        .transcribe(
            ONE_SECOND,
            TranscriptionOptions.builder().build(),
            new TranscriptStream() {
              @Override
              public void onTranscript(Transcript transcript) {
                events.add("transcript:" + transcript.text());
              }

              @Override
              public void onComplete() {
                events.add("complete");
              }

              @Override
              public void onError(Throwable failure) {
                events.add("error");
              }
            });

    assertThat(events).containsExactly("transcript:hello world", "complete");
  }

  @Test
  void aFailureReachesOnErrorAndIsNotFollowedByCompletion() {
    // The contract this pins: onError and onComplete are both terminal, so a consumer that saw an
    // error must not then be told the stream completed normally. TextToSpeechModel's default makes
    // the same promise, and the two are expected to behave alike.
    List<String> events = new ArrayList<>();
    model(null, new IllegalStateException("decoder failed"))
        .transcribe(
            ONE_SECOND,
            TranscriptionOptions.builder().build(),
            new TranscriptStream() {
              @Override
              public void onTranscript(Transcript transcript) {
                events.add("transcript");
              }

              @Override
              public void onComplete() {
                events.add("complete");
              }

              @Override
              public void onError(Throwable failure) {
                events.add("error:" + failure.getMessage());
              }
            });

    assertThat(events).containsExactly("error:decoder failed");
  }

  @Test
  void aConsumerThrowingFromOnCompleteSurfacesRatherThanBecomingAnError() {
    // If onComplete throws, that is the consumer's failure and it must propagate. Routing it to
    // onError would report a transcription failure that did not happen, and would deliver two
    // terminal signals.
    assertThatThrownBy(
            () ->
                model(Transcript.of("x", Duration.ofSeconds(1)), null)
                    .transcribe(
                        ONE_SECOND,
                        TranscriptionOptions.builder().build(),
                        new TranscriptStream() {
                          @Override
                          public void onTranscript(Transcript transcript) {}

                          @Override
                          public void onComplete() {
                            throw new IllegalArgumentException("consumer broke");
                          }
                        }))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessage("consumer broke");
  }

  @Test
  void aTranscriptCarriesItsSpanAndWhetherItWasTruncated() {
    // A word error rate only compares when reference and hypothesis cover the same audio, and a
    // transcript that stopped early otherwise reads as a short utterance rather than a cut one.
    Transcript full = Transcript.of("complete sentence", Duration.ofSeconds(3));
    assertThat(full.truncated()).isFalse();
    assertThat(full.audioDuration()).isEqualTo(Duration.ofSeconds(3));
    assertThat(full.segments()).isEmpty();

    Transcript cut =
        new Transcript(
            "half a sent",
            List.of(new Transcript.Segment("half a sent", Duration.ZERO, Duration.ofSeconds(1))),
            Duration.ofSeconds(3),
            true);
    assertThat(cut.truncated()).isTrue();
    assertThat(cut.segments()).hasSize(1);
  }

  @Test
  void segmentAndTranscriptGuardsRejectImpossibleSpans() {
    assertThatThrownBy(
            () -> new Transcript.Segment("x", Duration.ofSeconds(2), Duration.ofSeconds(1)))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("precedes its start");
    assertThatThrownBy(() -> Transcript.of("x", Duration.ofSeconds(-1)))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> TranscriptionOptions.builder().maximumTokens(-1))
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void theRequiredSampleRateIsExposedSoAMismatchCanBeRefused() {
    // parakeet declares preprocessor.sample_rate = 16000. Feeding 44.1 kHz to a 16 kHz mel front
    // end transcribes noise and throws nothing, so the rate has to be readable by a caller.
    assertThat(model(null, null).requiredSampleRate()).isEqualTo(16_000);
    assertThat(TranscriptionOptions.builder().build().head())
        .isEqualTo(TranscriptionOptions.Head.MODEL_DEFAULT);
    assertThat(TranscriptionOptions.builder().build().language()).isEmpty();
  }
}
