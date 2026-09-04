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

/** Push callback for ordered PCM chunks from one speech synthesis. */
public interface AudioStream {

  /** Called with the next non-null audio chunk. All chunks use the same format. */
  void onAudio(PcmAudio audio);

  /** Called once after the final audio chunk. */
  void onComplete();

  /** Called once when synthesis fails. */
  void onError(Throwable failure);
}
