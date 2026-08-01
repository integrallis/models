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

import com.integrallis.models.api.LogitBatch;
import java.io.IOException;

/** Architecture-neutral decoder contract retained inside the pure-Java backend. */
interface PureJavaDecoder extends AutoCloseable {

  interface Session {
    int checkpoint();
  }

  int maxBatchSize();

  float[] forward(int token, int position);

  float[] forwardTransient(int token, int position);

  float[] prefill(int[] tokens, int startPosition);

  Session openSession();

  float[] forward(Session session, int token, int position);

  float[] forwardTransient(Session session, int token, int position);

  float[] prefill(Session session, int[] tokens, int startPosition);

  LogitBatch forwardBatch(Session[] sessions, int[] tokens);

  LogitBatch forwardBatchTransient(Session[] sessions, int[] tokens);

  void rewind(Session session, int checkpoint);

  void reset(Session session);

  int checkpoint();

  LogitBatch verify(int[] tokens, int startPosition);

  LogitBatch verifyTransient(int[] tokens, int startPosition);

  void rewind(int checkpoint);

  void reset();

  @Override
  void close() throws IOException;
}
