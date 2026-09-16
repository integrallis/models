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
package com.integrallis.models.runtime;

import com.integrallis.models.api.ActivatedAdapterMetadata;
import com.integrallis.models.api.ModelPrompt;
import java.util.List;

/** A model that opens request-scoped activated tool and exact base branches. */
public interface ActivatedToolModel extends ConstrainedTextGenerationModel {

  /** Returns the exact adapter provenance and activation contract. */
  ActivatedAdapterMetadata adapter();

  /**
   * Returns complete protocol outputs by which the specialist declines every declared tool.
   *
   * <p>Framework adapters include these values in finite constrained-decoding grammars so schema
   * enforcement cannot turn an abstention into an irrelevant tool call.
   */
  default List<String> toolAbstentionOutputs() {
    return List.of();
  }

  /** Opens one physically shared tool/base turn from a rendered tool prompt. */
  SharedToolTurn openToolTurn(ModelPrompt renderedToolPrompt);
}
