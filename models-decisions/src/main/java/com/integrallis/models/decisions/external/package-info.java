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

/**
 * Verdicts obtained from third-party decision services, for scoring only.
 *
 * <p>Nothing in this package may be referenced by the training path. A third-party service's terms
 * may prohibit using its output to train, distil or imitate a model -- TypeSafe's Master Customer
 * Agreement section 2.3(b) does -- so the boundary is kept structural and is enforced by a test.
 */
package com.integrallis.models.decisions.external;
