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
 * Typed, bounded, calibrated decisions taken over model state.
 *
 * <p>A decision names a finite answer space up front and returns a probability distribution over
 * exactly that space. Nothing here generates text: the answer can only be one of the outcomes the
 * caller declared, so a decision cannot invent a value its schema does not contain. That bound is a
 * structural property of the types, not a validation step applied afterwards.
 *
 * <p>Calibration is measured, never assumed. A distribution is only useful if its probabilities
 * track observed frequencies, and this package supplies the scaling and the scoring rules that
 * establish whether they do. A well-calibrated distribution still carries no guarantee that any
 * individual decision is correct.
 *
 * <p>Value objects in this package are immutable and safe to share between threads.
 */
package com.integrallis.models.decisions;
