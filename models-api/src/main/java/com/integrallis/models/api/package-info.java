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
 * Contracts for model inference, tokenization, sampling, and streaming.
 *
 * <p>Public parameters, collection elements, and return values are non-null unless a declaration
 * explicitly documents an optional value. Immutable value objects are safe to share between
 * threads. Inference backends and text-generation models own mutable sequence state and require
 * serialized use; tokenizer instances support concurrent read-only calls.
 */
package com.integrallis.models.api;
