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
package com.integrallis.models.api.catalog;

import java.util.List;

/**
 * Reports the models available on this machine, so callers do not have to describe them by hand.
 *
 * <p>Discovered through {@link java.util.ServiceLoader}. The interface lives here rather than
 * beside the router because the dependency runs the other way: ModelJars depends on models, so an
 * SPI it implements has to sit on the models side. Anything else would invert the tiers.
 *
 * <p>Written by hand, the fields of a model description are guesswork. Nobody knows their model's
 * time to first token, and the number is hardware-specific, so a value copied from a README is not
 * merely imprecise — it is a measurement of somebody else's machine. A catalog reports what was
 * actually measured here, and says nothing where nothing was measured.
 *
 * <p>Implementations must not throw for an absent or unreadable catalog. Returning an empty list is
 * the correct answer for a machine with no models installed, and is not an error worth propagating
 * into a caller that may be perfectly happy using hosted models.
 */
public interface ModelCatalogProvider {

  /**
   * Human-readable name of this catalog, used in diagnostics.
   *
   * @return a short name, e.g. {@code modeljars}
   */
  String name();

  /**
   * Everything this catalog knows about.
   *
   * @return the models, or an empty list when none are installed
   */
  List<DiscoveredModel> discover();
}
