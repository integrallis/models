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
package com.integrallis.models.tinystruct;

import java.util.Objects;
import org.tinystruct.ApplicationRuntimeException;
import org.tinystruct.system.Configuration;
import org.tinystruct.typesafe.client.TypesafeClient;
import org.tinystruct.typesafe.core.TypesafeRuntime;
import org.tinystruct.typesafe.core.argument.ArgumentValidator;
import org.tinystruct.typesafe.core.cache.CachingTypesafeClient;
import org.tinystruct.typesafe.core.cache.MemoryRoutingCache;
import org.tinystruct.typesafe.core.cache.RedisRoutingCache;
import org.tinystruct.typesafe.core.cache.RoutingCache;
import org.tinystruct.typesafe.core.candidate.TokenSpanExtractor;
import org.tinystruct.typesafe.core.catalog.ActionCatalog;
import org.tinystruct.typesafe.core.config.RoutingSettings;
import org.tinystruct.typesafe.core.config.TypesafeConfig;
import org.tinystruct.typesafe.core.confirmation.ConfirmationHandler;
import org.tinystruct.typesafe.core.confirmation.ConfirmationService;
import org.tinystruct.typesafe.core.confirmation.ConfirmedCallRunner;
import org.tinystruct.typesafe.core.confirmation.DefaultPrincipalResolver;
import org.tinystruct.typesafe.core.confirmation.PrincipalResolver;
import org.tinystruct.typesafe.core.execution.ActionExecutor;
import org.tinystruct.typesafe.core.execution.PathActionExecutor;
import org.tinystruct.typesafe.core.metrics.DispatchMetrics;
import org.tinystruct.typesafe.core.metrics.MeteredTypesafeClient;
import org.tinystruct.typesafe.core.pipeline.DispatchPipeline;
import org.tinystruct.typesafe.core.policy.ConfidencePolicy;

/**
 * Composition helper that wires an in-process {@link ModelsTypesafeClient} or {@link
 * RoutedTypesafeClient} into tinystruct's {@link TypesafeRuntime}.
 *
 * <p>Installing a runtime via {@link #install(TypesafeClient, Configuration)} allows tinystruct's
 * {@code SemanticDispatcher} ({@code bin/dispatcher semantic --input "..."}) to classify and
 * execute allowlisted {@code @Action} methods locally in-process without requiring a hosted API
 * key, while preserving TypeSafe's confidence gating, argument validation, confirmation workflow,
 * caching, and metrics.
 */
public final class ModelsTypesafeRuntime {

  private ModelsTypesafeRuntime() {}

  /**
   * Builds a {@link TypesafeRuntime} around the given {@link TypesafeClient} and tinystruct {@link
   * Configuration}.
   *
   * @param client the in-process or fleet-routed {@link TypesafeClient}
   * @param config the tinystruct configuration supplying routing, cache, and confirmation settings
   * @return a configured {@link TypesafeRuntime}
   */
  public static TypesafeRuntime create(TypesafeClient client, Configuration<String> config) {
    Objects.requireNonNull(client, "client");
    Objects.requireNonNull(config, "config");

    DispatchMetrics metrics = new DispatchMetrics();
    RoutingSettings settings = RoutingSettings.from(config);
    TypesafeClient decorated = decorate(client, config, metrics);

    PrincipalResolver principals =
        instantiate(
            config,
            TypesafeConfig.PRINCIPAL_RESOLVER,
            PrincipalResolver.class,
            new DefaultPrincipalResolver());
    ConfirmationService confirmation =
        instantiate(config, TypesafeConfig.CONFIRMATION_SERVICE, ConfirmationService.class, null);
    if (confirmation != null) {
      confirmation.configure(config);
    }

    ArgumentValidator validator = new ArgumentValidator(settings.maxArgumentLength());
    ActionExecutor executor = new PathActionExecutor();
    DispatchPipeline pipeline =
        DispatchPipeline.of(
            decorated,
            new TokenSpanExtractor(),
            ConfidencePolicy.from(config),
            validator,
            executor,
            confirmation,
            principals,
            settings,
            metrics);
    ConfirmedCallRunner runner =
        new ConfirmedCallRunner(new ActionCatalog(settings.allowedActions()), validator, executor);
    return new TypesafeRuntime(
        pipeline, new ConfirmationHandler(confirmation, principals, metrics), runner, metrics);
  }

  /**
   * Builds and installs a shared {@link TypesafeRuntime} backed by the given {@link TypesafeClient}
   * so {@code SemanticDispatcher} uses it immediately.
   *
   * @param client the in-process or fleet-routed {@link TypesafeClient}
   * @param config the tinystruct configuration
   * @return the installed {@link TypesafeRuntime}
   */
  public static TypesafeRuntime install(TypesafeClient client, Configuration<String> config) {
    TypesafeRuntime runtime = create(client, config);
    TypesafeRuntime.install(runtime);
    return runtime;
  }

  private static TypesafeClient decorate(
      TypesafeClient client, Configuration<String> config, DispatchMetrics metrics) {
    TypesafeClient metered = new MeteredTypesafeClient(client, metrics);
    RoutingCache cache = buildCache(config);
    return cache == null ? metered : new CachingTypesafeClient(metered, cache, metrics);
  }

  private static RoutingCache buildCache(Configuration<String> config) {
    String provider = config.get(TypesafeConfig.CACHE_PROVIDER);
    long ttl = TypesafeConfig.integer(config.get(TypesafeConfig.CACHE_TTL), 3600);
    String normalized =
        provider == null || provider.isBlank()
            ? "memory"
            : provider.strip().toLowerCase(java.util.Locale.ROOT);
    return switch (normalized) {
      case "none" -> null;
      case "memory" -> new MemoryRoutingCache(ttl);
      case "redis" -> new RedisRoutingCache(config, ttl);
      default ->
          throw new ApplicationRuntimeException(
              "Unknown "
                  + TypesafeConfig.CACHE_PROVIDER
                  + ": '"
                  + provider
                  + "'. Valid values: none, memory, redis.");
    };
  }

  private static <T> T instantiate(
      Configuration<String> config, String key, Class<T> type, T fallback) {
    String className = config.get(key);
    if (className == null || className.isBlank()) {
      return fallback;
    }
    try {
      return type.cast(Class.forName(className.strip()).getDeclaredConstructor().newInstance());
    } catch (ReflectiveOperationException | ClassCastException failure) {
      throw new ApplicationRuntimeException(
          "Cannot create " + key + " '" + className + "': " + failure, failure);
    }
  }
}
