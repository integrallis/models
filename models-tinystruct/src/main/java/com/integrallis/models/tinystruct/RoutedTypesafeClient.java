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

import com.integrallis.models.router.ModelFleet;
import com.integrallis.models.router.RoutingContinuity;
import com.integrallis.models.router.RoutingExecutionException;
import com.integrallis.models.router.RoutingExecutionOptions;
import com.integrallis.models.router.RoutingRequest;
import com.integrallis.models.router.RoutingRequirements;
import com.integrallis.models.router.RoutingUsage;
import java.util.Objects;
import java.util.function.Function;
import org.tinystruct.ApplicationException;
import org.tinystruct.data.component.Builder;
import org.tinystruct.typesafe.client.RoutingResult;
import org.tinystruct.typesafe.client.TypesafeClient;

/**
 * A {@link TypesafeClient} that routes TypeSafe semantic classification requests across a {@link
 * ModelFleet} of local and remote {@link TypesafeClient} instances.
 *
 * <p>This allows a tinystruct application to prioritize local in-process classification via {@link
 * ModelsTypesafeClient} while failing over to peer or hosted {@code HttpTypesafeClient} instances
 * according to capability, data-boundary, latency, and budget policies.
 */
public final class RoutedTypesafeClient implements TypesafeClient {

  private final ModelFleet<TypesafeClient> fleet;
  private final Function<org.tinystruct.typesafe.client.RoutingRequest, RoutingRequest>
      requestFactory;
  private final Function<org.tinystruct.typesafe.client.RoutingRequest, RoutingContinuity>
      continuityFactory;
  private final Function<org.tinystruct.typesafe.client.RoutingRequest, RoutingRequirements>
      requirementsFactory;
  private final Function<org.tinystruct.typesafe.client.RoutingRequest, RoutingExecutionOptions>
      executionFactory;

  /** Routes classification requests using default token estimation and no extra requirements. */
  public RoutedTypesafeClient(ModelFleet<TypesafeClient> fleet) {
    this(
        fleet,
        RoutedTypesafeClient::defaultRequest,
        ignored -> RoutingContinuity.none(),
        ignored -> RoutingRequirements.none(),
        ignored -> RoutingExecutionOptions.unlimited());
  }

  /** Routes classification requests with an application-supplied request factory. */
  public RoutedTypesafeClient(
      ModelFleet<TypesafeClient> fleet,
      Function<org.tinystruct.typesafe.client.RoutingRequest, RoutingRequest> requestFactory) {
    this(
        fleet,
        requestFactory,
        ignored -> RoutingContinuity.none(),
        ignored -> RoutingRequirements.none(),
        ignored -> RoutingExecutionOptions.unlimited());
  }

  /**
   * Routes classification requests with custom request, continuity, requirements, and execution
   * options extractors.
   */
  public RoutedTypesafeClient(
      ModelFleet<TypesafeClient> fleet,
      Function<org.tinystruct.typesafe.client.RoutingRequest, RoutingRequest> requestFactory,
      Function<org.tinystruct.typesafe.client.RoutingRequest, RoutingContinuity> continuityFactory,
      Function<org.tinystruct.typesafe.client.RoutingRequest, RoutingRequirements>
          requirementsFactory,
      Function<org.tinystruct.typesafe.client.RoutingRequest, RoutingExecutionOptions>
          executionFactory) {
    this.fleet = Objects.requireNonNull(fleet, "fleet");
    this.requestFactory = Objects.requireNonNull(requestFactory, "requestFactory");
    this.continuityFactory = Objects.requireNonNull(continuityFactory, "continuityFactory");
    this.requirementsFactory = Objects.requireNonNull(requirementsFactory, "requirementsFactory");
    this.executionFactory = Objects.requireNonNull(executionFactory, "executionFactory");
  }

  @Override
  public RoutingResult classify(org.tinystruct.typesafe.client.RoutingRequest request)
      throws ApplicationException {
    Objects.requireNonNull(request, "request");
    RoutingRequest routeRequest =
        Objects.requireNonNull(requestFactory.apply(request), "routing request");
    RoutingRequirements requirements =
        Objects.requireNonNull(requirementsFactory.apply(request), "routing requirements");
    RoutingContinuity continuity =
        Objects.requireNonNull(continuityFactory.apply(request), "routing continuity");
    RoutingExecutionOptions options =
        Objects.requireNonNull(executionFactory.apply(request), "routing execution options");

    try {
      return fleet
          .execute(
              routeRequest,
              requirements,
              continuity,
              options,
              client -> invokeClassify(client, request),
              RoutedTypesafeClient::extractUsage)
          .value();
    } catch (RoutingExecutionException executionException) {
      if (executionException.getCause() instanceof WrappedApplicationException wrapped) {
        throw wrapped.cause();
      }
      if (executionException.getCause() instanceof ApplicationException appEx) {
        throw appEx;
      }
      throw executionException;
    } catch (WrappedApplicationException wrapped) {
      throw wrapped.cause();
    }
  }

  /**
   * Estimates a router {@link RoutingRequest} from a TypeSafe {@link
   * org.tinystruct.typesafe.client.RoutingRequest}.
   */
  public static RoutingRequest defaultRequest(
      org.tinystruct.typesafe.client.RoutingRequest request) {
    Objects.requireNonNull(request, "request");
    String stateText = request.getState() == null ? "" : request.getState().toString();
    String questionsText = request.getQuestions() == null ? "" : request.getQuestions().toString();
    String combined = stateText + "\n" + questionsText;
    int estimatedInputTokens = Math.max(1, (combined.length() + 3) / 4);
    return RoutingRequest.builder(stateText)
        .taskType("classification")
        .estimatedTokens(estimatedInputTokens)
        .build();
  }

  private static RoutingResult invokeClassify(
      TypesafeClient client, org.tinystruct.typesafe.client.RoutingRequest request) {
    try {
      return client.classify(request);
    } catch (ApplicationException failure) {
      throw new WrappedApplicationException(failure);
    }
  }

  static RoutingUsage extractUsage(RoutingResult result) {
    if (result == null || result.getUsage() == null) {
      return null;
    }
    Builder usage = result.getUsage();
    Integer inputTokens = intValue(usage, "input_tokens", "prompt_tokens");
    Integer outputTokens = intValue(usage, "output_tokens", "completion_tokens");
    if (inputTokens == null || outputTokens == null || (inputTokens == 0 && outputTokens == 0)) {
      return null;
    }
    return new RoutingUsage(Math.max(0, inputTokens), Math.max(0, outputTokens));
  }

  private static Integer intValue(Builder usage, String primaryKey, String fallbackKey) {
    Object value = usage.get(primaryKey);
    if (value == null) {
      value = usage.get(fallbackKey);
    }
    if (value instanceof Number number) {
      return number.intValue();
    }
    if (value != null) {
      try {
        return Integer.parseInt(value.toString().strip());
      } catch (NumberFormatException ignored) {
        return null;
      }
    }
    return null;
  }

  private static final class WrappedApplicationException extends RuntimeException {
    private static final long serialVersionUID = 1L;
    private final ApplicationException applicationException;

    private WrappedApplicationException(ApplicationException cause) {
      super(cause.getMessage(), cause);
      this.applicationException = cause;
    }

    private ApplicationException cause() {
      return applicationException;
    }
  }
}
