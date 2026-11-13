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

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

import com.integrallis.models.api.InferenceSession;
import com.integrallis.models.api.LogitBatch;
import com.integrallis.models.api.ModelMetadata;
import com.integrallis.models.api.SharedInferencePrefix;
import com.integrallis.models.api.SharedPrefixInferenceBackend;
import com.integrallis.models.api.Tokenizer;
import com.integrallis.models.decisions.Choice;
import com.integrallis.models.decisions.Noul;
import com.integrallis.models.decisions.Verdict;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.tinystruct.AbstractApplication;
import org.tinystruct.ApplicationContext;
import org.tinystruct.ApplicationException;
import org.tinystruct.ApplicationRuntimeException;
import org.tinystruct.application.ActionRegistry;
import org.tinystruct.data.component.Builder;
import org.tinystruct.system.ApplicationManager;
import org.tinystruct.system.Configuration;
import org.tinystruct.system.Settings;
import org.tinystruct.system.annotation.Action;
import org.tinystruct.system.annotation.Argument;
import org.tinystruct.typesafe.client.RoutingRequest;
import org.tinystruct.typesafe.client.RoutingResult;
import org.tinystruct.typesafe.core.SemanticDispatcher;
import org.tinystruct.typesafe.core.TypesafeRuntime;
import org.tinystruct.typesafe.core.api.DispatchResult;
import org.tinystruct.typesafe.core.config.TypesafeConfig;
import org.tinystruct.typesafe.core.confirmation.ConfirmationService;
import org.tinystruct.typesafe.core.confirmation.PendingCall;

@Tag("unit")
class ModelsTypesafeClientTest {

  @BeforeEach
  void resetRuntime() {
    TypesafeRuntime.install(null);
  }

  @AfterEach
  void cleanUpRuntime() {
    TypesafeRuntime.install(null);
  }

  @Test
  void classifiesChoiceNoulAndScoreQuestionsOverSharedPrefixInSinglePrefill() throws Exception {
    StubDecisionBackend backend = new StubDecisionBackend();
    // Favour option B (token 11) -> second option in choice/noul/score
    backend.favour(11, 6.0f);

    Builder questions = new Builder();
    Map<String, String> toolOptions = new LinkedHashMap<>();
    toolOptions.put("delete-user", "Delete user");
    toolOptions.put("create-user", "Create user");
    toolOptions.put("other", "None");
    questions.put("__tool__", choiceQuestion("Which tool applies?", toolOptions));
    questions.put("create-user.dryRun", noulQuestion("Did the user enable dryRun?"));
    Map<String, String> priorityLevels = new LinkedHashMap<>();
    priorityLevels.put("low", "Low");
    priorityLevels.put("high", "High");
    questions.put("create-user.priority", scoreQuestion("Priority level?", priorityLevels));

    try (ModelsTypesafeClient client = new ModelsTypesafeClient(backend)) {
      assertThat(client.modelVersion()).isEqualTo("stub-decision-backend");
      RoutingResult result =
          client.classify(
              new RoutingRequest("create user Alice with high priority", questions, ""));

      assertThat(result.getModelVersion()).isEqualTo("stub-decision-backend");
      assertThat(result.getChoice("__tool__")).isEqualTo("create-user");
      assertThat(result.getChoiceConfidence("__tool__")).isGreaterThan(0.90);
      assertThat(((Builder) result.getAnswers().get("__tool__")).get("probabilities")).isNotNull();

      assertThat(result.hasNoul("create-user.dryRun")).isTrue();
      assertThat(result.getNoul("create-user.dryRun")).isGreaterThan(0.90);

      assertThat(result.getChoice("create-user.priority")).isEqualTo("low");
      assertThat(result.getUsage().get("state_prefills")).isEqualTo(1);
      assertThat(result.getUsage().get("shared_prefix_proven")).isEqualTo(true);
      assertThat(backend.freezeCount).isEqualTo(1);
    }
    assertThat(backend.closed).isTrue();
  }

  @Test
  void fallsBackToPerQuestionPrefillWhenSharedPrefixesAreNotSupported() throws Exception {
    StubDecisionBackend backend = new StubDecisionBackend();
    backend.supportsSharedPrefixes = false;
    backend.favour(10, 5.0f);

    Builder questions = new Builder();
    questions.put(
        "q1", choiceQuestion("First?", Map.of("alpha", "First option", "beta", "Second option")));
    questions.put("q2", noulQuestion("Second?"));

    Builder stateBuilder = new Builder();
    stateBuilder.put("input", "alpha state");

    try (ModelsTypesafeClient client = new ModelsTypesafeClient(backend, "custom-v1", 1.0)) {
      RoutingResult result =
          client.classify(new RoutingRequest(stateBuilder, questions, "override-model"));

      assertThat(result.getModelVersion()).isEqualTo("override-model");
      assertThat(result.getChoice("q1")).isEqualTo("alpha");
      assertThat(result.getNoul("q2")).isLessThan(0.10);
      assertThat(result.getUsage().get("state_prefills")).isEqualTo(2);
      assertThat(result.getUsage().get("shared_prefix_proven")).isEqualTo(false);
    }
  }

  @Test
  void supportsCustomEvaluatorFunctionAndNoulCriteria() throws Exception {
    ModelsTypesafeClient client =
        new ModelsTypesafeClient(
            (state, space) -> {
              if (space instanceof Choice choice) {
                return new Verdict(choice, new double[] {0.15, 0.85});
              }
              return new Verdict(space, new double[] {0.20, 0.80});
            },
            "custom-head");

    Builder questions = new Builder();
    questions.put("tool", choiceQuestion("Which tool?", Map.of("a", "a", "b", "Option B")));
    Builder noulWithCriteria = noulQuestion("Is it stated?");
    Builder criteria = new Builder();
    criteria.put("false", "No value stated");
    criteria.put("true", "Explicitly stated");
    noulWithCriteria.put("criteria", criteria);
    questions.put("stated", noulWithCriteria);

    RoutingResult result = client.classify(new RoutingRequest("state text", questions, null));
    assertThat(result.getModelVersion()).isEqualTo("custom-head");
    assertThat(result.getChoice("tool")).isEqualTo("b");
    assertThat(result.getChoiceConfidence("tool")).isCloseTo(0.390159, within(1e-4));
    assertThat(result.getNoul("stated")).isCloseTo(0.80, within(1e-6));
    client.close();
  }

  @Test
  void validatesMalformedRoutingRequestsAndQuestions() {
    ModelsTypesafeClient client =
        new ModelsTypesafeClient(
            (state, space) -> new Verdict(space, new double[] {0.5, 0.5}), "test-model");

    Builder validQuestions = new Builder();
    validQuestions.put("q", noulQuestion("Valid?"));

    assertThatThrownBy(() -> client.classify(new RoutingRequest(null, validQuestions, "m")))
        .isInstanceOf(ApplicationException.class)
        .hasMessageContaining("state must not be null");

    assertThatThrownBy(() -> client.classify(new RoutingRequest("   ", validQuestions, "m")))
        .isInstanceOf(ApplicationException.class)
        .hasMessageContaining("state must not be blank");

    assertThatThrownBy(() -> client.classify(new RoutingRequest("hi", new Builder(), "m")))
        .isInstanceOf(ApplicationException.class)
        .hasMessageContaining("at least one question");

    Builder nonBuilderQuestion = new Builder();
    nonBuilderQuestion.put("bad", "not-a-builder");
    assertThatThrownBy(() -> client.classify(new RoutingRequest("hi", nonBuilderQuestion, "m")))
        .isInstanceOf(ApplicationException.class)
        .hasMessageContaining("must be a Builder");

    assertThatThrownBy(() -> ModelsTypesafeClient.toAnswerSpace("q", null))
        .isInstanceOf(ApplicationException.class)
        .hasMessageContaining("must not be null");

    Builder missingType = new Builder();
    missingType.put("instructions", "Something");
    assertThatThrownBy(() -> ModelsTypesafeClient.toAnswerSpace("q", missingType))
        .isInstanceOf(ApplicationException.class)
        .hasMessageContaining("missing 'type'");

    Builder missingInstructions = new Builder();
    missingInstructions.put("type", "noul");
    assertThatThrownBy(() -> ModelsTypesafeClient.toAnswerSpace("q", missingInstructions))
        .isInstanceOf(ApplicationException.class)
        .hasMessageContaining("missing 'instructions'");

    Builder unknownType = new Builder();
    unknownType.put("type", "unknown");
    unknownType.put("instructions", "What?");
    assertThatThrownBy(() -> ModelsTypesafeClient.toAnswerSpace("q", unknownType))
        .isInstanceOf(ApplicationException.class)
        .hasMessageContaining("unsupported type");

    Builder choiceWithoutCriteria = new Builder();
    choiceWithoutCriteria.put("type", "choice");
    choiceWithoutCriteria.put("instructions", "Pick one");
    assertThatThrownBy(() -> ModelsTypesafeClient.toAnswerSpace("q", choiceWithoutCriteria))
        .isInstanceOf(ApplicationException.class)
        .hasMessageContaining("requires a non-empty 'criteria'");

    Builder scoreWithoutCriteria = new Builder();
    scoreWithoutCriteria.put("type", "score");
    scoreWithoutCriteria.put("instructions", "Rate it");
    assertThatThrownBy(() -> ModelsTypesafeClient.toAnswerSpace("q", scoreWithoutCriteria))
        .isInstanceOf(ApplicationException.class)
        .hasMessageContaining("requires a non-empty 'criteria'");

    Builder singleOptionChoice = new Builder();
    singleOptionChoice.put("type", "choice");
    singleOptionChoice.put("instructions", "Pick one");
    Builder oneCriterion = new Builder();
    oneCriterion.put("only", "Only option");
    singleOptionChoice.put("criteria", oneCriterion);
    assertThatThrownBy(() -> ModelsTypesafeClient.toAnswerSpace("q", singleOptionChoice))
        .isInstanceOf(ApplicationException.class)
        .hasMessageContaining("is invalid");
  }

  @Test
  void wrapsEvaluatorAndBackendRuntimeFailuresInApplicationException() {
    ModelsTypesafeClient failingEvaluator =
        new ModelsTypesafeClient(
            (state, space) -> {
              throw new IllegalStateException("head failed");
            },
            "failing");
    Builder questions = new Builder();
    questions.put("q", noulQuestion("Is it?"));

    assertThatThrownBy(() -> failingEvaluator.classify(new RoutingRequest("hi", questions, "m")))
        .isInstanceOf(ApplicationException.class)
        .hasMessageContaining("head failed");

    StubDecisionBackend emptyTokenBackend = new StubDecisionBackend();
    emptyTokenBackend.returnEmptyTokens = true;
    ModelsTypesafeClient emptyTokenClient = new ModelsTypesafeClient(emptyTokenBackend);
    assertThatThrownBy(() -> emptyTokenClient.classify(new RoutingRequest("hi", questions, "m")))
        .isInstanceOf(ApplicationException.class)
        .hasMessageContaining("no tokens");
  }

  @Test
  void installsModelsTypesafeRuntimeAndRoutesSemanticActionInProcess() throws Exception {
    UserAdminApp app = new UserAdminApp();
    ApplicationManager.install(app, new Settings());

    ModelsTypesafeClient client =
        new ModelsTypesafeClient(
            (state, space) -> {
              if (space instanceof Choice choice) {
                List<String> options = choice.options();
                double[] probs = new double[options.size()];
                int winnerIndex = 0;
                if (options.contains("create-user")) {
                  winnerIndex = options.indexOf("create-user");
                } else if (options.contains("Alice")) {
                  winnerIndex = options.indexOf("Alice");
                } else if (options.contains("ADMIN")) {
                  winnerIndex = options.indexOf("ADMIN");
                }
                for (int i = 0; i < probs.length; i++) {
                  probs[i] = i == winnerIndex ? 0.96 : 0.04 / Math.max(1, probs.length - 1);
                }
                return new Verdict(choice, probs);
              }
              if (space instanceof Noul noul) {
                return new Verdict(noul, new double[] {0.05, 0.95});
              }
              throw new AssertionError("unexpected space: " + space);
            },
            "models-local-1.0");

    SimpleConfig config =
        new SimpleConfig(
            Map.of(
                TypesafeConfig.ALLOWED_ACTIONS, "create-user",
                TypesafeConfig.CACHE_PROVIDER, "memory",
                TypesafeConfig.MIN_CONFIDENCE, "0.75",
                TypesafeConfig.CONFIRMATION_SERVICE, StubConfirmationService.class.getName()));

    TypesafeRuntime runtime = ModelsTypesafeRuntime.install(client, config);
    assertThat(runtime).isSameAs(TypesafeRuntime.shared(config));

    DispatchResult dispatchResult =
        runtime
            .router()
            .route(
                "create user Alice with ADMIN role",
                ActionRegistry.getInstance(),
                new ApplicationContext());
    assertThat(dispatchResult.getStatus()).isEqualTo(DispatchResult.Status.EXECUTED);
    assertThat(dispatchResult.getResult()).isEqualTo("created:Alice:ADMIN");

    SemanticDispatcher dispatcher = new SemanticDispatcher();
    dispatcher.setConfiguration(config);
    ApplicationContext context = new ApplicationContext();
    context.setAttribute("--input", "create user Alice with ADMIN role");
    dispatcher.setContext(context);
    Builder formatted = dispatcher.semantic();
    assertThat(formatted.get("status")).isEqualTo("EXECUTED");
    assertThat(formatted.get("result")).isEqualTo("created:Alice:ADMIN");
    assertThat(dispatcher.metrics().get("totalRoutings")).isEqualTo(2L);
    assertThat(dispatcher.metrics().get("executed")).isEqualTo(2L);
  }

  @Test
  void rejectsUnknownCacheProviderAndInvalidConfiguredClass() {
    ModelsTypesafeClient client =
        new ModelsTypesafeClient(
            (state, space) -> new Verdict(space, new double[] {0.5, 0.5}), "m");

    SimpleConfig badCache =
        new SimpleConfig(Map.of(TypesafeConfig.CACHE_PROVIDER, "invalid-cache"));
    assertThatThrownBy(() -> ModelsTypesafeRuntime.create(client, badCache))
        .isInstanceOf(ApplicationRuntimeException.class)
        .hasMessageContaining("invalid-cache");

    SimpleConfig badClass =
        new SimpleConfig(
            Map.of(
                TypesafeConfig.CACHE_PROVIDER, "none",
                TypesafeConfig.PRINCIPAL_RESOLVER, "no.such.ResolverClass"));
    assertThatThrownBy(() -> ModelsTypesafeRuntime.create(client, badClass))
        .isInstanceOf(ApplicationRuntimeException.class)
        .hasMessageContaining("no.such.ResolverClass");
  }

  private static Builder choiceQuestion(String instructions, Map<String, String> criteriaMap) {
    Builder question = new Builder();
    question.put("type", "choice");
    question.put("instructions", instructions);
    Builder criteria = new Builder();
    criteriaMap.forEach(criteria::put);
    question.put("criteria", criteria);
    return question;
  }

  private static Builder noulQuestion(String instructions) {
    Builder question = new Builder();
    question.put("type", "noul");
    question.put("instructions", instructions);
    return question;
  }

  private static Builder scoreQuestion(String instructions, Map<String, String> criteriaMap) {
    Builder question = new Builder();
    question.put("type", "score");
    question.put("instructions", instructions);
    Builder criteria = new Builder();
    criteriaMap.forEach(criteria::put);
    question.put("criteria", criteria);
    return question;
  }

  public enum Role {
    USER,
    ADMIN
  }

  public static final class UserAdminApp extends AbstractApplication {
    @Override
    public void init() {
      setTemplateRequired(false);
    }

    @Override
    public String version() {
      return "1.0.0";
    }

    @Action(
        value = "create-user",
        description = "Create a user account",
        arguments = {
          @Argument(key = "name", description = "The user name"),
          @Argument(key = "role", description = "The user role")
        })
    public String createUser(String name, Role role) {
      return "created:" + name + ":" + role.name();
    }
  }

  public static final class StubConfirmationService implements ConfirmationService {
    @Override
    public String open(PendingCall call) {
      return "pending-1";
    }

    @Override
    public DispatchResult confirm(String pendingId, String principal) {
      return DispatchResult.executed("confirmed");
    }

    @Override
    public void reject(String pendingId, String principal) {}
  }

  @SuppressWarnings("serial")
  private static final class SimpleConfig implements Configuration<String> {
    private static final long serialVersionUID = 1L;
    private final HashMap<String, String> values = new HashMap<>();

    SimpleConfig(Map<String, String> initial) {
      values.putAll(initial);
    }

    @Override
    public void set(String name, String value) {
      values.put(name, value);
    }

    @Override
    public String get(String name) {
      return values.getOrDefault(name, "");
    }

    @Override
    public void remove(String name) {
      values.remove(name);
    }

    @Override
    public Set<String> propertyNames() {
      return values.keySet();
    }

    @Override
    public String getOrDefault(String key, String defaultValue) {
      return values.getOrDefault(key, defaultValue);
    }

    @Override
    public void setIfAbsent(String key, String value) {
      values.putIfAbsent(key, value);
    }
  }

  private static final class StubDecisionBackend implements SharedPrefixInferenceBackend {
    private static final int VOCAB = 32;
    boolean supportsSharedPrefixes = true;
    boolean returnEmptyTokens;
    boolean closed;
    int freezeCount;
    private final Map<Integer, Float> favoured = new HashMap<>();
    private final Set<InferenceSession> forked = new HashSet<>();

    void favour(int token, float logit) {
      favoured.put(token, logit);
    }

    private float[] logits() {
      float[] logits = new float[VOCAB];
      favoured.forEach((token, value) -> logits[token] = value);
      return logits;
    }

    @Override
    public String name() {
      return "stub-decision-backend";
    }

    @Override
    public boolean supportsSharedPrefixes() {
      return supportsSharedPrefixes;
    }

    @Override
    public boolean supportsHiddenState() {
      return true;
    }

    @Override
    public boolean supportsActivatedBranch() {
      return false;
    }

    @Override
    public int maxBatchSize() {
      return 4;
    }

    @Override
    public InferenceSession openSession() {
      return new StubSession();
    }

    @Override
    public float[] prefill(InferenceSession session, int[] tokens, int startPosition) {
      ((StubSession) session).position = startPosition + tokens.length;
      return logits();
    }

    @Override
    public float[] forward(InferenceSession session, int token, int position) {
      ((StubSession) session).position = position + 1;
      return logits();
    }

    @Override
    public float[] forward(int token, int position) {
      return logits();
    }

    @Override
    public LogitBatch forwardBatch(InferenceSession[] sessions, int[] tokens) {
      throw new UnsupportedOperationException();
    }

    @Override
    public SharedInferencePrefix freezePrefix(InferenceSession source) {
      freezeCount++;
      int length = ((StubSession) source).position;
      return new SharedInferencePrefix() {
        @Override
        public int checkpoint() {
          return length;
        }

        @Override
        public long sharedBytes() {
          return length * 64L;
        }
      };
    }

    @Override
    public InferenceSession fork(SharedInferencePrefix prefix, Branch branch) {
      StubSession session = new StubSession();
      session.position = prefix.checkpoint();
      forked.add(session);
      return session;
    }

    @Override
    public boolean sharesPrefixStorage(InferenceSession first, InferenceSession second) {
      return supportsSharedPrefixes && forked.contains(first) && forked.contains(second);
    }

    @Override
    public void rewind(InferenceSession session, int position) {
      ((StubSession) session).position = position;
    }

    @Override
    public void reset(InferenceSession session) {
      ((StubSession) session).position = 0;
    }

    @Override
    public ModelMetadata metadata() {
      throw new UnsupportedOperationException();
    }

    @Override
    public Tokenizer tokenizer() {
      return new Tokenizer() {
        @Override
        public int[] encode(String text) {
          if (returnEmptyTokens) {
            return new int[0];
          }
          String trimmed = text.strip();
          if (trimmed.length() == 1 && trimmed.charAt(0) >= 'A' && trimmed.charAt(0) <= 'Z') {
            return new int[] {10 + (trimmed.charAt(0) - 'A')};
          }
          return new int[] {1, 2, 3};
        }

        @Override
        public String decode(int[] tokens) {
          return "decoded";
        }

        @Override
        public String decode(int token) {
          return "t" + token;
        }

        @Override
        public int vocabSize() {
          return VOCAB;
        }

        @Override
        public int bosToken() {
          return 0;
        }

        @Override
        public int eosToken() {
          return 1;
        }
      };
    }

    @Override
    public void close() {
      closed = true;
    }
  }

  private static final class StubSession implements InferenceSession {
    int position;
    boolean closed;

    @Override
    public int checkpoint() {
      return position;
    }

    @Override
    public boolean isClosed() {
      return closed;
    }

    @Override
    public void close() {
      closed = true;
    }
  }
}
