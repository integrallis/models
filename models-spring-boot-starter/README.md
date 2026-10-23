# Models Spring Boot Starter

Auto-configures a Models runtime as a Spring AI `ChatModel`.

```kotlin
val modelsVersion = providers.gradleProperty("modelsVersion").get()

dependencies {
    implementation("com.integrallis:models-spring-boot-starter:$modelsVersion")
}
```

The application provides one `TextGenerationModel` or `InferenceBackend` bean. The starter registers
`ModelsSpringAiChatModel` under the bean name `modelsChatModel`; other Spring AI chat models can
coexist under their own names.

```java
@Configuration
class LocalModelConfiguration {
    @Bean(destroyMethod = "close")
    TextGenerationModel localModel() {
        return openLocalModel();
    }
}
```

```yaml
integrallis:
  models:
    chat-template: chatml-no-think
    sampling:
      temperature: 0.0
      max-tokens: 128
```

## Naming an installed catalog model instead

If `backend-java` is on the classpath, the model can come from configuration instead of a bean. The
value is a **catalog id** — one that an installed `ModelCatalogProvider` reports, such as a ModelJars
artifact already on the classpath — not a file path and not a Maven coordinate:

```yaml
integrallis:
  models:
    catalog-model: qwen3_1_7b_q8_0
```

The starter resolves it through the installed catalogs, opens it with the Java decoder (native shims
picked up underneath as for any other load), and closes it with the context. The artifact stays a
pinned build dependency; nothing is downloaded at startup, and an id whose weights are not on the
machine is refused saying so rather than fetched.

An application bean always wins: with a `TextGenerationModel` or `InferenceBackend` of your own, the
property is ignored. Two installed catalogs holding the same id is refused rather than resolved.

## `ChatClient.Builder`

With `spring-ai-client-chat` on the classpath, the starter also contributes a
`ChatClient.Builder` over that chat model, under the bean name `modelsChatClientBuilder`, with every
`ChatClientBuilderCustomizer` bean applied in order:

```java
@Bean
ChatClient chatClient(ChatClient.Builder builder) {
    return builder.defaultSystem("You are concise.").build();
}
```

It is **prototype-scoped**, so each injection point gets its own builder and one consumer's defaults
cannot leak into another's. A `ChatClient.Builder` the application or another auto-configuration
supplies is left alone, and nothing is registered when there is no local chat model.

For local RAG, add `com.integrallis:vectors-spring-boot-starter:0.1.30` and provide a Spring AI
`EmbeddingModel`. Vectors contributes `JavaVectorsVectorStore` while this starter contributes the
local chat model. Both starters are verified together on Spring AI 1.1 and 2.0.

With Spring Boot Actuator on the application classpath, the starter registers
Spring AI's standard chat and embedding meter handlers after Boot creates its
meter registry. Local prompt and completion counts are then available through
`gen_ai.client.token.usage` with `gen_ai.token.type=input` and
`gen_ai.token.type=output`; no hosted-provider starter or custom handler is
required.
