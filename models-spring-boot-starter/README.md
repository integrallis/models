# Models Spring Boot Starter

Auto-configures a Models runtime as a Spring AI `ChatModel`.

```kotlin
dependencies {
    implementation("com.integrallis:models-spring-boot-starter:0.3.25")
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

For local RAG, add `com.integrallis:vectors-spring-boot-starter:0.1.9` and provide a Spring AI
`EmbeddingModel`. Vectors contributes `JavaVectorsVectorStore` while this starter contributes the
local chat model. Both starters are verified together on Spring AI 1.1 and 2.0.

With Spring Boot Actuator on the application classpath, the starter registers
Spring AI's standard chat and embedding meter handlers after Boot creates its
meter registry. Local prompt and completion counts are then available through
`gen_ai.client.token.usage` with `gen_ai.token.type=input` and
`gen_ai.token.type=output`; no hosted-provider starter or custom handler is
required.
