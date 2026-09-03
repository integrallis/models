import java.nio.file.Path

// models-spring-ai — Spring AI ChatModel + StreamingChatModel adapter

val springAiVersion = providers.gradleProperty("springAiVersion").getOrElse("2.0.0")

dependencies {
    api(project(":models-runtime"))
    api(project(":models-router"))

    implementation("com.fasterxml.jackson.core:jackson-databind:2.21.4")

    compileOnly("org.springframework.ai:spring-ai-model:$springAiVersion")
    compileOnly("org.springframework.ai:spring-ai-rag:$springAiVersion")
    testImplementation("org.springframework.ai:spring-ai-model:$springAiVersion")
    testImplementation("org.springframework.ai:spring-ai-client-chat:$springAiVersion")
    testImplementation("org.springframework.ai:spring-ai-rag:$springAiVersion")
    testImplementation(project(":backend-java"))
}

val configuredGptOssHuggingFaceDirectory =
    providers.systemProperty("models.fixtures.gptOssHuggingFaceDirectory")
val fixtureDirectory = providers.systemProperty("models.fixtures.directory")
val needle2Fixture =
    providers.systemProperty("models.fixtures.needle2Cact").orElse(
        providers.provider {
            Path.of(System.getProperty("user.home"), ".jvllm", "models", "needle2.cact").toString()
        },
    )
val qwen317bFixture =
    Path.of(
        fixtureDirectory.orNull
            ?: Path.of(System.getProperty("user.home"), ".jvllm", "models").toString(),
        "Qwen3-1.7B-Q8_0.gguf",
    ).toString()
val miniLmFixture =
    Path.of(
        fixtureDirectory.orNull
            ?: Path.of(System.getProperty("user.home"), ".jvllm", "models").toString(),
        "all-MiniLM-L6-v2-Q4_K_M.gguf",
    ).toString()
val msMarcoRerankerFixture =
    Path.of(
        fixtureDirectory.orNull
            ?: Path.of(System.getProperty("user.home"), ".jvllm", "models").toString(),
        "ms-marco-MiniLM-L-6-v2-q4_k-imatrix-g7c-f7.gguf",
    ).toString()

tasks.withType<Test>().configureEach {
    configuredGptOssHuggingFaceDirectory.orNull?.let {
        systemProperty("models.fixtures.gptOssHuggingFaceDirectory", it)
    }
    systemProperty("models.fixtures.needle2Cact", needle2Fixture.get())
    systemProperty("models.fixtures.qwen317b", qwen317bFixture)
    systemProperty("models.fixtures.miniLm", miniLmFixture)
    systemProperty("models.fixtures.msMarcoReranker", msMarcoRerankerFixture)
}

tasks.register<Test>("qwen3SpringAiToolCallingIntegrationTest") {
    description = "Run Spring AI's tool loop against the pinned Qwen3 1.7B artifact"
    group = "verification"
    testClassesDirs = sourceSets["test"].output.classesDirs
    classpath = sourceSets["test"].runtimeClasspath
    useJUnitPlatform {
        includeTags("integration")
    }
    filter {
        includeTestsMatching(
            "com.integrallis.models.spring.ai.Qwen3SpringAiToolCallingIntegrationTest",
        )
    }
    dependsOn(project(":backend-java").tasks.named("downloadQwen317BQ80Model"))
    outputs.upToDateWhen { false }
    maxParallelForks = 1
    maxHeapSize = "4g"
}

tasks.register<Test>("needle2SpringAiIntegrationTest") {
    description = "Reproduce Spring AI tool calling against the pinned official Needle 2 artifact"
    group = "verification"
    testClassesDirs = sourceSets["test"].output.classesDirs
    classpath = sourceSets["test"].runtimeClasspath
    useJUnitPlatform {
        includeTags("integration")
    }
    filter {
        includeTestsMatching(
            "com.integrallis.models.spring.ai.Needle2SpringAiToolCallingIntegrationTest",
        )
    }
    dependsOn(project(":backend-java").tasks.named("downloadNeedle2Cact"))
    outputs.upToDateWhen { false }
    maxParallelForks = 1
    maxHeapSize = "2g"
}

tasks.register<Test>("miniLmSpringAiIntegrationTest") {
    description = "Run Spring AI embeddings against the pinned All-MiniLM-L6-v2 GGUF"
    group = "verification"
    testClassesDirs = sourceSets["test"].output.classesDirs
    classpath = sourceSets["test"].runtimeClasspath
    useJUnitPlatform {
        includeTags("integration")
    }
    filter {
        includeTestsMatching(
            "com.integrallis.models.spring.ai.MiniLmSpringAiEmbeddingIntegrationTest",
        )
    }
    dependsOn(project(":backend-java").tasks.named("downloadAllMiniLmL6V2Q4KMModel"))
    outputs.upToDateWhen { false }
    maxParallelForks = 1
    maxHeapSize = "2g"
}

tasks.register<Test>("msMarcoRerankerSpringAiIntegrationTest") {
    description = "Run Spring AI document reranking against the pinned MS MARCO MiniLM model"
    group = "verification"
    testClassesDirs = sourceSets["test"].output.classesDirs
    classpath = sourceSets["test"].runtimeClasspath
    useJUnitPlatform {
        includeTags("integration")
    }
    filter {
        includeTestsMatching(
            "com.integrallis.models.spring.ai.MsMarcoSpringAiRerankerIntegrationTest",
        )
    }
    dependsOn(project(":backend-java").tasks.named("downloadMsMarcoMiniLmL6V2RerankerModel"))
    outputs.upToDateWhen { false }
    maxParallelForks = 1
    maxHeapSize = "2g"
}

tasks.register<Test>("gptOssSpringAiIntegrationTest") {
    description = "Run the Spring AI tool loop against the pinned official GPT-OSS 20B checkpoint"
    group = "verification"
    testClassesDirs = sourceSets["test"].output.classesDirs
    classpath = sourceSets["test"].runtimeClasspath
    useJUnitPlatform {
        includeTags("integration")
    }
    filter {
        includeTestsMatching(
            "com.integrallis.models.spring.ai.GptOssOfficialToolCallingIntegrationTest",
        )
    }
    outputs.upToDateWhen { false }
    maxParallelForks = 1
    maxHeapSize = "4g"
}
