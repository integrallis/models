import java.nio.file.Path

// models-langchain4j — LangChain4j chat model adapter

val langchain4jVersion = providers.gradleProperty("langchain4jVersion").getOrElse("1.17.2")

dependencies {
    api(project(":models-runtime"))
    api(project(":models-router"))

    implementation("com.fasterxml.jackson.core:jackson-databind:2.21.4")

    compileOnly("dev.langchain4j:langchain4j-core:$langchain4jVersion")
    testImplementation("dev.langchain4j:langchain4j-core:$langchain4jVersion")
    testImplementation("dev.langchain4j:langchain4j:$langchain4jVersion")
    testImplementation(project(":backend-java"))

    testImplementation(
        "com.integrallis:vectors-langchain4j:${providers.gradleProperty("vectorsVersion").get()}"
    )
}

val fixtureDirectory = providers.systemProperty("models.fixtures.directory")
val miniLmFixture =
    Path.of(
        fixtureDirectory.orNull
            ?: Path.of(System.getProperty("user.home"), ".jvllm", "models").toString(),
        "all-MiniLM-L6-v2-Q4_K_M.gguf",
    ).toString()
val qwen317bFixture =
    Path.of(
        fixtureDirectory.orNull
            ?: Path.of(System.getProperty("user.home"), ".jvllm", "models").toString(),
        "Qwen3-1.7B-Q8_0.gguf",
    ).toString()
val msMarcoRerankerFixture =
    Path.of(
        fixtureDirectory.orNull
            ?: Path.of(System.getProperty("user.home"), ".jvllm", "models").toString(),
        "ms-marco-MiniLM-L-6-v2-q4_k-imatrix-g7c-f7.gguf",
    ).toString()

tasks.withType<Test>().configureEach {
    systemProperty("models.fixtures.miniLm", miniLmFixture)
    systemProperty("models.fixtures.qwen317b", qwen317bFixture)
    systemProperty("models.fixtures.msMarcoReranker", msMarcoRerankerFixture)
}

tasks.register<Test>("qwen3LangChain4jToolCallingIntegrationTest") {
    description = "Run LangChain4j's tool loop against the pinned Qwen3 1.7B artifact"
    group = "verification"
    testClassesDirs = sourceSets["test"].output.classesDirs
    classpath = sourceSets["test"].runtimeClasspath
    useJUnitPlatform {
        includeTags("integration")
    }
    filter {
        includeTestsMatching(
            "com.integrallis.models.langchain4j.Qwen3LangChain4jToolCallingIntegrationTest",
        )
    }
    dependsOn(project(":backend-java").tasks.named("downloadQwen317BQ80Model"))
    outputs.upToDateWhen { false }
    maxParallelForks = 1
    maxHeapSize = "4g"
}

tasks.register<Test>("miniLmLangChain4jIntegrationTest") {
    description = "Run LangChain4j embeddings against the pinned All-MiniLM-L6-v2 GGUF"
    group = "verification"
    testClassesDirs = sourceSets["test"].output.classesDirs
    classpath = sourceSets["test"].runtimeClasspath
    useJUnitPlatform {
        includeTags("integration")
    }
    filter {
        includeTestsMatching(
            "com.integrallis.models.langchain4j.MiniLmLangChain4jEmbeddingIntegrationTest",
        )
    }
    dependsOn(project(":backend-java").tasks.named("downloadAllMiniLmL6V2Q4KMModel"))
    outputs.upToDateWhen { false }
    maxParallelForks = 1
    maxHeapSize = "2g"
}

tasks.register<Test>("msMarcoRerankerLangChain4jIntegrationTest") {
    description = "Run LangChain4j scoring against the pinned MS MARCO MiniLM reranker"
    group = "verification"
    testClassesDirs = sourceSets["test"].output.classesDirs
    classpath = sourceSets["test"].runtimeClasspath
    useJUnitPlatform {
        includeTags("integration")
    }
    filter {
        includeTestsMatching(
            "com.integrallis.models.langchain4j.MsMarcoLangChain4jRerankerIntegrationTest",
        )
    }
    dependsOn(project(":backend-java").tasks.named("downloadMsMarcoMiniLmL6V2RerankerModel"))
    outputs.upToDateWhen { false }
    maxParallelForks = 1
    maxHeapSize = "2g"
}
