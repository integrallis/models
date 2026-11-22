import java.nio.file.Files
import java.nio.file.Path
import java.util.zip.ZipFile

// models-spring-ai — Spring AI ChatModel + StreamingChatModel adapter

val springAiVersion = providers.gradleProperty("springAiVersion").getOrElse("2.0.0")

// --- API that only exists from Spring AI 2.1 (DecisionModel) -----------------------------------
//
// One artifact serves every supported Spring AI line, and the main source set is compiled against
// the baseline above so it cannot bind to anything newer. A class that implements a 2.1-only
// interface cannot be compiled against that baseline, so it lives in its own source set, compiled
// against 2.1, and its classes are packed into the same jar. On an older line nothing references
// them, so they are never loaded; this is how Spring Boot's own auto-configuration jar carries
// integrations for libraries that may be absent.
//
// The version is fixed here, not taken from -PspringAiVersion: these sources need at least 2.1
// whatever line the rest of the build is being tested against.
val springAi21Version = providers.gradleProperty("springAi21Version").getOrElse("2.1.0")

val springAi21: SourceSet by sourceSets.creating {
    compileClasspath += sourceSets["main"].output
    runtimeClasspath += sourceSets["main"].output
}
val springAi21Test: SourceSet by sourceSets.creating {
    compileClasspath += sourceSets["main"].output + springAi21.output
    runtimeClasspath += sourceSets["main"].output + springAi21.output
}
configurations[springAi21.implementationConfigurationName].extendsFrom(
    configurations["implementation"],
    configurations["api"],
)
// The test libraries, and backend-java for the real-model test, come from the ordinary test
// configuration. That also brings the baseline Spring AI; requesting 2.1 beside it resolves upward.
configurations[springAi21Test.implementationConfigurationName].extendsFrom(
    configurations["testImplementation"],
)
configurations[springAi21Test.runtimeOnlyConfigurationName].extendsFrom(
    configurations["testRuntimeOnly"],
)

dependencies {
    api(project(":models-runtime"))
    api(project(":models-router"))
    // Used only by the Spring AI 2.1 DecisionModel adapter, but declared here so it reaches the
    // published POM: the adapter's classes ship in this jar. It is the contracts-and-arithmetic
    // tier and brings nothing beyond models-api.
    implementation(project(":models-decisions"))

    add(springAi21.compileOnlyConfigurationName, "org.springframework.ai:spring-ai-model:$springAi21Version")
    add(springAi21Test.implementationConfigurationName, "org.springframework.ai:spring-ai-model:$springAi21Version")

    implementation("com.fasterxml.jackson.core:jackson-databind:${rootProject.extra["jacksonVersion"]}")

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
val configuredTinyBertRerankerPath =
    providers.systemProperty("models.fixtures.tinyBertReranker")
val configuredMxbaiRerankerDirectory =
    providers.systemProperty("models.fixtures.mxbaiRerankerDirectory")
val configuredActivatedLoraDirectory =
    providers.systemProperty("models.fixtures.activatedLoraDirectory")

tasks.withType<Test>().configureEach {
    configuredGptOssHuggingFaceDirectory.orNull?.let {
        systemProperty("models.fixtures.gptOssHuggingFaceDirectory", it)
    }
    systemProperty("models.fixtures.needle2Cact", needle2Fixture.get())
    systemProperty("models.fixtures.qwen317b", qwen317bFixture)
    systemProperty("models.fixtures.miniLm", miniLmFixture)
    systemProperty("models.fixtures.msMarcoReranker", msMarcoRerankerFixture)
    configuredTinyBertRerankerPath.orNull?.let {
        systemProperty("models.fixtures.tinyBertReranker", it)
    }
    configuredMxbaiRerankerDirectory.orNull?.let {
        systemProperty("models.fixtures.mxbaiRerankerDirectory", it)
    }
    configuredActivatedLoraDirectory.orNull?.let {
        systemProperty("models.fixtures.activatedLoraDirectory", it)
    }
}

tasks.register<Test>("qwen3ActivatedSpringAiToolCallingIntegrationTest") {
    description = "Run Spring AI's complete tool loop over the Qwen3 activated adapter"
    group = "verification"
    testClassesDirs = sourceSets["test"].output.classesDirs
    classpath = sourceSets["test"].runtimeClasspath
    useJUnitPlatform {
        includeTags("activated-integration")
    }
    filter {
        includeTestsMatching(
            "com.integrallis.models.spring.ai.Qwen3ActivatedSpringAiToolCallingIntegrationTest",
        )
    }
    dependsOn(project(":backend-java").tasks.named("downloadQwen317BQ80Model"))
    doFirst {
        val adapter = configuredActivatedLoraDirectory.orNull?.let(Path::of)
        require(adapter != null && Files.isDirectory(adapter)) {
            "-Dmodels.fixtures.activatedLoraDirectory must name the packaged adapter directory"
        }
    }
    outputs.upToDateWhen { false }
    maxParallelForks = 1
    maxHeapSize = "4g"
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

tasks.register<Test>("tinyBertStandardGgufSpringAiIntegrationTest") {
    description = "Run Spring AI reranking against a corrected standard-GGUF TinyBERT reranker"
    group = "verification"
    testClassesDirs = sourceSets["test"].output.classesDirs
    classpath = sourceSets["test"].runtimeClasspath
    useJUnitPlatform {
        includeTags("integration")
    }
    filter {
        includeTestsMatching(
            "com.integrallis.models.spring.ai.TinyBertStandardGgufSpringAiIntegrationTest",
        )
    }
    outputs.upToDateWhen { false }
    maxParallelForks = 1
    maxHeapSize = "1g"
}

tasks.register<Test>("mxbaiDebertaSpringAiIntegrationTest") {
    description = "Run Spring AI reranking against the pinned mxbai DeBERTa reranker"
    group = "verification"
    testClassesDirs = sourceSets["test"].output.classesDirs
    classpath = sourceSets["test"].runtimeClasspath
    useJUnitPlatform {
        includeTags("model-fixture")
    }
    filter {
        includeTestsMatching(
            "com.integrallis.models.spring.ai.MxbaiDebertaSpringAiRerankerIntegrationTest",
        )
    }
    outputs.upToDateWhen { false }
    maxParallelForks = 1
    maxHeapSize = "3g"
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

// --- Spring AI 2.1 source set: packaging and verification ----------------------------------------

tasks.named<Jar>("jar") {
    from(springAi21.output)
}
tasks.named<Jar>("sourcesJar") {
    from(springAi21.allSource)
}
tasks.named<Javadoc>("javadoc") {
    source(springAi21.allJava)
    // 2.1 first: it is a superset for these sources, and the baseline jar has no DecisionModel.
    classpath = springAi21.compileClasspath + classpath
}

val springAi21TestTask =
    tasks.register<Test>("springAi21Test") {
        description = "Tests the Spring AI 2.1 additions on Spring AI $springAi21Version"
        group = "verification"
        testClassesDirs = springAi21Test.output.classesDirs
        classpath = springAi21Test.runtimeClasspath
        // Like the module's ordinary test task: the real-model check is not part of `check`. It
        // takes minutes on a real model and would otherwise run whenever a fixture happens to be
        // on the machine. decisionModelIntegrationTest runs it.
        useJUnitPlatform {
            excludeTags("slow", "benchmark", "integration", "model-fixture")
        }
    }
// The adapter merged in #466 was excluded from the default build, so the published jar would not
// have contained it and nothing noticed. This fails the build if the class is not in the jar.
val verifyDecisionAdapterInJar =
    tasks.register("verifyDecisionAdapterInJar") {
        description = "Fails unless the published jar contains the Spring AI 2.1 DecisionModel adapter"
        group = "verification"
        val archive = tasks.named<Jar>("jar").flatMap { it.archiveFile }
        inputs.file(archive)
        doLast {
            val entry = "com/integrallis/models/spring/ai/decision/ModelsSpringAiDecisionModel.class"
            ZipFile(archive.get().asFile).use { zip ->
                if (zip.getEntry(entry) == null) {
                    throw GradleException("${archive.get().asFile.name} does not contain $entry")
                }
            }
        }
    }
tasks.named("check") {
    dependsOn(springAi21TestTask, verifyDecisionAdapterInJar)
}

tasks.register<Test>("decisionModelIntegrationTest") {
    description = "Run the Spring AI DecisionModel adapter end to end on the pinned Qwen3 1.7B artifact"
    group = "verification"
    testClassesDirs = springAi21Test.output.classesDirs
    classpath = springAi21Test.runtimeClasspath
    useJUnitPlatform {
        includeTags("integration")
    }
    filter {
        includeTestsMatching(
            "com.integrallis.models.spring.ai.decision.RealModelDecisionIntegrationTest",
        )
    }
    // Qwen3 1.7B is a llama-style family, so this is the run that exercises a real shared prefix.
    // -Dmodels.fixtures.decisionModel=<file> points it at another fixture in the same directory.
    systemProperty(
        "models.fixtures.directory",
        fixtureDirectory.orNull
            ?: Path.of(System.getProperty("user.home"), ".jvllm", "models").toString(),
    )
    systemProperty(
        "models.fixtures.decisionModel",
        providers.systemProperty("models.fixtures.decisionModel").getOrElse("Qwen3-1.7B-Q8_0.gguf"),
    )
    dependsOn(project(":backend-java").tasks.named("downloadQwen317BQ80Model"))
    outputs.upToDateWhen { false }
    maxParallelForks = 1
    maxHeapSize = "4g"
}
// The coverage gate has to see the adapter too, or the one class in this module that touches the
// newest API would be the one class nothing measures.
tasks.withType<JacocoReportBase>().configureEach {
    dependsOn(springAi21TestTask)
    executionData(springAi21TestTask.get())
    sourceSets(springAi21)
}

