import java.nio.file.Path
import java.util.Properties

// backend-java - GGUF parser, vectors-backed inference kernels, and KV cache

data class ModelFixture(
    val taskName: String,
    val id: String,
    val displayName: String,
    val slow: Boolean,
    val backend: String,
)

val fixtureProperties =
    Properties().apply {
        file("src/test/resources/model-fixtures.properties").inputStream().use { load(it) }
    }

fun modelFixture(taskName: String, id: String): ModelFixture {
    val encoded =
        requireNotNull(fixtureProperties.getProperty(id)) {
            "No pinned model fixture is declared for $id"
        }
    val fields = encoded.split('|')
    require(fields.size in 15..16) {
        "Pinned model fixture $id has ${fields.size} fields; expected 15 or 16"
    }
    return ModelFixture(
        taskName = taskName,
        id = id,
        displayName = fields[0],
        slow = fields[14].toBooleanStrict(),
        backend = fields[4],
    )
}

val modelFixtures =
    listOf(
        modelFixture(
            "downloadQwen306BQ40Model",
            "qwen3_0_6b_q4_0",
        ),
        modelFixture(
            "downloadQwen317BQ80Model",
            "qwen3_1_7b_q8_0",
        ),
        modelFixture(
            "downloadQwen3Embedding06BQ80Model",
            "qwen3_embedding_0_6b_q8_0",
        ),
        modelFixture(
            "downloadQwen38BQ4KMModel",
            "qwen3_8b_q4_k_m",
        ),
        modelFixture(
            "downloadGemma426BA4BQ4KMModel",
            "gemma4_26b_a4b_it_q4_k_m",
        ),
        modelFixture(
            "downloadQwen25Coder05BQ40Model",
            "qwen2_5_coder_0_5b_instruct_q4_0",
        ),
        modelFixture(
            "downloadQwen25Coder05BQ80Model",
            "qwen2_5_coder_0_5b_instruct_q8_0",
        ),
        modelFixture(
            "downloadQwen25Coder15BQ40Model",
            "qwen2_5_coder_1_5b_instruct_q4_0",
        ),
        modelFixture(
            "downloadQwen25Coder15BQ80Model",
            "qwen2_5_coder_1_5b_instruct_q8_0",
        ),
        modelFixture(
            "downloadQwen25Coder3BQ40Model",
            "qwen2_5_coder_3b_instruct_q4_0",
        ),
        modelFixture(
            "downloadQwen25Coder7BQ40Model",
            "qwen2_5_coder_7b_instruct_q4_0",
        ),
        modelFixture(
            "downloadQwen25Math15BQ4KMModel",
            "qwen2_5_math_1_5b_instruct_q4_k_m",
        ),
        modelFixture(
            "downloadSmolLm2360MQ80Model",
            "smollm2_360m_instruct_q8_0",
        ),
        modelFixture(
            "downloadTinyLlama11BChatV10Q40Model",
            "tinyllama_1_1b_chat_v1_0_q4_0",
        ),
        modelFixture(
            "downloadDeepSeekCoder13BQ4KMModel",
            "deepseek_coder_1_3b_instruct_q4_k_m",
        ),
        modelFixture(
            "downloadDeepSeekCoder67BQ4KMModel",
            "deepseek_coder_6_7b_instruct_q4_k_m",
        ),
        modelFixture(
            "downloadDeepSeekR1DistillQwen7BQ4KMModel",
            "deepseek_r1_distill_qwen_7b_q4_k_m",
        ),
        modelFixture(
            "downloadHuatuoGptO17BQ4KMModel",
            "huatuogpt_o1_7b_q4_k_m",
        ),
        modelFixture(
            "downloadSqlCoder7B2Q5KMModel",
            "sqlcoder_7b_2_q5_k_m",
        ),
        modelFixture(
            "downloadSmolLm33BQ4KMModel",
            "smollm3_3b_q4_k_m",
        ),
        modelFixture(
            "downloadMiniCpm51BQ4KMModel",
            "minicpm5_1b_q4_k_m",
        ),
        modelFixture(
            "downloadGemma31BQ4KMModel",
            "bartowski_google_gemma_3_1b_it_gguf_q4_k_m",
        ),
        modelFixture(
            "downloadEuroLlm17BQ4KMModel",
            "eurollm_1_7b_instruct_q4_k_m",
        ),
        modelFixture(
            "downloadFinR17BQ4KMModel",
            "fin_r1_7b_q4_k_m",
        ),
    )

val needle2CactFixture = modelFixture("downloadNeedle2Cact", "needle2_cq2")
val qwen35GgufFixture =
    modelFixture("downloadQwen3508BQ4KMGguf", "qwen3_5_0_8b_q4_k_m")
val qwen354BGgufFixture =
    modelFixture("downloadQwen354BQ4KMGguf", "qwen3_5_4b_q4_k_m")

dependencies {
    api(project(":models-api"))

    implementation("com.integrallis:vectors-core:${providers.gradleProperty("vectorsVersion").get()}")
    implementation("com.fasterxml.jackson.core:jackson-core:2.21.4")

    testImplementation(project(":models-runtime"))
}

val fixtureDirectory = providers.systemProperty("models.fixtures.directory")
val configuredNeedle2CactPath = providers.systemProperty("models.fixtures.needle2Cact")
val configuredSafetensorsReferencePath =
    providers.systemProperty("models.fixtures.safetensorsReference")
val configuredQwen25HuggingFaceDirectory =
    providers.systemProperty("models.fixtures.qwen25HuggingFaceDirectory")
val configuredGptOssHuggingFaceDirectory =
    providers.systemProperty("models.fixtures.gptOssHuggingFaceDirectory")
val configuredGptOssOracleLogits =
    providers.systemProperty("models.fixtures.gptOssOracleLogits")

tasks.withType<Test>().configureEach {
    fixtureDirectory.orNull?.let { systemProperty("models.fixtures.directory", it) }
    configuredNeedle2CactPath.orNull?.let {
        systemProperty("models.fixtures.needle2Cact", it)
    }
    configuredSafetensorsReferencePath.orNull?.let {
        systemProperty("models.fixtures.safetensorsReference", it)
    }
    configuredQwen25HuggingFaceDirectory.orNull?.let {
        systemProperty("models.fixtures.qwen25HuggingFaceDirectory", it)
    }
    configuredGptOssHuggingFaceDirectory.orNull?.let {
        systemProperty("models.fixtures.gptOssHuggingFaceDirectory", it)
    }
    configuredGptOssOracleLogits.orNull?.let {
        systemProperty("models.fixtures.gptOssOracleLogits", it)
    }
}

tasks.withType<JavaExec>().configureEach {
    fixtureDirectory.orNull?.let { systemProperty("models.fixtures.directory", it) }
}

tasks.register<JavaExec>(needle2CactFixture.taskName) {
    description = "Download and verify the pinned ${needle2CactFixture.displayName} parser fixture"
    group = "model acquisition"
    dependsOn(tasks.named("testClasses"))
    classpath = sourceSets["test"].runtimeClasspath
    mainClass.set("com.integrallis.models.backend.purejava.fixture.ModelFixtureInstallerCli")
    args(needle2CactFixture.id)
}

tasks.register<JavaExec>(qwen35GgufFixture.taskName) {
    description = "Download and verify the pinned ${qwen35GgufFixture.displayName} parser fixture"
    group = "model acquisition"
    dependsOn(tasks.named("testClasses"))
    classpath = sourceSets["test"].runtimeClasspath
    mainClass.set("com.integrallis.models.backend.purejava.fixture.ModelFixtureInstallerCli")
    args(qwen35GgufFixture.id)
}

tasks.register<JavaExec>(qwen354BGgufFixture.taskName) {
    description = "Download and verify the pinned ${qwen354BGgufFixture.displayName} grouped-GDN fixture"
    group = "model acquisition"
    dependsOn(tasks.named("testClasses"))
    classpath = sourceSets["test"].runtimeClasspath
    mainClass.set("com.integrallis.models.backend.purejava.fixture.ModelFixtureInstallerCli")
    args(qwen354BGgufFixture.id)
}

tasks.register<Test>("qwen35GgufLayoutTest") {
    description = "Verify the pinned Qwen3.5 0.8B hybrid-attention GGUF layout"
    group = "verification"
    testClassesDirs = sourceSets["test"].output.classesDirs
    classpath = sourceSets["test"].runtimeClasspath
    useJUnitPlatform()
    filter {
        includeTestsMatching(
            "com.integrallis.models.backend.purejava.qwen35.Qwen35GgufLayoutTest",
        )
        includeTestsMatching(
            "com.integrallis.models.backend.purejava.qwen35.Qwen35ForwardPassIntegrationTest",
        )
    }
    dependsOn(tasks.named(qwen35GgufFixture.taskName))
    outputs.upToDateWhen { false }
}

tasks.register<Test>("qwen35GroupedGdnCompatibilityTest") {
    description = "Verify the pinned Qwen3.5 4B grouped-GDN execution path"
    group = "verification"
    testClassesDirs = sourceSets["test"].output.classesDirs
    classpath = sourceSets["test"].runtimeClasspath
    useJUnitPlatform()
    filter {
        includeTestsMatching(
            "com.integrallis.models.backend.purejava.qwen35.Qwen35GroupedGdnIntegrationTest",
        )
    }
    dependsOn(tasks.named(qwen354BGgufFixture.taskName))
    outputs.upToDateWhen { false }
}

tasks.register<JavaExec>("qwen35LinearStateSnapshotExperiment") {
    description = "Compare Qwen3.5 prefix replay with bounded Java linear-state snapshot restore"
    group = "verification"
    dependsOn(tasks.named(qwen35GgufFixture.taskName))
    classpath = sourceSets["test"].runtimeClasspath
    mainClass.set(
        "com.integrallis.models.backend.purejava.qwen35.Qwen35LinearStateSnapshotExperiment",
    )
    jvmArgs("--add-modules", "jdk.incubator.vector")
    args(
        providers.gradleProperty("qwen35.snapshot.prefixTokens").getOrElse("32"),
        providers.gradleProperty("qwen35.snapshot.warmups").getOrElse("1"),
        providers.gradleProperty("qwen35.snapshot.iterations").getOrElse("5"),
    )
    maxHeapSize = "4g"
}

tasks.register<Test>("needle2CactCompatibilityTest") {
    description = "Verify the pinned official Needle 2 .cact artifact and tokenizer"
    group = "verification"
    testClassesDirs = sourceSets["test"].output.classesDirs
    classpath = sourceSets["test"].runtimeClasspath
    useJUnitPlatform()
    filter {
        includeTestsMatching(
            "com.integrallis.models.backend.purejava.cact.CactParserTest.parsesPinnedOfficialNeedle2ArtifactWhenProvided",
        )
        includeTestsMatching(
            "com.integrallis.models.backend.purejava.cact.CactTokenizerTest.matchesPinnedOfficialNeedle2ReferenceValuesWhenProvided",
        )
        includeTestsMatching(
            "com.integrallis.models.backend.purejava.cact.CactTokenizerTest.matchesPinnedNeedleReferenceForRobotToolPromptWhenProvided",
        )
        includeTestsMatching(
            "com.integrallis.models.backend.purejava.cact.CactNeedle2LayoutTest.matchesPinnedOfficialNeedle2TensorLayoutWhenProvided",
        )
        includeTestsMatching(
            "com.integrallis.models.backend.purejava.cact.CactCqMatrixTest.multipliesPinnedOfficialCq4AndCq2MatricesWhenProvided",
        )
        includeTestsMatching(
            "com.integrallis.models.backend.purejava.cact.Needle2ForwardPassTest.matchesPinnedNeedleJaxReferenceForBosTokenWhenProvided",
        )
        includeTestsMatching(
            "com.integrallis.models.backend.purejava.cact.Needle2ForwardPassTest.advancingWithoutLogitsPreservesTheNextTokenResultWhenProvided",
        )
        includeTestsMatching(
            "com.integrallis.models.backend.purejava.Needle2BackendIntegrationTest.loadsCactThroughThePublicBackendContractWhenProvided",
        )
        includeTestsMatching(
            "com.integrallis.models.backend.purejava.Needle2BackendIntegrationTest.generatesTheOfficialWeatherToolCallWhenProvided",
        )
        includeTestsMatching(
            "com.integrallis.models.backend.purejava.Needle2BackendIntegrationTest.generatesTheOfficialRobotToolSequenceWithSchemaGrammarWhenProvided",
        )
    }
    val configuredArtifact = configuredNeedle2CactPath.orNull
    val directory =
        fixtureDirectory.orNull
            ?: Path.of(System.getProperty("user.home"), ".jvllm", "models").toString()
    systemProperty(
        "models.fixtures.needle2Cact",
        configuredArtifact ?: Path.of(directory, "needle2.cact").toString(),
    )
    if (configuredArtifact == null) {
        dependsOn(tasks.named(needle2CactFixture.taskName))
    }
    outputs.upToDateWhen { false }
}

modelFixtures.forEach { fixture ->
    tasks.register<JavaExec>(fixture.taskName) {
        description =
            "Download and verify the pinned ${fixture.displayName} test fixture"
        dependsOn(tasks.named("testClasses"))
        classpath = sourceSets["test"].runtimeClasspath
        mainClass.set(
            "com.integrallis.models.backend.purejava.fixture.ModelFixtureInstallerCli",
        )
        args(fixture.id)
    }

    val runtimeSuffix =
        when (fixture.backend) {
            "pure-java" -> ""
            "rust-ffm" -> "Native"
            else -> error("No public task suffix is defined for backend '${fixture.backend}'")
        }
    val publicTaskName =
        fixture.taskName.removePrefix("download").removeSuffix("Model") + runtimeSuffix
    rootProject.project(":models").tasks.register(publicTaskName) {
        group = "model acquisition"
        description = "Download and verify ${fixture.displayName}"
        dependsOn(tasks.named(fixture.taskName))
    }
}

tasks.named<Test>("integrationTest") {
    // A preceding --tests filter can leave a successful but partial report that Gradle otherwise
    // considers up to date. An explicit real-model integration run must execute the full suite.
    outputs.upToDateWhen { false }
    dependsOn(
        modelFixtures
            .filterNot(ModelFixture::slow)
            .map { tasks.named(it.taskName) },
    )
}

tasks.register<Test>("qwen25HuggingFaceIntegrationTest") {
    description = "Run the pinned Qwen 2.5 0.5B Hugging Face Safetensors compatibility test"
    group = "verification"
    testClassesDirs = sourceSets["test"].output.classesDirs
    classpath = sourceSets["test"].runtimeClasspath
    useJUnitPlatform {
        includeTags("integration")
    }
    filter {
        includeTestsMatching(
            "com.integrallis.models.backend.purejava.Qwen2HuggingFaceBackendIntegrationTest",
        )
    }
    maxHeapSize = "4g"
}

tasks.register<Test>("gptOssHuggingFaceIntegrationTest") {
    description = "Run the pinned GPT-OSS 20B MXFP4 Safetensors compatibility test"
    group = "verification"
    testClassesDirs = sourceSets["test"].output.classesDirs
    classpath = sourceSets["test"].runtimeClasspath
    useJUnitPlatform {
        includeTags("integration")
    }
    filter {
        includeTestsMatching(
            "com.integrallis.models.backend.purejava.GptOssHuggingFaceBackendIntegrationTest",
        )
    }
    outputs.upToDateWhen { false }
    maxParallelForks = 1
    maxHeapSize = "4g"
}

tasks.register<Test>("qwen306BQ40IntegrationTest") {
    description = "Run the pinned Qwen3 0.6B Q4_0 pure-Java integration tests"
    group = "verification"
    testClassesDirs = sourceSets["test"].output.classesDirs
    classpath = sourceSets["test"].runtimeClasspath
    useJUnitPlatform {
        includeTags("integration")
    }
    filter {
        includeTestsMatching(
            "com.integrallis.models.backend.purejava.Qwen3ModelFixtureIntegrationTest.*Q40*",
        )
        includeTestsMatching(
            "com.integrallis.models.backend.purejava.llama.Qwen3BatchedPrefillIntegrationTest",
        )
        includeTestsMatching(
            "com.integrallis.models.backend.purejava.tokenizer.GgufTokenizerIntegrationTest",
        )
    }
    dependsOn(tasks.named("downloadQwen306BQ40Model"))
    outputs.upToDateWhen { false }
    maxParallelForks = 1
    maxHeapSize = "4g"
}

tasks.register<Test>("qwen3EmbeddingIntegrationTest") {
    description = "Run the pinned Qwen3-Embedding 0.6B pure-Java embedding integration tests"
    group = "verification"
    testClassesDirs = sourceSets["test"].output.classesDirs
    classpath = sourceSets["test"].runtimeClasspath
    useJUnitPlatform {
        includeTags("integration")
    }
    filter {
        includeTestsMatching(
            "com.integrallis.models.backend.purejava.Qwen3EmbeddingModelFixtureIntegrationTest",
        )
    }
    dependsOn(tasks.named("downloadQwen3Embedding06BQ80Model"))
    outputs.upToDateWhen { false }
    maxParallelForks = 1
    maxHeapSize = "4g"
}

tasks.register<Test>("qwen25Math15BIntegrationTest") {
    description = "Run the pinned Qwen2.5-Math 1.5B model integration tests"
    group = "verification"
    testClassesDirs = sourceSets["test"].output.classesDirs
    classpath = sourceSets["test"].runtimeClasspath
    useJUnitPlatform {
        includeTags("integration")
    }
    filter {
        includeTestsMatching(
            "com.integrallis.models.backend.purejava.Qwen25MathModelFixtureIntegrationTest",
        )
    }
    dependsOn(tasks.named("downloadQwen25Math15BQ4KMModel"))
    outputs.upToDateWhen { false }
    maxParallelForks = 1
    maxHeapSize = "4g"
}

tasks.register<Test>("euroLlm17BIntegrationTest") {
    description = "Run the pinned EuroLLM 1.7B pure-Java integration tests"
    group = "verification"
    testClassesDirs = sourceSets["test"].output.classesDirs
    classpath = sourceSets["test"].runtimeClasspath
    useJUnitPlatform {
        includeTags("integration")
    }
    filter {
        includeTestsMatching(
            "com.integrallis.models.backend.purejava.EuroLlmModelFixtureIntegrationTest",
        )
    }
    dependsOn(tasks.named("downloadEuroLlm17BQ4KMModel"))
    outputs.upToDateWhen { false }
    maxParallelForks = 1
    maxHeapSize = "4g"
}

tasks.named<Test>("slowTest") {
    dependsOn(
        modelFixtures
            .filter(ModelFixture::slow)
            .map { tasks.named(it.taskName) },
    )
    maxParallelForks = 1
    maxHeapSize = "8g"
}

data class LargeModelTest(
    val taskName: String,
    val displayName: String,
    val fixtureTaskName: String,
    val testClassName: String,
)

listOf(
    LargeModelTest(
        "qwen25Coder7BSlowTest",
        "Qwen2.5-Coder 7B",
        "downloadQwen25Coder7BQ40Model",
        "com.integrallis.models.backend.purejava.Qwen25CoderLargeModelFixtureSlowTest",
    ),
    LargeModelTest(
        "deepSeekCoder67BSlowTest",
        "DeepSeek-Coder 6.7B",
        "downloadDeepSeekCoder67BQ4KMModel",
        "com.integrallis.models.backend.purejava.DeepSeekCoderLargeModelFixtureSlowTest",
    ),
    LargeModelTest(
        "qwen38BSlowTest",
        "Qwen3 8B",
        "downloadQwen38BQ4KMModel",
        "com.integrallis.models.backend.purejava.Qwen3LargeModelFixtureSlowTest",
    ),
    LargeModelTest(
        "deepSeekR1DistillQwen7BSlowTest",
        "DeepSeek-R1-Distill-Qwen-7B",
        "downloadDeepSeekR1DistillQwen7BQ4KMModel",
        "com.integrallis.models.backend.purejava.DeepSeekR1DistillQwenLargeModelFixtureSlowTest",
    ),
    LargeModelTest(
        "huatuoGptO17BSlowTest",
        "HuatuoGPT-o1-7B",
        "downloadHuatuoGptO17BQ4KMModel",
        "com.integrallis.models.backend.purejava.HuatuoGptO1LargeModelFixtureSlowTest",
    ),
    LargeModelTest(
        "sqlCoder7B2SlowTest",
        "SQLCoder-7B-2",
        "downloadSqlCoder7B2Q5KMModel",
        "com.integrallis.models.backend.purejava.SqlCoderLargeModelFixtureSlowTest",
    ),
    LargeModelTest(
        "smolLm33BSlowTest",
        "SmolLM3 3B",
        "downloadSmolLm33BQ4KMModel",
        "com.integrallis.models.backend.purejava.SmolLm3ModelFixtureSlowTest",
    ),
    LargeModelTest(
        "finR17BSlowTest",
        "Fin-R1 7B",
        "downloadFinR17BQ4KMModel",
        "com.integrallis.models.backend.purejava.FinR1LargeModelFixtureSlowTest",
    ),
    LargeModelTest(
        "gemma426BA4BSlowTest",
        "Gemma 4 26B-A4B IT",
        "downloadGemma426BA4BQ4KMModel",
        "com.integrallis.models.backend.purejava.gemma4.Gemma4LargeModelFixtureSlowTest",
    ),
).forEach { largeModelTest ->
    tasks.register<Test>(largeModelTest.taskName) {
        group = "verification"
        description =
            "Resolve, verify, and run the ${largeModelTest.displayName} pure-Java slow test"
        dependsOn(tasks.named(largeModelTest.fixtureTaskName))
        testClassesDirs = sourceSets["test"].output.classesDirs
        classpath = sourceSets["test"].runtimeClasspath
        useJUnitPlatform {
            includeTags("slow")
        }
        filter {
            includeTestsMatching(largeModelTest.testClassName)
        }
        outputs.upToDateWhen { false }
        maxParallelForks = 1
        maxHeapSize = if (largeModelTest.taskName == "gemma426BA4BSlowTest") "4g" else "8g"
        if (largeModelTest.taskName == "gemma426BA4BSlowTest") {
            systemProperty("models.gemma4.expertCacheSlots", "8")
            systemProperty("models.purejava.maxContextLength", "128")
        }
    }
}
