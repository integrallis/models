// models-backend-purejava - GGUF parser, vectors-backed inference kernels, and KV cache

data class ModelFixture(
    val taskName: String,
    val displayName: String,
    val sourceId: String,
    val versionRange: String,
    val variant: String,
    val capability: String,
    val slow: Boolean = false,
)

val modelFixtures =
    listOf(
        ModelFixture(
            "downloadQwen306BQ40Model",
            "Qwen3 0.6B Q4_0",
            "hf://ggml-org/Qwen3-0.6B-GGUF",
            "[3.0.0,4.0.0)",
            "q4_0",
            "text-generation",
        ),
        ModelFixture(
            "downloadQwen317BQ80Model",
            "Qwen3 1.7B Q8_0",
            "hf://Qwen/Qwen3-1.7B-GGUF",
            "[3.0.0,4.0.0)",
            "q8_0",
            "text-generation",
        ),
        ModelFixture(
            "downloadQwen38BQ4KMModel",
            "Qwen3 8B Q4_K_M",
            "hf://Qwen/Qwen3-8B-GGUF",
            "[3.0.0,4.0.0)",
            "q4_k_m",
            "chat",
            slow = true,
        ),
        ModelFixture(
            "downloadQwen25Coder05BQ40Model",
            "Qwen2.5-Coder 0.5B Q4_0",
            "hf://Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF",
            "[2.5.0,3.0.0)",
            "q4_0",
            "code-completion",
        ),
        ModelFixture(
            "downloadQwen25Coder05BQ80Model",
            "Qwen2.5-Coder 0.5B Q8_0",
            "hf://Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF",
            "[2.5.0,3.0.0)",
            "q8_0",
            "code-completion",
        ),
        ModelFixture(
            "downloadQwen25Coder15BQ40Model",
            "Qwen2.5-Coder 1.5B Q4_0",
            "hf://Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF",
            "[2.5.0,3.0.0)",
            "q4_0",
            "code-completion",
        ),
        ModelFixture(
            "downloadQwen25Coder15BQ80Model",
            "Qwen2.5-Coder 1.5B Q8_0",
            "hf://Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF",
            "[2.5.0,3.0.0)",
            "q8_0",
            "code-completion",
        ),
        ModelFixture(
            "downloadQwen25Coder3BQ40Model",
            "Qwen2.5-Coder 3B Q4_0",
            "hf://Qwen/Qwen2.5-Coder-3B-Instruct-GGUF",
            "[2.5.0,3.0.0)",
            "q4_0",
            "code-completion",
        ),
        ModelFixture(
            "downloadQwen25Coder7BQ40Model",
            "Qwen2.5-Coder 7B Q4_0",
            "hf://Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
            "[2.5.0,3.0.0)",
            "q4_0",
            "code-completion",
            slow = true,
        ),
        ModelFixture(
            "downloadQwen25Math15BQ4KMModel",
            "Qwen2.5-Math 1.5B Instruct Q4_K_M",
            "hf://bartowski/Qwen2.5-Math-1.5B-Instruct-GGUF",
            "[2.5.0,3.0.0)",
            "q4_k_m",
            "math",
        ),
        ModelFixture(
            "downloadSmolLm2360MQ80Model",
            "SmolLM2 360M Q8_0",
            "hf://HuggingFaceTB/SmolLM2-360M-Instruct-GGUF",
            "[2.0.0,3.0.0)",
            "q8_0",
            "chat",
        ),
        ModelFixture(
            "downloadTinyLlama11BChatV10Q40Model",
            "TinyLlama 1.1B Chat v1.0 Q4_0",
            "hf://TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
            "[1.0.0,2.0.0)",
            "q4_0",
            "chat",
        ),
        ModelFixture(
            "downloadDeepSeekCoder13BQ4KMModel",
            "DeepSeek-Coder 1.3B Instruct Q4_K_M",
            "hf://TheBloke/deepseek-coder-1.3b-instruct-GGUF",
            "[1.3.0,2.0.0)",
            "q4_k_m",
            "code-completion",
        ),
        ModelFixture(
            "downloadDeepSeekCoder67BQ4KMModel",
            "DeepSeek-Coder 6.7B Instruct Q4_K_M",
            "hf://TheBloke/deepseek-coder-6.7B-instruct-GGUF",
            "[6.7.0,7.0.0)",
            "q4_k_m",
            "code-completion",
            slow = true,
        ),
        ModelFixture(
            "downloadDeepSeekR1DistillQwen7BQ4KMModel",
            "DeepSeek-R1-Distill-Qwen-7B Q4_K_M",
            "hf://bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF",
            "[1.0.0,2.0.0)",
            "q4_k_m",
            "reasoning",
            slow = true,
        ),
        ModelFixture(
            "downloadHuatuoGptO17BQ4KMModel",
            "HuatuoGPT-o1-7B Q4_K_M",
            "hf://bartowski/HuatuoGPT-o1-7B-GGUF",
            "[1.0.0,2.0.0)",
            "q4_k_m",
            "medical-reasoning",
            slow = true,
        ),
        ModelFixture(
            "downloadSqlCoder7B2Q5KMModel",
            "SQLCoder-7B-2 Q5_K_M",
            "hf://defog/sqlcoder-7b-2",
            "[2.0.0,3.0.0)",
            "q5_k_m",
            "text-to-sql",
            slow = true,
        ),
        ModelFixture(
            "downloadSmolLm33BQ4KMModel",
            "SmolLM3 3B Q4_K_M",
            "hf://ggml-org/SmolLM3-3B-GGUF",
            "[3.0.0,4.0.0)",
            "q4_k_m",
            "text-generation",
            slow = true,
        ),
        ModelFixture(
            "downloadMiniCpm51BQ4KMModel",
            "MiniCPM5 1B Q4_K_M",
            "hf://openbmb/MiniCPM5-1B-GGUF",
            "[5.0.0,6.0.0)",
            "q4_k_m",
            "text-generation",
        ),
    )

dependencies {
    api(project(":models-api"))
    api("org.modeljars:modeljars-core:0.1.0-SNAPSHOT")

    implementation("com.integrallis:vectors-core:0.1.0-SNAPSHOT")

    testImplementation(project(":models-runtime"))
    testRuntimeOnly("org.modeljars:modeljars-catalog:0.1.0-SNAPSHOT")
}

modelFixtures.forEach { fixture ->
    tasks.register<JavaExec>(fixture.taskName) {
        group = "verification"
        description =
            "Resolve, download, and verify the ${fixture.displayName} fixture through ModelJars"
        dependsOn(tasks.named("testClasses"))
        classpath = sourceSets["test"].runtimeClasspath
        mainClass.set("org.modeljars.ModelJarInstallerCli")
        args(
            fixture.sourceId,
            "--version",
            fixture.versionRange,
            "--variant",
            fixture.variant,
            "--backend",
            "pure-java",
            "--capability",
            fixture.capability,
        )
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
            "com.integrallis.models.backend.purejava.Qwen3ModelJarsIntegrationTest.*Q40*",
        )
    }
    dependsOn(tasks.named("downloadQwen306BQ40Model"))
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
        "com.integrallis.models.backend.purejava.Qwen25CoderLargeModelJarsSlowTest",
    ),
    LargeModelTest(
        "deepSeekCoder67BSlowTest",
        "DeepSeek-Coder 6.7B",
        "downloadDeepSeekCoder67BQ4KMModel",
        "com.integrallis.models.backend.purejava.DeepSeekCoderLargeModelJarsSlowTest",
    ),
    LargeModelTest(
        "qwen38BSlowTest",
        "Qwen3 8B",
        "downloadQwen38BQ4KMModel",
        "com.integrallis.models.backend.purejava.Qwen3LargeModelJarsSlowTest",
    ),
    LargeModelTest(
        "deepSeekR1DistillQwen7BSlowTest",
        "DeepSeek-R1-Distill-Qwen-7B",
        "downloadDeepSeekR1DistillQwen7BQ4KMModel",
        "com.integrallis.models.backend.purejava.DeepSeekR1DistillQwenLargeModelJarsSlowTest",
    ),
    LargeModelTest(
        "huatuoGptO17BSlowTest",
        "HuatuoGPT-o1-7B",
        "downloadHuatuoGptO17BQ4KMModel",
        "com.integrallis.models.backend.purejava.HuatuoGptO1LargeModelJarsSlowTest",
    ),
    LargeModelTest(
        "sqlCoder7B2SlowTest",
        "SQLCoder-7B-2",
        "downloadSqlCoder7B2Q5KMModel",
        "com.integrallis.models.backend.purejava.SqlCoderLargeModelJarsSlowTest",
    ),
    LargeModelTest(
        "smolLm33BSlowTest",
        "SmolLM3 3B",
        "downloadSmolLm33BQ4KMModel",
        "com.integrallis.models.backend.purejava.SmolLm3ModelJarsSlowTest",
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
        maxHeapSize = "8g"
    }
}
