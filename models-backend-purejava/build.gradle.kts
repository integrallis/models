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

tasks.named<Test>("slowTest") {
    dependsOn(
        modelFixtures
            .filter(ModelFixture::slow)
            .map { tasks.named(it.taskName) },
    )
    maxParallelForks = 1
    maxHeapSize = "8g"
}
