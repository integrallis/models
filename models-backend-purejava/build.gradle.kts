import java.net.URI
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.StandardCopyOption

// models-backend-purejava — GGUF parser, scalar inference kernels, KV cache

val qwen25Coder05BQ40FileName = "qwen2.5-coder-0.5b-instruct-q4_0.gguf"
val qwen25Coder05BQ40Url =
    "https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF/resolve/main/$qwen25Coder05BQ40FileName"
val qwen25Coder15BQ40FileName = "qwen2.5-coder-1.5b-instruct-q4_0.gguf"
val qwen25Coder15BQ40Url =
    "https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/resolve/main/$qwen25Coder15BQ40FileName"
val modelsCacheDir =
    providers.gradleProperty("models.cacheDir")
        .orElse(providers.systemProperty("user.home").map { "$it/.jvllm/models" })
val qwen25Coder05BQ40Path = modelsCacheDir.map { "$it/$qwen25Coder05BQ40FileName" }
val qwen25Coder15BQ40Path = modelsCacheDir.map { "$it/$qwen25Coder15BQ40FileName" }

dependencies {
    api(project(":models-api"))
    api("org.modeljars:modeljars-core:0.1.0-SNAPSHOT")

    implementation("com.integrallis:vectors-core:0.1.0-SNAPSHOT")

    // Integration tests use GenerationLoop from models-runtime
    testImplementation(project(":models-runtime"))
    testRuntimeOnly("org.modeljars.huggingface:ggml-org.qwen3-0.6b-gguf.q4_0:3.0.0-q4_0.1")
    testRuntimeOnly("org.modeljars.huggingface:qwen.qwen2.5-coder-0.5b-instruct-gguf.q4_0:2.5.0-q4_0.1")
    testRuntimeOnly("org.modeljars.huggingface:qwen.qwen2.5-coder-0.5b-instruct-gguf.q8_0:2.5.0-q8_0.1")
    testRuntimeOnly("org.modeljars.huggingface:qwen.qwen2.5-coder-1.5b-instruct-gguf.q4_0:2.5.0-q4_0.1")
    testRuntimeOnly("org.modeljars.huggingface:qwen.qwen2.5-coder-1.5b-instruct-gguf.q8_0:2.5.0-q8_0.1")
}

fun registerModelDownloadTask(
    taskName: String,
    modelName: String,
    url: String,
    path: org.gradle.api.provider.Provider<String>,
) {
    tasks.register(taskName) {
        group = "verification"
        description = "Download the $modelName GGUF fixture used by pure-Java integration tests"
        outputs.file(path)

        doLast {
            val destination = Path.of(path.get())
            if (Files.exists(destination)) {
                logger.lifecycle("Model fixture already exists: $destination")
                return@doLast
            }

            Files.createDirectories(destination.parent)
            val temporary = destination.resolveSibling("${destination.fileName}.tmp")
            logger.lifecycle("Downloading $url")
            URI.create(url).toURL().openStream().use { input ->
                Files.copy(input, temporary, StandardCopyOption.REPLACE_EXISTING)
            }
            Files.move(temporary, destination, StandardCopyOption.REPLACE_EXISTING)
            logger.lifecycle("Downloaded model fixture: $destination")
        }
    }
}

registerModelDownloadTask(
    "downloadQwen25Coder05BQ40Model",
    "Qwen2.5-Coder 0.5B Q4_0",
    qwen25Coder05BQ40Url,
    qwen25Coder05BQ40Path,
)

registerModelDownloadTask(
    "downloadQwen25Coder15BQ40Model",
    "Qwen2.5-Coder 1.5B Q4_0",
    qwen25Coder15BQ40Url,
    qwen25Coder15BQ40Path,
)

tasks.named<Test>("integrationTest") {
    dependsOn(tasks.named("downloadQwen25Coder05BQ40Model"))
    dependsOn(tasks.named("downloadQwen25Coder15BQ40Model"))
}
