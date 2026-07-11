rootProject.name = "models"

// Composite build: the sibling java-vectors repo is consumed in-place so the embedding bridge
// (models-embedding) can depend on vectors-db / vectors-cache-semantic-db without requiring those
// artifacts to be published. Gradle resolves the requested coordinates to the local source build.
includeBuild("../vectors") {
    dependencySubstitution {
        substitute(module("com.integrallis:vectors-core")).using(project(":vectors-core"))
        substitute(module("com.integrallis:vectors-db")).using(project(":vectors-db"))
        substitute(module("com.integrallis:vectors-cache-semantic-db"))
            .using(project(":vectors-cache-semantic-db"))
    }
}

// ModelJars is a sibling OSS marker-JAR project. Consuming it through a composite
// build keeps local development independent of publication to modeljars.org.
includeBuild("../model-jars") {
    dependencySubstitution {
        substitute(module("org.modeljars:modeljars-core")).using(project(":modeljars-core"))
        substitute(module("org.modeljars.huggingface:ggml-org.qwen3-0.6b-gguf.q4_0"))
            .using(project(":modeljars-catalog-qwen3-0-6b-q4-0"))
        substitute(module("org.modeljars.huggingface:qwen.qwen3-1.7b-gguf.q8_0"))
            .using(project(":modeljars-catalog-qwen3-1-7b-q8-0"))
        substitute(module("org.modeljars.huggingface:qwen.qwen2.5-coder-0.5b-instruct-gguf.q4_0"))
            .using(project(":modeljars-catalog-qwen2-5-coder-0-5b-instruct-q4-0"))
        substitute(module("org.modeljars.huggingface:qwen.qwen2.5-coder-0.5b-instruct-gguf.q8_0"))
            .using(project(":modeljars-catalog-qwen2-5-coder-0-5b-instruct-q8-0"))
        substitute(module("org.modeljars.huggingface:qwen.qwen2.5-coder-1.5b-instruct-gguf.q4_0"))
            .using(project(":modeljars-catalog-qwen2-5-coder-1-5b-instruct-q4-0"))
        substitute(module("org.modeljars.huggingface:qwen.qwen2.5-coder-1.5b-instruct-gguf.q8_0"))
            .using(project(":modeljars-catalog-qwen2-5-coder-1-5b-instruct-q8-0"))
        substitute(module("org.modeljars.huggingface:qwen.qwen2.5-coder-3b-instruct-gguf.q4_0"))
            .using(project(":modeljars-catalog-qwen2-5-coder-3b-instruct-q4-0"))
        substitute(module("org.modeljars.huggingface:qwen.qwen2.5-coder-7b-instruct-gguf.q4_0"))
            .using(project(":modeljars-catalog-qwen2-5-coder-7b-instruct-q4-0"))
        substitute(module("org.modeljars.huggingface:huggingfacetb.smollm2-360m-instruct-gguf.q8_0"))
            .using(project(":modeljars-catalog-huggingfacetb-smollm2-360m-instruct-q8-0"))
    }
}

// --- Core ---
include("models-api")
include("models-runtime")

// --- Backends ---
include("models-backend-purejava")
include("models-backend-onnx")
include("models-backend-native")
include("models-backend-apple")

// --- Framework adapters ---
include("models-spring-ai")
include("models-langchain4j")
include("models-quarkus")
include("models-semantic-kernel")
include("models-spring-boot-starter")

// --- Bridges ---
include("models-embedding")

// --- Testing & benchmarks ---
include("models-test")
include("models-bench")

// Enable build cache
buildCache {
    local {
        isEnabled = true
    }
}
