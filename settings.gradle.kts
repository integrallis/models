rootProject.name = "models"

// --- Core ---
include("models-api")
include("models-runtime")

// --- Backends ---
include("models-backend-purejava")
include("models-backend-onnx")
include("models-backend-native")

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
