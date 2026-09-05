pluginManagement {
    repositories {
        gradlePluginPortal()
        mavenCentral()
    }
}

rootProject.name = "models"

// --- Core ---
include("models-api")
include("models-runtime")
include("models")
include("models-rag")
include("models-semantic-order")
include("models-router")

// --- Backends ---
include("backend-java")
include("backend-tornado")
include("models-backend-onnx")
include("backend-native")
include("backend-apple")
project(":backend-apple").projectDir = file("models-backend-apple")

// --- Framework adapters ---
include("models-spring-ai")
include("models-spring-boot-starter")
include("models-langchain4j")
include("models-quarkus")
include("models-semantic-kernel")

// --- Bridges ---
include("models-embedding")
include("models-audio")

// --- Testing & benchmarks ---
include("models-test")
include("models-bench")
include("models-accelerator-bench")
include("models-rag-bench")
include("docs")

// Enable build cache
buildCache {
    local {
        isEnabled = true
    }
}
