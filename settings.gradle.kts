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
