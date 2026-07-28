pluginManagement {
    repositories {
        gradlePluginPortal()
        mavenCentral()
    }
}

rootProject.name = "models"

// ModelJars is a sibling OSS marker-JAR project. Consuming it through a composite
// build keeps local development independent of publication to modeljars.org.
includeBuild("../model-jars") {
    dependencySubstitution {
        substitute(module("org.modeljars:modeljars-core")).using(project(":modeljars-core"))
        substitute(module("org.modeljars:modeljars-catalog"))
            .using(project(":modeljars-catalog"))
    }
}

// --- Core ---
include("models-api")
include("models-runtime")
include("models")
include("models-rag")
include("models-semantic-order")
include("models-modeljars")

// --- Backends ---
include("backend-java")
include("models-backend-onnx")
include("backend-native")
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
include("models-rag-bench")
include("docs")

// Enable build cache
buildCache {
    local {
        isEnabled = true
    }
}
