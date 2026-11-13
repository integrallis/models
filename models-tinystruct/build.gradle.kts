// models-tinystruct — Tinystruct and TypeSafe semantic dispatch, chat, embedding, and reranking adapters.

description =
    "Tinystruct and TypeSafe semantic dispatch, chat, embedding, and reranking adapters"

val tinystructVersion = "1.7.38"
val tinystructTypesafeVersion = "1.0.1"

dependencies {
    api(project(":models-runtime"))
    api(project(":models-router"))
    api(project(":models-decisions"))

    // Applications own their framework versions; Models adapts their public contracts.
    compileOnly("org.tinystruct:tinystruct:$tinystructVersion") {
        isTransitive = false
    }
    compileOnly("org.tinystruct:tinystruct-typesafe-client:$tinystructTypesafeVersion") {
        isTransitive = false
    }
    compileOnly("org.tinystruct:tinystruct-typesafe-core:$tinystructTypesafeVersion") {
        isTransitive = false
    }

    testImplementation(project(":backend-java"))
    testImplementation("org.tinystruct:tinystruct:$tinystructVersion") {
        isTransitive = false
    }
    testImplementation("org.tinystruct:tinystruct-typesafe-client:$tinystructTypesafeVersion") {
        isTransitive = false
    }
    testImplementation("org.tinystruct:tinystruct-typesafe-core:$tinystructTypesafeVersion") {
        isTransitive = false
    }
}

tasks.withType<Test>().configureEach {
    val testWorkingDir = layout.buildDirectory.dir("test-workdir")
    doFirst {
        testWorkingDir.get().asFile.mkdirs()
    }
    workingDir(testWorkingDir)
}
