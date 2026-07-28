// Single application-facing dependency for Models generation APIs and runtime.

description = "In-process Java model inference runtime facade"

dependencies {
    api(project(":models-runtime"))
}

val verifyFacadeRuntimeFootprint by tasks.registering {
    group = "verification"
    description = "Verify the Models facade contains only the API and runtime"
    val facadeJar = tasks.named<Jar>("jar")
    dependsOn(configurations.runtimeClasspath, facadeJar)

    val maximumRuntimeBytes = 2L * 1024 * 1024

    doLast {
        val artifacts =
            configurations.runtimeClasspath.get().resolvedConfiguration.resolvedArtifacts
        val runtimeJars =
            (artifacts.map { it.file } + facadeJar.get().archiveFile.get().asFile).distinct()
        val unexpectedModules =
            artifacts
                .map { "${it.moduleVersion.id.group}:${it.moduleVersion.id.name}" }
                .filter { it !in setOf("com.integrallis:models-api", "com.integrallis:models-runtime") }
                .distinct()
                .sorted()
        val runtimeBytes = runtimeJars.sumOf(File::length)

        require(unexpectedModules.isEmpty()) {
            "The com.integrallis:models facade contains unexpected modules: " +
                unexpectedModules.joinToString()
        }
        require(runtimeBytes <= maximumRuntimeBytes) {
            "The com.integrallis:models runtime footprint is ${"%,d".format(runtimeBytes)} bytes; " +
                "limit is ${"%,d".format(maximumRuntimeBytes)} bytes"
        }
    }
}

tasks.named("check") {
    dependsOn(verifyFacadeRuntimeFootprint)
}
