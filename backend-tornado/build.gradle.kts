import org.gradle.testing.jacoco.tasks.JacocoCoverageVerification
import org.gradle.testing.jacoco.tasks.JacocoReport

description = "Optional Java-authored GPU acceleration for the Models pure-Java backend"

dependencies {
    api(project(":backend-java"))
    compileOnly("io.github.beehive-lab:tornado-api:5.2.0-jdk25")
    testImplementation("io.github.beehive-lab:tornado-api:5.2.0-jdk25")
    testRuntimeOnly("io.github.beehive-lab:tornado-runtime:5.2.0-jdk25")
    testImplementation("com.integrallis:vectors-core:${providers.gradleProperty("vectorsVersion").get()}")
}

val configuredQwenModel = providers.systemProperty("models.fixtures.qwen306BQ40")
val acceleratorRequired = providers.systemProperty("models.accelerator.required")
val acceleratorExpected = providers.systemProperty("models.accelerator.expected")

tasks.withType<Test>().configureEach {
    configuredQwenModel.orNull?.let {
        systemProperty("models.fixtures.qwen306BQ40", it)
    }
    acceleratorRequired.orNull?.let {
        systemProperty("models.accelerator.required", it)
    }
    acceleratorExpected.orNull?.let {
        systemProperty("models.accelerator.expected", it)
    }
}

// Driver discovery, model loading, and Tornado execution plans are covered by the opt-in model
// integration test and the release hardware gates. Keep the ordinary unit-coverage denominator on
// the device-independent selection, validation, and Java kernel logic.
val hardwareIntegrationClasses =
    listOf(
        "**/TornadoBackend.class",
        "**/TornadoBackendRuntime.class",
        "**/TornadoRuntimeDevices.class",
        "**/TornadoGgufBatchedMatrixKernel*.class"
    )

tasks.withType<JacocoReport>().configureEach {
    classDirectories.setFrom(
        files(classDirectories.files.map { directory ->
            fileTree(directory) { exclude(hardwareIntegrationClasses) }
        })
    )
}

tasks.withType<JacocoCoverageVerification>().configureEach {
    classDirectories.setFrom(
        files(classDirectories.files.map { directory ->
            fileTree(directory) { exclude(hardwareIntegrationClasses) }
        })
    )
}
