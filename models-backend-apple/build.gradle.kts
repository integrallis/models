import java.nio.charset.StandardCharsets
import java.security.MessageDigest
import java.util.HexFormat
import org.gradle.testing.jacoco.tasks.JacocoCoverageVerification
import org.gradle.testing.jacoco.tasks.JacocoReport

// backend-apple - Apple Foundation Models bridge via Java FFM

dependencies {
    api(project(":models-api"))
}

val appleBridgeFileName = "libjavamodels_apple_foundation.dylib"
val appleBridgePlatform = "macos-aarch64"
val appleBridgeAbi = 1
val appleBridgeArtifactDirectory =
    providers.gradleProperty("modelsAppleBridgeArtifactDirectory").map(rootProject::file).orNull
val appleBridgeArtifact =
    appleBridgeArtifactDirectory?.let { directory ->
        fileTree(directory) {
            include("**/$appleBridgeFileName")
        }.files.singleOrNull()
            ?: error(
                "Expected exactly one $appleBridgeFileName under ${directory.absolutePath}"
            )
    }
val generatedAppleResources =
    layout.buildDirectory.dir("generated/apple-foundation-resources")

val prepareBundledAppleBridge =
    appleBridgeArtifact?.let { bridge ->
        tasks.register("prepareBundledAppleBridge") {
            group = "build"
            description = "Prepare the signed-release Apple Foundation Models bridge resources"
            inputs.file(bridge)
            inputs.property("abi", appleBridgeAbi)
            inputs.property("platform", appleBridgePlatform)
            outputs.dir(generatedAppleResources)
            doLast {
                val outputRoot = generatedAppleResources.get().asFile
                delete(outputRoot)
                val outputDirectory =
                    outputRoot.resolve(
                        "META-INF/models/apple-foundation/$appleBridgePlatform"
                    )
                outputDirectory.mkdirs()
                val outputBridge = outputDirectory.resolve(appleBridgeFileName)
                bridge.copyTo(outputBridge, overwrite = true)
                val digest =
                    HexFormat.of().formatHex(
                        MessageDigest.getInstance("SHA-256")
                            .digest(outputBridge.readBytes())
                    )
                outputDirectory.resolve("native.properties").writeText(
                    """
                    abi=$appleBridgeAbi
                    platform=$appleBridgePlatform
                    library=$appleBridgeFileName
                    sha256=$digest
                    """.trimIndent() + "\n",
                    StandardCharsets.UTF_8
                )
            }
        }
    }

if (prepareBundledAppleBridge != null) {
    sourceSets {
        main {
            resources.srcDir(generatedAppleResources)
        }
    }
    tasks.named("processResources") {
        dependsOn(prepareBundledAppleBridge)
    }
    tasks.named("check") {
        dependsOn(prepareBundledAppleBridge)
    }
}

tasks.withType<Test>().configureEach {
    jvmArgs("--enable-native-access=ALL-UNNAMED")
    System.getProperty("models.apple.foundation.library")?.let {
        systemProperty("models.apple.foundation.library", it)
    }
}

val integrationTest = tasks.named<Test>("integrationTest")
val coverageData =
    files(
        layout.buildDirectory.file("jacoco/test.exec"),
        layout.buildDirectory.file("jacoco/integrationTest.exec")
    )

tasks.named<JacocoReport>("jacocoTestReport") {
    dependsOn(integrationTest)
    executionData(coverageData)
}

tasks.named<JacocoCoverageVerification>("jacocoTestCoverageVerification") {
    dependsOn(integrationTest)
    executionData(coverageData)
}
