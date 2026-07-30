import java.nio.charset.StandardCharsets
import java.security.MessageDigest
import java.util.HexFormat
import java.util.jar.JarFile
import org.gradle.jvm.tasks.Jar
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
            resources.srcDir(prepareBundledAppleBridge)
        }
    }
    tasks.named("processResources") {
        dependsOn(prepareBundledAppleBridge)
    }
}

val appleRuntimeJar = tasks.named<Jar>("jar")
val appleSourcesJar =
    tasks.named<Jar>("sourcesJar") {
        // Generated native resources belong only in the runtime artifact.
        exclude("META-INF/models/apple-foundation/**")
    }

val verifyAppleBackendArchives =
    tasks.register("verifyAppleBackendArchives") {
        group = "verification"
        description = "Verify Apple bridge runtime and source archive ownership"
        dependsOn(appleRuntimeJar, appleSourcesJar)
        inputs.files(
            appleRuntimeJar.flatMap { it.archiveFile },
            appleSourcesJar.flatMap { it.archiveFile }
        )
        inputs.property("bundledBridgeExpected", appleBridgeArtifact != null)

        doLast {
            val resourceDirectory =
                "META-INF/models/apple-foundation/$appleBridgePlatform/"
            val bridgeResource = resourceDirectory + appleBridgeFileName
            val metadataResource = resourceDirectory + "native.properties"

            JarFile(appleRuntimeJar.get().archiveFile.get().asFile).use { jar ->
                val bridge = jar.getJarEntry(bridgeResource)
                val metadata = jar.getJarEntry(metadataResource)
                if (appleBridgeArtifact != null) {
                    require(bridge != null && metadata != null) {
                        "backend-apple runtime JAR must contain the bundled bridge and metadata"
                    }
                } else {
                    require(bridge == null && metadata == null) {
                        "Source-only backend-apple runtime JAR must not contain bridge resources"
                    }
                }
            }

            JarFile(appleSourcesJar.get().archiveFile.get().asFile).use { jar ->
                val nativeResources =
                    jar.entries().asSequence()
                        .map { it.name }
                        .filter { it.startsWith(resourceDirectory) }
                        .toList()
                require(nativeResources.isEmpty()) {
                    "backend-apple sources JAR must not duplicate native resources: " +
                        nativeResources
                }
            }
        }
    }

tasks.named("check") {
    dependsOn(verifyAppleBackendArchives)
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
