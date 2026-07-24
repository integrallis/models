import java.io.InputStreamReader
import java.nio.charset.StandardCharsets
import java.security.MessageDigest
import java.util.HexFormat
import java.util.Properties
import java.util.jar.JarFile
import org.gradle.api.publish.PublishingExtension
import org.gradle.api.publish.maven.MavenPublication
import org.gradle.jvm.tasks.Jar

// models-backend-native - Models-owned Rust inference kernels exposed through Java FFM

apply(plugin = "maven-publish")

dependencies {
    api(project(":models-api"))
    implementation(project(":models-backend-purejava"))
    testImplementation(project(":models-runtime"))
}

val rustProjectDirectory = layout.projectDirectory.dir("src/main/rust/model-kernels")
val rustManifest = rustProjectDirectory.file("Cargo.toml")
val rustSources = fileTree(rustProjectDirectory) {
    include("Cargo.toml", "Cargo.lock", "src/**/*.rs")
}
val rustTargetDirectory = layout.buildDirectory.dir("rust-target")
val nativeLibraryName =
    when {
        System.getProperty("os.name").startsWith("Mac", ignoreCase = true) ->
            "libjmodels_kernels.dylib"
        System.getProperty("os.name").startsWith("Windows", ignoreCase = true) ->
            "jmodels_kernels.dll"
        else -> "libjmodels_kernels.so"
    }
val nativeLibrary =
    rustTargetDirectory.map { directory -> directory.file("release/$nativeLibraryName") }

data class NativePlatformSpec(val id: String, val libraryFileName: String)

val nativePlatforms =
    listOf(
        NativePlatformSpec("linux-x86_64", "libjmodels_kernels.so"),
        NativePlatformSpec("linux-aarch64", "libjmodels_kernels.so"),
        NativePlatformSpec("macos-x86_64", "libjmodels_kernels.dylib"),
        NativePlatformSpec("macos-aarch64", "libjmodels_kernels.dylib"),
        NativePlatformSpec("windows-x86_64", "jmodels_kernels.dll"),
        NativePlatformSpec("windows-aarch64", "jmodels_kernels.dll"),
    )

fun normalizedHostPlatform(): String {
    val osName = System.getProperty("os.name").lowercase()
    val os =
        when {
            "linux" in osName -> "linux"
            "mac" in osName || "darwin" in osName -> "macos"
            "win" in osName -> "windows"
            else -> error("Unsupported native-kernel build operating system: $osName")
        }
    val architecture =
        when (System.getProperty("os.arch").lowercase()) {
            "amd64", "x86_64", "x64" -> "x86_64"
            "aarch64", "arm64" -> "aarch64"
            else ->
                error(
                    "Unsupported native-kernel build architecture: ${System.getProperty("os.arch")}"
                )
        }
    return "$os-$architecture"
}

val hostNativePlatform = normalizedHostPlatform()
val requestedNativePlatform =
    providers.gradleProperty("modelsNativePlatform").getOrElse(hostNativePlatform)
val nativePlatform =
    nativePlatforms.singleOrNull { it.id == requestedNativePlatform }
        ?: error(
            "Unsupported modelsNativePlatform=$requestedNativePlatform; expected one of " +
                nativePlatforms.joinToString { it.id }
        )
val nativeResourceRoot = layout.buildDirectory.dir("generated/native-platform-resources")
val nativeResourceDirectory =
    nativeResourceRoot.map { directory ->
        directory.dir("META-INF/models/native/${nativePlatform.id}")
    }

val cargoBuildRelease by tasks.registering(Exec::class) {
    group = "build"
    description = "Build the release Models Rust kernel library for the host platform"
    inputs.files(rustSources)
    outputs.file(nativeLibrary)
    environment("CARGO_TARGET_DIR", rustTargetDirectory.get().asFile.absolutePath)
    commandLine(
        "cargo",
        "build",
        "--release",
        "--manifest-path",
        rustManifest.asFile.absolutePath,
    )
}

val cargoTest by tasks.registering(Exec::class) {
    group = "verification"
    description = "Run the Models Rust kernel unit tests"
    inputs.files(rustSources)
    environment("CARGO_TARGET_DIR", rustTargetDirectory.get().asFile.absolutePath)
    commandLine(
        "cargo",
        "test",
        "--release",
        "--manifest-path",
        rustManifest.asFile.absolutePath,
    )
}

val cargoClippy by tasks.registering(Exec::class) {
    group = "verification"
    description = "Lint the Models Rust kernel library"
    inputs.files(rustSources)
    environment("CARGO_TARGET_DIR", rustTargetDirectory.get().asFile.absolutePath)
    commandLine(
        "cargo",
        "clippy",
        "--release",
        "--all-targets",
        "--manifest-path",
        rustManifest.asFile.absolutePath,
        "--",
        "-D",
        "warnings",
    )
}

val verifyNativeBuildPlatform by tasks.registering {
    group = "verification"
    description = "Reject a native artifact label that does not match the build host"
    inputs.property("requestedPlatform", nativePlatform.id)
    inputs.property("hostPlatform", hostNativePlatform)
    doLast {
        require(nativePlatform.id == hostNativePlatform) {
            "Native artifact ${nativePlatform.id} must be built on a matching host; " +
                "this host is $hostNativePlatform"
        }
        require(nativePlatform.libraryFileName == nativeLibraryName) {
            "Native artifact ${nativePlatform.id} expects ${nativePlatform.libraryFileName} " +
                "but Cargo produces $nativeLibraryName"
        }
    }
}

val prepareNativePlatformResources by tasks.registering {
    group = "build"
    description = "Prepare the host library and integrity metadata for its platform JAR"
    dependsOn(cargoBuildRelease, verifyNativeBuildPlatform)
    inputs.file(nativeLibrary)
    inputs.property("abi", 1)
    inputs.property("platform", nativePlatform.id)
    outputs.dir(nativeResourceRoot)
    doLast {
        val source = nativeLibrary.get().asFile
        require(source.isFile) { "Cargo did not produce ${source.absolutePath}" }
        val outputRoot = nativeResourceRoot.get().asFile
        delete(outputRoot)
        val outputDirectory = nativeResourceDirectory.get().asFile
        outputDirectory.mkdirs()
        val outputLibrary = outputDirectory.resolve(nativePlatform.libraryFileName)
        source.copyTo(outputLibrary, overwrite = true)
        val digest =
            HexFormat.of()
                .formatHex(MessageDigest.getInstance("SHA-256").digest(outputLibrary.readBytes()))
        outputDirectory.resolve("native.properties").writeText(
            """
            abi=1
            platform=${nativePlatform.id}
            library=${nativePlatform.libraryFileName}
            sha256=$digest
            """.trimIndent() + "\n",
            StandardCharsets.UTF_8,
        )
    }
}

val nativePlatformJar by tasks.registering(Jar::class) {
    group = "build"
    description = "Package the host Models kernel as a platform-specific native JAR"
    dependsOn(prepareNativePlatformResources)
    archiveBaseName.set("models-kernels-${nativePlatform.id}")
    destinationDirectory.set(layout.buildDirectory.dir("native-artifacts"))
    from(nativeResourceRoot)
    isPreserveFileTimestamps = false
    isReproducibleFileOrder = true
    manifest {
        attributes(
            "Models-Native-ABI" to "1",
            "Models-Native-Platform" to nativePlatform.id,
        )
    }
}

fun verifyNativeArtifact(artifact: File, platform: NativePlatformSpec) {
    require(artifact.isFile) { "Missing native artifact: ${artifact.absolutePath}" }
    JarFile(artifact).use { jar ->
        val resourceDirectory = "META-INF/models/native/${platform.id}/"
        val metadataEntry =
            jar.getJarEntry("${resourceDirectory}native.properties")
                ?: error("${artifact.name} has no native.properties for ${platform.id}")
        val metadata =
            Properties().apply {
                jar.getInputStream(metadataEntry).use { input ->
                    InputStreamReader(input, StandardCharsets.UTF_8).use(::load)
                }
            }
        require(metadata.getProperty("abi") == "1") {
            "${artifact.name} does not declare native ABI 1"
        }
        require(metadata.getProperty("platform") == platform.id) {
            "${artifact.name} declares the wrong platform: ${metadata.getProperty("platform")}"
        }
        require(metadata.getProperty("library") == platform.libraryFileName) {
            "${artifact.name} declares the wrong library filename"
        }
        val libraryEntry =
            jar.getJarEntry(resourceDirectory + platform.libraryFileName)
                ?: error("${artifact.name} has no ${platform.libraryFileName}")
        val actualDigest =
            jar.getInputStream(libraryEntry).use { input ->
                HexFormat.of()
                    .formatHex(MessageDigest.getInstance("SHA-256").digest(input.readAllBytes()))
            }
        require(metadata.getProperty("sha256") == actualDigest) {
            "${artifact.name} native library SHA-256 does not match its metadata"
        }
    }
}

val verifyNativePlatformJar by tasks.registering {
    group = "verification"
    description = "Verify the host platform JAR layout, ABI metadata, and native checksum"
    dependsOn(nativePlatformJar)
    inputs.file(nativePlatformJar.flatMap { it.archiveFile })
    doLast {
        verifyNativeArtifact(nativePlatformJar.get().archiveFile.get().asFile, nativePlatform)
    }
}

tasks.withType<Test>().configureEach {
    dependsOn(cargoBuildRelease, nativePlatformJar)
    jvmArgs("--enable-native-access=ALL-UNNAMED")
    systemProperty("models.native.kernels.library", nativeLibrary.get().asFile.absolutePath)
    systemProperty(
        "models.native.test.artifact",
        nativePlatformJar.get().archiveFile.get().asFile.absolutePath,
    )
}

tasks.named("check") {
    dependsOn(cargoTest, cargoClippy, verifyNativePlatformJar)
}

val githubPackagesEnabled =
    providers.gradleProperty("modelsGithubPackages").map(String::toBoolean).getOrElse(false)

val nativeArtifactDirectory =
    providers.gradleProperty("modelsNativeArtifactDirectory").map(rootProject::file).orNull
val aggregateNativeArtifacts =
    nativeArtifactDirectory?.let { directory ->
        nativePlatforms.associateWith { platform ->
            fileTree(directory) {
                include("**/models-kernels-${platform.id}-${project.version}.jar")
            }.files.singleOrNull()
                ?: error(
                    "Expected exactly one models-kernels-${platform.id}-${project.version}.jar " +
                        "under ${directory.absolutePath}"
                )
        }
    }

val privateTestBundleJar =
    aggregateNativeArtifacts?.let { nativeArtifacts ->
        tasks.register<Jar>("privateTestBundleJar") {
            group = "build"
            description =
                "Build a private-test JAR containing the Java backend and all native platforms"
            archiveBaseName.set("models-backend-native-standalone")
            destinationDirectory.set(layout.buildDirectory.dir("native-artifacts"))
            duplicatesStrategy = DuplicatesStrategy.EXCLUDE
            isPreserveFileTimestamps = false
            isReproducibleFileOrder = true
            isZip64 = true
            dependsOn(configurations.runtimeClasspath)
            from(sourceSets["main"].output)
            from({
                configurations.runtimeClasspath.get().map { dependency ->
                    if (dependency.isDirectory) dependency else zipTree(dependency)
                }
            })
            nativeArtifacts.values.forEach { artifact ->
                from(zipTree(artifact))
            }
            exclude(
                "META-INF/MANIFEST.MF",
                "META-INF/*.SF",
                "META-INF/*.RSA",
                "META-INF/*.DSA",
                "module-info.class",
                "META-INF/versions/**/module-info.class",
            )
            manifest {
                attributes(
                    "Models-Native-ABI" to "1",
                    "Models-Native-Platforms" to
                        nativePlatforms.joinToString(",") { it.id },
                    "Models-Private-Test-Bundle" to "true",
                )
            }
        }
    }

val verifyNativePublicationArtifacts =
    aggregateNativeArtifacts?.let { nativeArtifacts ->
        tasks.register("verifyNativePublicationArtifacts") {
            group = "verification"
            description = "Verify all native JARs before bundling or publishing a CI version"
            inputs.files(nativeArtifacts.values)
            doLast {
                nativeArtifacts.forEach { (platform, artifact) ->
                    verifyNativeArtifact(artifact, platform)
                }
            }
        }
    }

val verifyPrivateTestBundle =
    privateTestBundleJar?.let { bundleJar ->
        val nativeVerification =
            verifyNativePublicationArtifacts
                ?: error("Native artifact verification was not configured")
        tasks.register("verifyPrivateTestBundle") {
            group = "verification"
            description = "Verify the standalone private-test JAR is complete"
            dependsOn(bundleJar, nativeVerification)
            inputs.file(bundleJar.flatMap { it.archiveFile })
            doLast {
                JarFile(bundleJar.get().archiveFile.get().asFile).use { jar ->
                    listOf(
                        "com/integrallis/models/backend/nativekernel/RustFfmBackend.class",
                        "com/integrallis/models/backend/purejava/PureJavaBackend.class",
                        "com/integrallis/models/api/InferenceBackend.class",
                        "com/integrallis/vectors/core/VectorUtil.class",
                        "org/modeljars/ModelJarRegistry.class",
                    ).forEach { classEntry ->
                        require(jar.getJarEntry(classEntry) != null) {
                            "Standalone private-test JAR is missing $classEntry"
                        }
                    }
                    nativePlatforms.forEach { platform ->
                        require(
                            jar.getJarEntry(
                                "META-INF/models/native/${platform.id}/native.properties"
                            ) != null
                        ) {
                            "Standalone private-test JAR is missing ${platform.id}"
                        }
                    }
                }
            }
        }
    }

if (githubPackagesEnabled) {
    val nativeArtifacts =
        aggregateNativeArtifacts
            ?: error(
                "modelsNativeArtifactDirectory is required when modelsGithubPackages=true"
            )
    val bundleJar =
        privateTestBundleJar
            ?: error("privateTestBundleJar was not configured for GitHub Packages")
    val nativeVerification =
        verifyNativePublicationArtifacts
            ?: error("Native artifact verification was not configured for GitHub Packages")
    val bundleVerification =
        verifyPrivateTestBundle
            ?: error("Private-test bundle verification was not configured for GitHub Packages")

    configure<PublishingExtension> {
        publications {
            create<MavenPublication>("githubStandalone") {
                artifactId = "models-backend-native-standalone"
                artifact(bundleJar)
                pom {
                    name.set("Models native backend private-test bundle")
                    description.set(
                        "Self-contained Java 25 FFM backend and Models-owned Rust kernels"
                    )
                    url.set("https://github.com/integrallis/models")
                    licenses {
                        license {
                            name.set("The Apache License, Version 2.0")
                            url.set("https://www.apache.org/licenses/LICENSE-2.0.txt")
                        }
                    }
                    scm {
                        connection.set("scm:git:https://github.com/integrallis/models.git")
                        url.set("https://github.com/integrallis/models")
                    }
                }
            }
            nativeArtifacts.forEach { (platform, artifactFile) ->
                create<MavenPublication>(
                    "github" +
                        platform.id
                            .split('-', '_')
                            .joinToString("") { part ->
                                part.replaceFirstChar(Char::uppercase)
                            }
                ) {
                    artifactId = "models-kernels-${platform.id}"
                    artifact(artifactFile)
                    pom {
                        name.set("Models native kernels for ${platform.id}")
                        description.set(
                            "Models-owned Rust inference kernels for ${platform.id}"
                        )
                        url.set("https://github.com/integrallis/models")
                        licenses {
                            license {
                                name.set("The Apache License, Version 2.0")
                                url.set(
                                    "https://www.apache.org/licenses/LICENSE-2.0.txt"
                                )
                            }
                        }
                        scm {
                            connection.set(
                                "scm:git:https://github.com/integrallis/models.git"
                            )
                            url.set("https://github.com/integrallis/models")
                        }
                    }
                }
            }
        }
        repositories {
            maven {
                name = "GitHubPackages"
                val repository =
                    System.getenv("GITHUB_REPOSITORY")?.takeIf(String::isNotBlank)
                        ?: "integrallis/models"
                url = uri("https://maven.pkg.github.com/$repository")
                credentials {
                    username = System.getenv("GITHUB_ACTOR") ?: ""
                    password = System.getenv("GITHUB_TOKEN") ?: ""
                }
            }
        }
    }

    tasks.matching {
        it.name.startsWith("publish") && it.name.endsWith("ToGitHubPackagesRepository")
    }.configureEach {
        dependsOn(nativeVerification, bundleVerification)
    }
}
