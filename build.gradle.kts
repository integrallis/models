import groovy.json.JsonSlurper
import java.security.MessageDigest
import java.util.jar.JarFile

plugins {
    java
    `java-library`
    `maven-publish`
    id("com.github.spotbugs") version "6.4.4" apply false
    id("com.diffplug.spotless") version "6.25.0" apply false
    id("org.cyclonedx.bom") version "3.2.4" apply false
    id("org.owasp.dependencycheck") version "12.2.1" apply false
    id("com.integrallis.mfcqi") version "0.7.0"
    jacoco
}

val notebookRepositoryUrl =
    providers.gradleProperty("notebookRepository")
        .orElse(providers.environmentVariable("MODELS_NOTEBOOK_REPOSITORY"))
        .map(String::trim)
        .filter(String::isNotEmpty)

allprojects {
    group = "com.integrallis"

    repositories {
        if (project == rootProject && notebookRepositoryUrl.isPresent) {
            maven {
                name = "notebookCandidate"
                url = uri(notebookRepositoryUrl.get())
                content {
                    includeGroup("com.integrallis")
                }
            }
        }
        mavenCentral()
    }
}

// Library subprojects (excludes executable benchmark applications)
val benchmarkProjectNames = setOf("models-bench", "models-rag-bench")
val libraryProjects =
    subprojects.filterNot { it.name in benchmarkProjectNames || it.name == "docs" }
val benchmarkProjects = subprojects.filter { it.name in benchmarkProjectNames }
val libraryModuleNames = libraryProjects.map { it.name }.toSet()
val publishedModuleNames =
    setOf(
        "models-api",
        "models-runtime",
        "models",
        "models-rag",
        "models-semantic-order",
        "backend-java",
        "backend-native",
        "models-langchain4j",
        "models-spring-ai",
        "models-embedding"
    )
val publishedProjects = libraryProjects.filter { it.name in publishedModuleNames }
val scaffoldProjects = libraryProjects.filterNot { it.name in publishedModuleNames }
extra["publishedJavadocModuleNames"] = publishedModuleNames - setOf("models")

val notebookMode =
    providers.gradleProperty("notebookMode")
        .orElse(providers.environmentVariable("MODELS_NOTEBOOK_MODE"))
        .orElse("source")
val notebookVersion =
    providers.gradleProperty("notebookVersion")
        .orElse(providers.environmentVariable("MODELS_VERSION"))
        .orElse(provider { project.version.toString() })

fun Configuration.asNotebookRuntimeClasspath() {
    isCanBeConsumed = false
    isCanBeResolved = true
    attributes {
        attribute(Category.CATEGORY_ATTRIBUTE, objects.named(Category.LIBRARY))
        attribute(Usage.USAGE_ATTRIBUTE, objects.named(Usage.JAVA_RUNTIME))
    }
}

val notebookSourceClasspath by configurations.creating {
    description = "Runtime classpath for notebooks using the current project sources"
    asNotebookRuntimeClasspath()
}
val notebookReleaseClasspath by configurations.creating {
    description = "Runtime classpath for notebooks using released com.integrallis artifacts"
    asNotebookRuntimeClasspath()
}

dependencies {
    listOf(
        ":models",
        ":backend-java",
        ":models-langchain4j",
        ":models-spring-ai",
        ":models-rag",
        ":models-test"
    ).forEach { notebookSourceClasspath(project(it)) }

    notebookReleaseClasspath(project(":models-test"))
    listOf(
        "models",
        "backend-java",
        "models-langchain4j",
        "models-spring-ai",
        "models-rag"
    ).forEach { artifact ->
        notebookReleaseClasspath("com.integrallis:$artifact:${notebookVersion.get()}")
    }

    listOf("org.slf4j:slf4j-nop:2.0.17").forEach { dependency ->
        notebookSourceClasspath(dependency)
        notebookReleaseClasspath(dependency)
    }
}

// MFCQI scores the production sources that make up the published Models release. CI stages those
// sources outside the repository because MFCQI intentionally excludes build output directories.
mfcqi {
    source.set(
        layout.projectDirectory.dir(providers.gradleProperty("mfcqi.sourceDir").getOrElse("."))
    )
    bytecodeSecurity.set(false)
    failOnGate.set(false)
}

// Every source-bearing module receives its own score and shields.io endpoint JSON.
configure(subprojects.filter { it.file("src/main/java").isDirectory }) {
    apply(plugin = "com.integrallis.mfcqi")
    configure<com.integrallis.mfcqi.gradle.MfcqiExtension> {
        source.set(layout.projectDirectory.dir("src/main/java"))
        bytecodeSecurity.set(false)
        failOnGate.set(false)
    }
}

val apacheLicenseHeader = """
    /*
     * Copyright 2025-2026 Integrallis Software, LLC
     *
     * Licensed under the Apache License, Version 2.0 (the "License");
     * you may not use this file except in compliance with the License.
     * You may obtain a copy of the License at
     *
     *     https://www.apache.org/licenses/LICENSE-2.0
     *
     * Unless required by applicable law or agreed to in writing, software
     * distributed under the License is distributed on an "AS IS" BASIS,
     * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
     * See the License for the specific language governing permissions and
     * limitations under the License.
     */
""".trimIndent()

configure(libraryProjects) {
    apply(plugin = "java")
    apply(plugin = "java-library")
    apply(plugin = "com.github.spotbugs")
    apply(plugin = "com.diffplug.spotless")
    apply(plugin = "jacoco")
    apply(plugin = "org.owasp.dependencycheck")
    if (project.name in publishedModuleNames) {
        apply(plugin = "org.cyclonedx.bom")
        // Realize the plugin's outgoing configuration before maven-publish observes variants.
        tasks.named("cyclonedxDirectBom").get()
    }

    // Dependency locking — enforced when lockfiles exist, lenient otherwise
    dependencyLocking {
        lockAllConfigurations()
    }

    tasks.register("resolveAndLockAllConfigurations") {
        group = "verification"
        description = "Resolve this module's dependencies and write its lockfile"
        notCompatibleWithConfigurationCache("Resolves and locks every module configuration")
        doFirst {
            require(gradle.startParameter.isWriteDependencyLocks) {
                "${path} must be run with the --write-locks flag"
            }
        }
        doLast {
            configurations.filter { it.isCanBeResolved }.forEach { configuration ->
                configuration.resolve()
            }
        }
    }

    java {
        toolchain {
            languageVersion = JavaLanguageVersion.of(25)
        }
        withJavadocJar()
        withSourcesJar()
    }

    tasks.withType<JavaCompile> {
        options.encoding = "UTF-8"
        options.compilerArgs.addAll(listOf(
            "-parameters",
            "-Xlint:all",
            "-Xlint:-processing",
            "-Xlint:-incubating",
            "-Xlint:-classfile",
            "-Werror"
        ))
    }

    // Common JVM args and logging for ALL Test tasks
    tasks.withType<Test> {
        testLogging {
            events("passed", "skipped", "failed")
            exceptionFormat = org.gradle.api.tasks.testing.logging.TestExceptionFormat.FULL
            showStandardStreams = false
        }
        maxParallelForks = (Runtime.getRuntime().availableProcessors() / 4).coerceAtLeast(1)
        // java-vectors (consumed by models-embedding via the composite build) links the incubating
        // Panama Vector API in its shared PanamaConstants class, so any test that exercises a
        // VectorCollection must have jdk.incubator.vector in the module graph or it fails to link.
        jvmArgs("--add-modules", "jdk.incubator.vector")
    }

    tasks.register<Test>("unitTest") {
        description = "Run only unit tests"
        group = "verification"
        testClassesDirs = sourceSets["test"].output.classesDirs
        classpath = sourceSets["test"].runtimeClasspath
        useJUnitPlatform {
            includeTags("unit")
        }
        testLogging {
            events("passed", "skipped", "failed")
        }
    }

    tasks.register<Test>("slowTest") {
        description = "Run slow tests (large models, extended inference)"
        group = "verification"
        testClassesDirs = sourceSets["test"].output.classesDirs
        classpath = sourceSets["test"].runtimeClasspath
        useJUnitPlatform {
            includeTags("slow")
        }
        testLogging {
            events("passed", "skipped", "failed")
        }
    }

    tasks.register<Test>("integrationTest") {
        description = "Run integration tests"
        group = "verification"
        testClassesDirs = sourceSets["test"].output.classesDirs
        classpath = sourceSets["test"].runtimeClasspath
        useJUnitPlatform {
            includeTags("integration")
        }
        testLogging {
            events("passed", "skipped", "failed")
            exceptionFormat = org.gradle.api.tasks.testing.logging.TestExceptionFormat.FULL
        }
        maxParallelForks = 1
        // Real model inference needs more heap than the default
        maxHeapSize = "4g"
    }

    // Default 'test' task excludes infrastructure-heavy tags
    tasks.named<Test>("test") {
        useJUnitPlatform {
            excludeTags("slow", "benchmark", "integration")
        }
    }

    tasks.withType<Javadoc> {
        val javadocOptions = options as StandardJavadocDocletOptions
        javadocOptions.addBooleanOption("Xdoclint:all,-missing", true)
        javadocOptions.addBooleanOption("html5", true)
        isFailOnError = true
        enabled = project !in scaffoldProjects
    }

    configure<com.diffplug.gradle.spotless.SpotlessExtension> {
        java {
            googleJavaFormat("1.35.0")
            removeUnusedImports()
            trimTrailingWhitespace()
            endWithNewline()
            licenseHeader(apacheLicenseHeader)
        }
    }

    // Configure SpotBugs
    tasks.withType<com.github.spotbugs.snom.SpotBugsTask> {
        excludeFilter.set(file("${rootProject.projectDir}/spotbugs-exclude.xml"))
    }

    // Disable SpotBugs for test code
    tasks.named("spotbugsTest") {
        enabled = false
    }

    tasks.jacocoTestReport {
        dependsOn(tasks.test)
        reports {
            xml.required = true
            html.required = true
        }
    }

    tasks.jacocoTestCoverageVerification {
        dependsOn(tasks.test)
        enabled = project in publishedProjects
        violationRules {
            rule {
                limit {
                    minimum = "0.80".toBigDecimal()
                }
            }
        }
    }

    if (project in publishedProjects) {
        tasks.named("check") {
            dependsOn(tasks.jacocoTestCoverageVerification)
        }
    }

    // OWASP Dependency-Check — runs only when explicitly invoked
    configure<org.owasp.dependencycheck.gradle.extension.DependencyCheckExtension> {
        failBuildOnCVSS = 7.0f
        formats.set(listOf("HTML", "JSON", "SARIF"))
        outputDirectory.set(layout.buildDirectory.dir("reports/dependency-check"))
        suppressionFile = "${rootProject.projectDir}/owasp-suppressions.xml"
        nvd.apiKey = System.getenv("NVD_API_KEY") ?: ""
        analyzers.ossIndex.enabled = true
        analyzers.ossIndex.username = System.getenv("OSS_INDEX_USERNAME") ?: ""
        analyzers.ossIndex.password = System.getenv("OSS_INDEX_TOKEN") ?: ""
    }

    // Reproducible JAR manifest attributes
    tasks.withType<Jar>().configureEach {
        manifest {
            attributes(
                "Build-Jdk-Spec" to "25",
                "Created-By" to "Gradle ${gradle.gradleVersion}"
            )
        }
    }

    dependencies {
        // Testing
        testImplementation("org.junit.jupiter:junit-jupiter:5.11.4")
        testImplementation("org.assertj:assertj-core:3.27.2")
        testImplementation("org.mockito:mockito-core:5.23.0")
        testImplementation("org.mockito:mockito-junit-jupiter:5.23.0")
        testRuntimeOnly("org.junit.platform:junit-platform-launcher:1.11.4")
    }
}

// Runnable benchmark projects use the same formatting and license policy as library sources.
configure(benchmarkProjects) {
    apply(plugin = "com.diffplug.spotless")

    configure<com.diffplug.gradle.spotless.SpotlessExtension> {
        java {
            googleJavaFormat("1.35.0")
            removeUnusedImports()
            trimTrailingWhitespace()
            endWithNewline()
            licenseHeader(apacheLicenseHeader)
        }
    }
}

// ---------------------------------------------------------------------------
// Publishing: only implemented 0.1.x modules stage for Maven Central.
// ---------------------------------------------------------------------------

configure(publishedProjects) {
    apply(plugin = "maven-publish")

    configure<PublishingExtension> {
        publications {
            create<MavenPublication>("maven") {
                from(components["java"])
                pom {
                    name.set(project.name)
                    description.set(provider { project.description ?: "models — ${project.name}" })
                    url.set("https://github.com/integrallis/models")
                    licenses {
                        license {
                            name.set("The Apache License, Version 2.0")
                            url.set("https://www.apache.org/licenses/LICENSE-2.0.txt")
                            distribution.set("repo")
                        }
                    }
                    developers {
                        developer {
                            id.set("bsbodden")
                            name.set("Brian Sam-Bodden")
                            email.set("bsbodden@gmail.com")
                            organization.set("Integrallis Software")
                            organizationUrl.set("https://integrallis.com")
                        }
                    }
                    scm {
                        connection.set("scm:git:https://github.com/integrallis/models.git")
                        developerConnection.set("scm:git:ssh://git@github.com/integrallis/models.git")
                        url.set("https://github.com/integrallis/models")
                    }
                }
            }
        }
        repositories {
            maven {
                name = "staging"
                url = uri(rootProject.layout.buildDirectory.dir("staging-deploy").get().asFile)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Compliance verification tasks
// ---------------------------------------------------------------------------

tasks.register("verifySbom") {
    group = "verification"
    description = "Verify CycloneDX SBOM generation for every published module"
    dependsOn(publishedProjects.map { "${it.path}:cyclonedxDirectBom" })
    doLast {
        publishedProjects.forEach { proj ->
            val file = proj.layout.buildDirectory.file("reports/cyclonedx-direct/bom.json").get().asFile
            require(file.exists()) { "SBOM not found: ${file.absolutePath}" }
            @Suppress("UNCHECKED_CAST")
            val json = groovy.json.JsonSlurper().parseText(file.readText()) as Map<String, Any?>
            require(json["bomFormat"] == "CycloneDX") {
                "Invalid bomFormat in ${proj.name}: ${json["bomFormat"]}"
            }
            val specVersion = json["specVersion"] as? String
            require(specVersion != null && specVersion.startsWith("1.")) {
                "Invalid specVersion in ${proj.name}: $specVersion"
            }
            println("  SBOM valid: ${proj.name} (CycloneDX $specVersion)")
        }
    }
}

tasks.register("verifyGovernanceFiles") {
    group = "verification"
    description = "Verify SECURITY.md and CONTRIBUTING.md exist"
    doLast {
        listOf("SECURITY.md", "CONTRIBUTING.md").forEach { name ->
            val f = file(name)
            require(f.exists()) { "$name not found in ${projectDir.absolutePath}" }
            require(f.length() > 0) { "$name is empty" }
            println("  $name exists (${f.length()} bytes)")
        }
    }
}

tasks.register("resolveAndLockAll") {
    group = "verification"
    description = "Resolve all library dependencies and write lockfiles (run with --write-locks)"
    dependsOn(libraryProjects.map { "${it.path}:resolveAndLockAllConfigurations" })
}

tasks.register("verifyLockfiles") {
    group = "verification"
    description = "Verify dependency lockfiles exist for every library module"
    doLast {
        libraryProjects.forEach { proj ->
            val lockfile = proj.file("gradle.lockfile")
            require(lockfile.isFile) { "Missing lockfile: ${lockfile.absolutePath}" }
            println("  Lockfile: ${proj.name}")
        }
    }
}

tasks.register("verifyPublishingConfigured") {
    group = "verification"
    description = "Verify Maven publications and JReleaser configuration for release modules"
    doLast {
        require(file("jreleaser.yml").isFile) { "jreleaser.yml not found" }
        publishedProjects.forEach { proj ->
            require(proj.plugins.hasPlugin("maven-publish")) {
                "maven-publish plugin not applied to ${proj.name}"
            }
            val publishing = proj.extensions.getByType<PublishingExtension>()
            require("maven" in publishing.publications.names) {
                "Maven publication not configured for ${proj.name}"
            }
            println("  Maven publication configured: ${proj.name}")
        }
    }
}

tasks.register("verifyReleaseDependencies") {
    group = "verification"
    description =
        "Reject source composites, snapshots, and non-release Vectors dependencies from published modules"
    doLast {
        val settingsText = file("settings.gradle.kts").readText()
        require("""includeBuild("../vectors")""" !in settingsText) {
            "Release builds must consume Vectors from Maven Central, not ../vectors"
        }

        val expectedVectorsVersion =
            providers.gradleProperty("vectorsVersion").orNull
                ?: error("gradle.properties must declare vectorsVersion")
        val vectorsDependencies = mutableListOf<String>()
        val publishedDependencyBuckets = setOf("api", "implementation", "runtimeOnly")

        publishedProjects.forEach { proj ->
            proj.configurations
                .matching { configuration -> configuration.name in publishedDependencyBuckets }
                .forEach { configuration ->
                configuration.dependencies
                    .withType<ExternalModuleDependency>()
                    .forEach { dependency ->
                        val coordinate =
                            "${dependency.group}:${dependency.name}:${dependency.version}"
                        require(dependency.version?.endsWith("-SNAPSHOT") != true) {
                            "${proj.name} publishes a snapshot dependency: $coordinate"
                        }
                        if (dependency.group == "com.integrallis" &&
                            dependency.name.startsWith("vectors")) {
                            vectorsDependencies += coordinate
                            require(dependency.version == expectedVectorsVersion) {
                                "${proj.name} must use Vectors $expectedVectorsVersion, not $coordinate"
                            }
                        }
                    }
            }
        }

        require(vectorsDependencies.isNotEmpty()) {
            "No published module declares a released Vectors dependency"
        }
        vectorsDependencies.distinct().sorted().forEach { coordinate ->
            println("  Released Vectors dependency: $coordinate")
        }
    }
}

tasks.register("verifyStagedPublications") {
    group = "verification"
    description = "Stage and validate every Maven Central artifact and its internal dependencies"
    dependsOn(publishedProjects.map { "${it.path}:publishMavenPublicationToStagingRepository" })
    val aggregateNativeRelease =
        providers.gradleProperty("modelsNativeArtifactDirectory").isPresent
    inputs.property("aggregateNativeRelease", aggregateNativeRelease)
    doLast {
        val releaseVersion = project.version.toString()
        val stagingRoot = layout.buildDirectory.dir("staging-deploy/com/integrallis").get().asFile
        val internalDependency = Regex(
            """<dependency>\s*<groupId>com\.integrallis</groupId>\s*<artifactId>([^<]+)</artifactId>"""
        )
        publishedProjects.forEach { proj ->
            val versionDir = stagingRoot.resolve("${proj.name}/$releaseVersion")
            val pomFile = versionDir.listFiles()
                ?.filter { it.isFile && it.name.endsWith(".pom") }
                ?.maxByOrNull { it.lastModified() }
                ?: error("Missing staged POM in $versionDir")
            val artifactBase = pomFile.name.removeSuffix(".pom")
            listOf(
                "$artifactBase.jar",
                "$artifactBase-sources.jar",
                "$artifactBase-javadoc.jar"
            ).forEach { name ->
                require(versionDir.resolve(name).isFile) {
                    "Missing staged artifact: ${versionDir.resolve(name)}"
                }
            }
            if (proj.name == "backend-native") {
                JarFile(versionDir.resolve("$artifactBase.jar")).use { jar ->
                    val platforms =
                        listOf(
                            "linux-x86_64",
                            "linux-aarch64",
                            "macos-x86_64",
                            "macos-aarch64",
                            "windows-x86_64",
                            "windows-aarch64"
                        )
                    if (aggregateNativeRelease) {
                        platforms.forEach { platform ->
                            require(
                                jar.getJarEntry(
                                    "META-INF/models/native/$platform/native.properties"
                                ) != null
                            ) {
                                "Staged backend-native JAR is missing $platform resources"
                            }
                        }
                    } else {
                        val unexpectedNativeResources =
                            jar.entries().asSequence().filter {
                                it.name.startsWith("META-INF/models/native/")
                            }.map { it.name }.toList()
                        require(unexpectedNativeResources.isEmpty()) {
                            "Bridge-only backend-native staging must not contain a partial " +
                                "platform bundle: $unexpectedNativeResources"
                        }
                    }
                    require(
                        jar.getJarEntry(
                            "com/integrallis/models/backend/purejava/PureJavaBackend.class"
                        ) == null
                    ) {
                        "Staged backend-native JAR must not shade its dependencies"
                    }
                }
            }

            val pom = pomFile.readText()
            require("<licenses>" in pom && "<developers>" in pom && "<scm>" in pom) {
                "Incomplete Maven Central metadata in ${proj.name} POM"
            }
            internalDependency.findAll(pom).forEach { match ->
                val artifactId = match.groupValues[1]
                if (artifactId in libraryModuleNames) {
                    require(artifactId in publishedModuleNames) {
                        "${proj.name} publishes an unavailable internal dependency: $artifactId"
                    }
                }
            }
            println("  Staged publication valid: ${proj.name}")
        }
    }
}

tasks.register("verifyReproducibleBuild") {
    group = "verification"
    description = "Verify JAR tasks are configured for reproducible builds"
    doLast {
        libraryProjects.forEach { proj ->
            proj.tasks.withType<Jar>().forEach { jar ->
                require(!jar.isPreserveFileTimestamps) {
                    "preserveFileTimestamps must be false for ${proj.name}:${jar.name}"
                }
                require(jar.isReproducibleFileOrder) {
                    "reproducibleFileOrder must be true for ${proj.name}:${jar.name}"
                }
            }
            println("  Reproducible JARs: ${proj.name}")
        }
    }
}

tasks.register("verifyGithubWorkflows") {
    group = "verification"
    description = "Verify GitHub Actions workflow files exist"
    doLast {
        val workflowDir = rootProject.file(".github/workflows")
        listOf(
            "ci.yml",
            "docs.yml",
            "scorecard.yml",
            "codeql.yml",
            "mfcqi.yml",
            "release.yml",
            "native-kernels.yml"
        ).forEach { name ->
            val f = workflowDir.resolve(name)
            require(f.exists()) { "Missing workflow: ${f.absolutePath}" }
            require(f.readText().contains("jobs:")) { "$name missing 'jobs:' section" }
            println("  Workflow: $name")
        }
        val mfcqiWorkflow = workflowDir.resolve("mfcqi.yml").readText()
        require("'backend-*/.github/badges/mfcqi.json'" in mfcqiWorkflow) {
            "MFCQI workflow must commit badges for the public backend modules"
        }
        require(
            "repository: ModelJars/modeljars" in mfcqiWorkflow &&
                "path: model-jars" in mfcqiWorkflow
        ) {
            "MFCQI workflow must checkout the required ModelJars composite build"
        }

        val docsWorkflow = workflowDir.resolve("docs.yml").readText()
        require(
            "working-directory: models" in docsWorkflow &&
                "repository: ModelJars/modeljars" in docsWorkflow &&
                Regex("""(?m)^\s+path: models\s*$""").containsMatchIn(docsWorkflow) &&
                Regex("""(?m)^\s+path: model-jars\s*$""").containsMatchIn(docsWorkflow) &&
                "cache-dependency-path: models/docs/package-lock.json" in docsWorkflow &&
                "path: models/docs/build/public" in docsWorkflow
        ) {
            "Docs workflow must checkout Models and ModelJars as sibling composite builds"
        }

        val nativeWorkflow = workflowDir.resolve("native-kernels.yml").readText()
        val modelJarsDownloads =
            Regex(
                """uses:\s+actions/download-artifact@v8\s+with:\s+""" +
                    """name:\s+modeljars-composite-sources\s+path:\s+model-jars"""
            )
        require(modelJarsDownloads.findAll(nativeWorkflow).count() == 4) {
            "Every native workflow job must download ModelJars into the composite-build path"
        }
    }
}

val verifyNotebookDocker by tasks.registering {
    group = "verification"
    description = "Verify that notebook containers never inherit host registry credentials"

    val dockerConfigFile = file("notebooks/.docker-public/config.json")
    val dockerStateIgnoreFile = file("notebooks/.docker-public/.gitignore")
    val composeLauncher = file("notebooks/docker-compose.sh")
    val dockerfile = file("notebooks/jupyter/Dockerfile")
    val documentation =
        files(
            "notebooks/README.md",
            "docs/content/modules/ROOT/pages/notebooks.adoc"
        )
    inputs.files(
        dockerConfigFile,
        dockerStateIgnoreFile,
        composeLauncher,
        dockerfile,
        documentation
    )

    doLast {
        require(dockerConfigFile.isFile) {
            "Missing credential-free notebook Docker configuration: $dockerConfigFile"
        }
        val dockerConfig = JsonSlurper().parse(dockerConfigFile) as Map<*, *>
        require("credsStore" !in dockerConfig && "credHelpers" !in dockerConfig) {
            "Notebook Docker configuration must not invoke credential helpers"
        }
        require((dockerConfig["auths"] as? Map<*, *>)?.isEmpty() == true) {
            "Notebook Docker configuration must use anonymous registry access"
        }
        require(
            dockerStateIgnoreFile.isFile &&
                "*" in dockerStateIgnoreFile.readLines() &&
                "!config.json" in dockerStateIgnoreFile.readLines()
        ) {
            "Notebook Docker runtime state must be excluded from version control"
        }

        require(composeLauncher.isFile && composeLauncher.canExecute()) {
            "Notebook Docker launcher must exist and be executable: $composeLauncher"
        }
        val launcher = composeLauncher.readText()
        require("DOCKER_CONFIG" in launcher && ".docker-public" in launcher) {
            "Notebook Docker launcher must isolate Docker from host credentials"
        }
        require(
            "DOCKER_HOST" in launcher &&
                "unset DOCKER_CONTEXT" in launcher &&
                "docker-buildx" in launcher
        ) {
            "Notebook Docker launcher must preserve the daemon endpoint and Docker plugins"
        }

        documentation.forEach { document ->
            val text = document.readText()
            require("docker compose " !in text && "docker-compose " !in text) {
                "${document.path} must use ./docker-compose.sh for notebook containers"
            }
        }

        val dockerfileText = dockerfile.readText()
        require("--ServerApp.token=" in dockerfileText && "--ServerApp.password=" in dockerfileText) {
            "Notebook Jupyter server must start without credential prompts"
        }
    }
}

tasks.register("verifyNotebooks") {
    group = "verification"
    description = "Verify that JJava notebooks are portable, executed, and free of errors"
    dependsOn(verifyNotebookDocker)

    val notebookFiles = fileTree("notebooks") {
        include("*.ipynb")
    }
    val kernelFile = file("notebooks/jupyter/kernel.json")
    val notebookReadme = file("notebooks/README.md")
    inputs.files(notebookFiles, kernelFile, notebookReadme)

    doLast {
        val expectedOutput =
            mapOf(
                "01_getting_started.ipynb" to
                    listOf(
                        "model: ModelsNano",
                        "family: llama",
                        "vocabulary: 32",
                        "logits: 32",
                        "backend: pure-java"
                    ),
                "02_framework_adapters.ipynb" to
                    listOf(
                        "plain java: local answer",
                        "langchain4j: local answer",
                        "spring ai: local answer",
                        "spring stream: local answer",
                        "diagnostics: notebook"
                    ),
                "03_guarded_rag.ipynb" to
                    listOf(
                        "cited: MODEL_ANSWER",
                        "derived: MODEL_ANSWER_WITH_DERIVED_CITATIONS",
                        "weak retrieval: RETRIEVAL_ABSTENTION",
                        "unsupported: EXTRACTIVE_FALLBACK"
                    )
            )
        val actualNotebookNames = notebookFiles.files.map(File::getName).toSet()
        require(actualNotebookNames == expectedOutput.keys) {
            "Notebook inventory differs. Expected ${expectedOutput.keys.sorted()}, " +
                "found ${actualNotebookNames.sorted()}"
        }

        val forbiddenText =
            mapOf(
                "0.1.0-SNAPSHOT" to "hardcoded snapshot dependency",
                "/build/libs/" to "hardcoded module JAR path",
                "%classpath" to "notebook-local classpath setup",
                "%%loadFromPOM" to "notebook-local Maven dependency setup",
                "models-backend-purejava" to "retired backend artifact name",
                "models-backend-java" to "retired backend artifact name"
            )

        fun fragments(value: Any?): String =
            when (value) {
                is List<*> -> value.joinToString("") { it.toString() }
                null -> ""
                else -> value.toString()
            }

        notebookFiles.files.sorted().forEach { notebookFile ->
            val notebook = JsonSlurper().parse(notebookFile) as Map<*, *>
            val metadata = notebook["metadata"] as? Map<*, *> ?: emptyMap<Any, Any>()
            val kernelSpec = metadata["kernelspec"] as? Map<*, *> ?: emptyMap<Any, Any>()
            require(kernelSpec["name"] == "java") {
                "${notebookFile.name} must use the JJava kernel"
            }

            val cells = notebook["cells"] as? List<*> ?: emptyList<Any>()
            val executionSourceDigest = MessageDigest.getInstance("SHA-256")
            cells.mapNotNull { it as? Map<*, *> }.forEach { cell ->
                executionSourceDigest.update(cell["cell_type"].toString().toByteArray())
                executionSourceDigest.update(0)
                executionSourceDigest.update(fragments(cell["source"]).toByteArray())
                executionSourceDigest.update(0)
            }
            val executionSourceSha256 =
                executionSourceDigest.digest().joinToString("") { "%02x".format(it) }
            val modelsMetadata = metadata["models"] as? Map<*, *> ?: emptyMap<Any, Any>()
            require(modelsMetadata["execution_source_sha256"] == executionSourceSha256) {
                "${notebookFile.name} outputs were not executed from its current cell sources"
            }

            val codeCells =
                cells.mapNotNull { it as? Map<*, *> }.filter { it["cell_type"] == "code" }
            require(codeCells.isNotEmpty()) {
                "${notebookFile.name} must contain executable code"
            }
            require(codeCells.all { it["execution_count"] is Number }) {
                "${notebookFile.name} must ship with every code cell executed"
            }

            val source =
                cells.joinToString("\n") { cell ->
                    fragments((cell as? Map<*, *>)?.get("source"))
                }
            forbiddenText.forEach { (needle, description) ->
                require(needle !in source) {
                    "${notebookFile.name} contains $description: $needle"
                }
            }

            val outputs =
                codeCells.flatMap { cell ->
                    (cell["outputs"] as? List<*> ?: emptyList<Any>())
                        .mapNotNull { it as? Map<*, *> }
                }
            require(outputs.none { it["output_type"] == "error" }) {
                "${notebookFile.name} contains checked-in execution errors"
            }
            require(
                outputs.none {
                    it["output_type"] == "stream" && it["name"] == "stderr"
                }
            ) {
                "${notebookFile.name} contains checked-in stderr output"
            }

            val renderedOutput =
                outputs.joinToString("") { output ->
                    when (output["output_type"]) {
                        "stream" -> fragments(output["text"])
                        "execute_result", "display_data" ->
                            fragments((output["data"] as? Map<*, *>)?.get("text/plain"))
                        else -> ""
                    }
                }
            require(!Regex("""\b(?:AssertionError|Exception|ERROR)\b""").containsMatchIn(renderedOutput)) {
                "${notebookFile.name} contains suspicious exception or error output"
            }
            expectedOutput.getValue(notebookFile.name).forEach { expected ->
                require(expected in renderedOutput) {
                    "${notebookFile.name} is missing checked-in output: $expected"
                }
            }
            println("  Notebook: ${notebookFile.name}")
        }

        require(kernelFile.isFile) { "Missing JJava kernelspec: $kernelFile" }
        val kernel = JsonSlurper().parse(kernelFile) as Map<*, *>
        val kernelEnv = kernel["env"] as? Map<*, *> ?: emptyMap<Any, Any>()
        require(
            kernelEnv["JJAVA_CLASSPATH"] ==
                "/home/jovyan/work/models/build/notebooks/classpath/*"
        ) {
            "JJava kernel must load the generated notebook classpath"
        }
        require(notebookReadme.isFile) { "Missing notebook README" }
        require("install.sh" !in notebookReadme.readText()) {
            "Notebook README references a nonexistent install script"
        }
    }
}

tasks.register("verifyDocumentation") {
    group = "verification"
    description = "Verify the Antora site, landing page, and documentation toolchain"

    val readmeFile = file("README.md")
    val docsFiles = fileTree("docs") {
        include("**/*.adoc", "**/*.html", "**/*.hbs", "**/*.yml", "**/*.yaml")
        exclude("build/**", "node_modules/**", ".gradle/**", "content/**/attachments/**")
    }
    val docsPackageFile = file("docs/package.json")
    val docsLockFile = file("docs/package-lock.json")
    val docsBuildFile = file("docs/build.gradle")
    val ciWorkflowFile = file(".github/workflows/ci.yml")
    val docsWorkflowFile = file(".github/workflows/docs.yml")
    val releaseWorkflowFile = file(".github/workflows/release.yml")
    inputs.files(
        readmeFile,
        docsFiles,
        docsPackageFile,
        docsLockFile,
        docsBuildFile,
        ciWorkflowFile,
        docsWorkflowFile,
        releaseWorkflowFile
    )

    doLast {
        require(docsLockFile.isFile) {
            "docs/package-lock.json must be committed for reproducible documentation builds"
        }

        val malformedPages =
            docsFiles.files.filter { documentation ->
                documentation.extension == "adoc"
                    && (
                        documentation.readText().contains("++_++")
                            || documentation.readText().contains("link:architecture.md")
                            || documentation.readLines().firstOrNull()?.startsWith("== ") == true
                    )
            }
        require(malformedPages.isEmpty()) {
            "Documentation contains malformed converted AsciiDoc: " +
                malformedPages.joinToString { it.relativeTo(rootDir).path }
        }

        val readme = readmeFile.readText()
        require("https://integrallis.github.io/models/" in readme) {
            "README must link to the published documentation site"
        }
        require("notebooks/README.md" in readme) {
            "README must link to the executable notebooks"
        }

        val docsIndex = file("docs/content/modules/ROOT/pages/index.adoc").readText()
        publishedModuleNames.forEach { module ->
            require("`$module`" in docsIndex) {
                "Documentation module inventory is missing $module"
            }
        }

        val conceptsFile = file("docs/content/modules/ROOT/pages/concepts.adoc")
        require(conceptsFile.isFile) {
            "Documentation must define the inference concepts used by technical pages"
        }
        val concepts = conceptsFile.readText()
        listOf(
            "== GGUF And Model Artifacts",
            "== Tokens, Prefill, And Decode",
            "== Tensors, Kernels, And SIMD",
            "== Performance Measurements",
            "== Retrieval-Augmented Generation"
        ).forEach { section ->
            require(section in concepts) {
                "Inference concepts page is missing $section"
            }
        }
        require("xref:concepts.adoc" in file("docs/content/modules/ROOT/nav.adoc").readText()) {
            "Documentation navigation must link to the inference concepts page"
        }

        val architectureSource = file("media/diagrams/models-0001.svg")
        require(architectureSource.isFile) {
            "Architecture diagram must retain an editable source"
        }
        val architectureSvg = architectureSource.readText()
        require("backend-java" in architectureSvg && "models-backend-purejava" !in architectureSvg) {
            "Architecture diagram must use the current backend-java module name"
        }
        val architecturePng = file("media/diagrams/models-0001.png")
        val documentationArchitecturePng =
            file("docs/content/modules/ROOT/images/models-0001.png")
        require(architecturePng.readBytes().contentEquals(documentationArchitecturePng.readBytes())) {
            "README and documentation architecture diagrams must be identical"
        }

        val frameworkGuide =
            file("docs/content/modules/ROOT/pages/framework-integrations.adoc").readText()
        val requiredFrameworkExamples =
            listOf(
                "new ModelsChatModel(",
                "new ModelsSpringAiChatModel(",
                "model.stream(new Prompt(",
                "ModelJarDescriptor descriptor",
                "GroundedAnswerPolicy.productionDefault()",
                "new VectorCollectionEmbeddingSink("
            )
        requiredFrameworkExamples.forEach { example ->
            require(example in frameworkGuide) {
                "Framework integration guide is missing an executable example containing $example"
            }
        }
        require(frameworkGuide.windowed("[source,java]".length).count { it == "[source,java]" } >= 6) {
            "Framework integration guide must contain Java examples for every supported adapter"
        }

        val landingPage = file("docs/landing/index.html").readText()
        require("backend-java" in landingPage && "backend-native" in landingPage) {
            "Landing page must describe both Models execution paths"
        }
        require("models-backend-java" !in landingPage) {
            "Landing page must use the canonical backend-java artifact name"
        }
        require("Java 25" in landingPage && "Vector API" in landingPage) {
            "Landing page must state the shared Java 25 and Vector API runtime"
        }

        val docsPackage = JsonSlurper().parse(docsPackageFile) as Map<*, *>
        val dependencies = docsPackage["dependencies"] as? Map<*, *> ?: emptyMap<Any, Any>()
        listOf("@antora/cli", "@antora/site-generator", "@antora/site-generator-default")
            .forEach { dependency ->
                require(dependencies[dependency] == "3.1.12") {
                    "$dependency must be pinned exactly to 3.1.12"
                }
            }
        val overrides = docsPackage["overrides"] as? Map<*, *> ?: emptyMap<Any, Any>()
        require(overrides["js-yaml"] == "4.3.0") {
            "Documentation dependencies must override js-yaml to 4.3.0"
        }
        require("args = ['ci']" in docsBuildFile.readText()) {
            "Documentation build must use npm ci"
        }
        require(":docs:build" in ciWorkflowFile.readText()) {
            "CI must execute the documentation build"
        }
        require(
            "test-notebooks.sh" in ciWorkflowFile.readText() &&
                "./docker-compose.sh" in ciWorkflowFile.readText()
        ) {
            "CI must execute Java notebooks through the credential-free Docker launcher"
        }
        require(":docs:build" in docsWorkflowFile.readText()) {
            "GitHub Pages workflow must execute the documentation build"
        }
        require(":docs:build" in releaseWorkflowFile.readText()) {
            "Release workflow must execute the documentation build"
        }
        require(
            "-PnotebookMode=release" in releaseWorkflowFile.readText() &&
                "test-notebooks.sh" in releaseWorkflowFile.readText() &&
                "./docker-compose.sh" in releaseWorkflowFile.readText()
        ) {
            "Release workflow must execute notebooks against the staged artifacts"
        }
        require(
            "-PmodelsNativeArtifactDirectory=native-release-artifacts" in
                releaseWorkflowFile.readText()
        ) {
            "Release workflow must aggregate every native platform before Maven staging"
        }
    }
}

tasks.register("complianceCheck") {
    group = "verification"
    description = "Run all compliance verification tasks"
    dependsOn(
        "verifySbom",
        "verifyGovernanceFiles",
        "verifyLockfiles",
        "verifyPublishingConfigured",
        "verifyReleaseDependencies",
        "verifyStagedPublications",
        "verifyReproducibleBuild",
        "verifyGithubWorkflows",
        "verifyNotebooks",
        "verifyDocumentation"
    )
}

tasks.register<Sync>("prepareNotebookClasspath") {
    group = "documentation"
    description = "Prepare the JJava runtime classpath from source or released artifacts"

    val selectedMode = notebookMode.map(String::lowercase)
    val selectedClasspath =
        when (selectedMode.get()) {
            "source" -> notebookSourceClasspath
            "release" -> notebookReleaseClasspath
            else ->
                throw GradleException(
                    "Unsupported notebookMode '${notebookMode.get()}'; expected source or release"
                )
        }

    from(selectedClasspath)
    into(layout.buildDirectory.dir("notebooks/classpath"))
    duplicatesStrategy = DuplicatesStrategy.FAIL
    inputs.property("notebookMode", selectedMode)
    inputs.property("notebookVersion", notebookVersion)
    inputs.property("notebookRepository", notebookRepositoryUrl.orElse(""))

    doLast {
        val details =
            if (selectedMode.get() == "release") {
                "com.integrallis ${notebookVersion.get()}"
            } else {
                "current project sources"
            }
        println("  Notebook classpath: ${destinationDir.absolutePath}")
        println("  Dependency mode: $details")
    }
}

tasks.wrapper {
    gradleVersion = "9.4.1"
    distributionType = Wrapper.DistributionType.ALL
}

val cleanCollectedJavadocs by tasks.registering(Delete::class) {
    description = "Remove stale Javadocs before collecting the published API"
    group = "documentation"
    delete(layout.buildDirectory.dir("docs/javadoc"))
}

// Aggregated Javadoc generation
tasks.register<Javadoc>("aggregateJavadoc") {
    description = "Generate aggregated Javadoc for all library modules"
    group = "documentation"
    dependsOn(cleanCollectedJavadocs)
    javadocTool.set(
        javaToolchains.javadocToolFor {
            languageVersion = JavaLanguageVersion.of(25)
        }
    )

    val libProjects = publishedProjects.filter { it.name != "models" }
    libProjects.forEach { proj ->
        dependsOn(proj.tasks.named("compileJava"))
        source(proj.the<SourceSetContainer>()["main"].allJava)
        classpath += files(proj.the<SourceSetContainer>()["main"].compileClasspath)
    }
    setDestinationDir(layout.buildDirectory.dir("docs/javadoc/aggregate").get().asFile)

    (options as StandardJavadocDocletOptions).apply {
        title = "Models ${project.version} API"
        windowTitle = "Models ${project.version}"
        author(true)
        version(true)
        use(true)
        splitIndex(true)
        links("https://docs.oracle.com/en/java/javase/25/docs/api/")
        addStringOption("Xdoclint:-missing", "-quiet")
    }

    isFailOnError = true
}

val perModuleJavadocCopies =
    publishedProjects
        .filter { it.name != "models" }
        .map { proj ->
            tasks.register<Sync>("copyJavadoc_${proj.name}") {
                description = "Collects ${proj.name} Javadoc into the documentation tree"
                group = "documentation"
                dependsOn(cleanCollectedJavadocs)
                from(proj.tasks.named<Javadoc>("javadoc"))
                into(layout.buildDirectory.dir("docs/javadoc/modules/${proj.name}"))
            }
        }

tasks.register("generateModuleJavadocs") {
    description = "Generate Javadoc for individual published modules"
    group = "documentation"
    dependsOn(perModuleJavadocCopies)
}
