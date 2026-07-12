plugins {
    java
    `java-library`
    `maven-publish`
    id("com.github.spotbugs") version "6.4.4" apply false
    id("com.diffplug.spotless") version "6.25.0" apply false
    id("org.cyclonedx.bom") version "3.2.4" apply false
    id("org.owasp.dependencycheck") version "12.2.1" apply false
    jacoco
}

allprojects {
    group = "com.integrallis"

    repositories {
        mavenCentral()
    }
}

// Library subprojects (excludes benchmarks)
val libraryProjects = subprojects.filter { it.name != "models-bench" }
val libraryModuleNames = libraryProjects.map { it.name }.toSet()
val publishedModuleNames = setOf("models-api", "models-runtime", "models-backend-purejava")
val publishedProjects = libraryProjects.filter { it.name in publishedModuleNames }
val scaffoldProjects = libraryProjects.filterNot { it.name in publishedModuleNames }

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
        testImplementation("org.mockito:mockito-core:5.15.2")
        testImplementation("org.mockito:mockito-junit-jupiter:5.15.2")
        testRuntimeOnly("org.junit.platform:junit-platform-launcher:1.11.4")
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

tasks.register("verifyStagedPublications") {
    group = "verification"
    description = "Stage and validate every Maven Central artifact and its internal dependencies"
    dependsOn(publishedProjects.map { "${it.path}:publishMavenPublicationToStagingRepository" })
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
        listOf("ci.yml", "scorecard.yml", "codeql.yml", "release.yml").forEach { name ->
            val f = workflowDir.resolve(name)
            require(f.exists()) { "Missing workflow: ${f.absolutePath}" }
            require(f.readText().contains("jobs:")) { "$name missing 'jobs:' section" }
            println("  Workflow: $name")
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
        "verifyStagedPublications",
        "verifyReproducibleBuild",
        "verifyGithubWorkflows"
    )
}

tasks.wrapper {
    gradleVersion = "9.4.1"
    distributionType = Wrapper.DistributionType.ALL
}

// Aggregated Javadoc generation
tasks.register<Javadoc>("aggregateJavadoc") {
    description = "Generate aggregated Javadoc for all library modules"
    group = "documentation"

    val libProjects = libraryProjects.filter { it.name != "models-bench" }
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
