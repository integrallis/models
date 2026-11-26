import java.nio.file.Path
import java.util.zip.ZipFile

description = "Spring Boot auto-configuration starter for Models and Spring AI"

val springAiVersion = providers.gradleProperty("springAiVersion").getOrElse("2.0.0")
val springBootVersion = providers.gradleProperty("springBootVersion").getOrElse("4.1.0")
val springFrameworkVersion = providers.gradleProperty("springFrameworkVersion").getOrElse("7.0.8")
val micrometerObservationVersion =
    providers.gradleProperty("micrometerObservationVersion").getOrElse("1.16.6")

// --- Auto-configuration that only exists from Spring AI 2.1 (DecisionModel) --------------------
//
// The same pattern as models-spring-ai: the main source set is compiled against the baseline
// Spring AI line, and the DecisionModel auto-configuration, which names 2.1-only types, lives in
// its own source set compiled against 2.1 and packed into the same jar. It is listed in the
// AutoConfiguration.imports file and guarded by @ConditionalOnClass(name = ...), which Spring Boot
// evaluates from the class file's metadata, so on an older line the class is never loaded.
val springAi21Version = providers.gradleProperty("springAi21Version").getOrElse("2.1.0")
// Spring AI 2.1 is built on Spring Boot 4.2 / Framework 7.1; its tests run on that line.
val springBoot42Version = providers.gradleProperty("springBoot42Version").getOrElse("4.2.0")
val springFramework71Version =
    providers.gradleProperty("springFramework71Version").getOrElse("7.1.0")

val springAi21: SourceSet by sourceSets.creating {
    compileClasspath += sourceSets["main"].output
    runtimeClasspath += sourceSets["main"].output
}
val springAi21Test: SourceSet by sourceSets.creating {
    compileClasspath += sourceSets["main"].output + springAi21.output
    runtimeClasspath += sourceSets["main"].output + springAi21.output
}
configurations[springAi21.implementationConfigurationName].extendsFrom(
    configurations["implementation"],
    configurations["api"],
)
configurations[springAi21.compileOnlyConfigurationName].extendsFrom(configurations["compileOnly"])
configurations[springAi21.annotationProcessorConfigurationName].extendsFrom(
    configurations["annotationProcessor"],
)
configurations[springAi21Test.implementationConfigurationName].extendsFrom(
    configurations["testImplementation"],
)
configurations[springAi21Test.runtimeOnlyConfigurationName].extendsFrom(
    configurations["testRuntimeOnly"],
)
// models-spring-ai packs its own 2.1 adapter into its jar, not its classes directory, so these
// compile classpaths ask project dependencies for the jar, where ModelsSpringAiDecisionModel is.
listOf(springAi21.compileClasspathConfigurationName, springAi21Test.compileClasspathConfigurationName)
    .forEach { name ->
        configurations[name].attributes {
            attribute(
                LibraryElements.LIBRARY_ELEMENTS_ATTRIBUTE,
                objects.named(LibraryElements::class.java, LibraryElements.JAR),
            )
        }
    }
// The ordinary test source set runs against the jar as published: the 2.1 class files are present,
// Spring AI 2.1 is not (on the baseline line). That is the situation an application on Spring AI
// 2.0 or 1.1 is in, and the baseline test checks it starts cleanly.
sourceSets["test"].runtimeClasspath += springAi21.output

dependencies {
    api(project(":models-spring-ai"))

    compileOnly("org.springframework.boot:spring-boot-autoconfigure:$springBootVersion")
    compileOnly("org.springframework.ai:spring-ai-model:$springAiVersion")
    // ChatClient lives here, and it stays compileOnly: an application that only wants the
    // ChatModel adapter must not be made to carry the client.
    compileOnly("org.springframework.ai:spring-ai-client-chat:$springAiVersion")
    // The engine, for integrallis.models.catalog-model. compileOnly on purpose: an application
    // that supplies its own model bean needs no decoder from this starter, and one that uses the
    // property has backend-java on the classpath anyway.
    compileOnly(project(":backend-java"))
    annotationProcessor(
        "org.springframework.boot:spring-boot-configuration-processor:$springBootVersion"
    )

    testImplementation("org.springframework.boot:spring-boot-test:$springBootVersion")
    testImplementation("org.springframework.boot:spring-boot-autoconfigure:$springBootVersion")
    testImplementation("org.springframework.boot:spring-boot-starter-actuator:$springBootVersion")
    testImplementation("org.springframework:spring-test:$springFrameworkVersion")
    testImplementation("org.springframework.ai:spring-ai-model:$springAiVersion")
    testImplementation("org.springframework.ai:spring-ai-client-chat:$springAiVersion")
    testImplementation(project(":backend-java"))
    testImplementation("org.springframework.ai:spring-ai-vector-store:$springAiVersion")
    testImplementation("io.micrometer:micrometer-observation:$micrometerObservationVersion")
    testImplementation(
        "com.integrallis:vectors-spring-boot-starter:" +
            providers.gradleProperty("vectorsVersion").get()
    )

    add(springAi21.compileOnlyConfigurationName, "org.springframework.ai:spring-ai-model:$springAi21Version")
    add(springAi21Test.implementationConfigurationName, "org.springframework.ai:spring-ai-model:$springAi21Version")
    add(springAi21Test.implementationConfigurationName, "org.springframework.boot:spring-boot-test:$springBoot42Version")
    add(springAi21Test.implementationConfigurationName, "org.springframework.boot:spring-boot-autoconfigure:$springBoot42Version")
    add(springAi21Test.implementationConfigurationName, "org.springframework:spring-test:$springFramework71Version")
}

// Both source sets run the configuration processor. The jar carries one metadata file holding both
// sets of properties, so integrallis.models.decision.* is known to IDEs like the rest.
val mergedConfigurationMetadata = layout.buildDirectory.file("merged-metadata/META-INF/spring-configuration-metadata.json")
val mergeConfigurationMetadata =
    tasks.register("mergeConfigurationMetadata") {
        val mainMeta = sourceSets["main"].output.classesDirs.files.map { it.resolve("META-INF/spring-configuration-metadata.json") }
        val decisionMeta = springAi21.output.classesDirs.files.map { it.resolve("META-INF/spring-configuration-metadata.json") }
        dependsOn(tasks.named("compileJava"), tasks.named(springAi21.compileJavaTaskName))
        outputs.file(mergedConfigurationMetadata)
        doLast {
            val merged = linkedMapOf<String, MutableList<Any?>>(
                "groups" to mutableListOf(), "properties" to mutableListOf(), "hints" to mutableListOf(),
            )
            (mainMeta + decisionMeta).filter { it.isFile }.forEach { file ->
                @Suppress("UNCHECKED_CAST")
                val json = groovy.json.JsonSlurper().parse(file) as Map<String, Any?>
                merged.keys.forEach { key -> (json[key] as? List<*>)?.let { merged.getValue(key).addAll(it) } }
            }
            val out = mergedConfigurationMetadata.get().asFile
            out.parentFile.mkdirs()
            out.writeText(groovy.json.JsonOutput.prettyPrint(groovy.json.JsonOutput.toJson(merged)))
        }
    }
tasks.named<Jar>("jar") {
    dependsOn(mergeConfigurationMetadata)
    from(springAi21.output)
    from(layout.buildDirectory.dir("merged-metadata"))
    // Keep only the merged copy of the metadata file; drop the two generated ones.
    exclude { element ->
        element.relativePath.pathString == "META-INF/spring-configuration-metadata.json" &&
            !element.file.path.contains("merged-metadata")
    }
}
tasks.matching { it.name == "sourcesJar" }.configureEach {
    (this as Jar).from(springAi21.allSource)
}
tasks.named<Javadoc>("javadoc") {
    source(springAi21.allJava)
    classpath = springAi21.compileClasspath + classpath
}

val springAi21TestTask =
    tasks.register<Test>("springAi21Test") {
        description = "Tests the Spring AI 2.1 auto-configuration on Spring AI $springAi21Version"
        group = "verification"
        testClassesDirs = springAi21Test.output.classesDirs
        classpath = springAi21Test.runtimeClasspath
        useJUnitPlatform {
            excludeTags("slow", "benchmark", "integration", "model-fixture")
        }
    }
// Fails the build unless the published jar carries the 2.1 auto-configuration and lists it.
val verifyDecisionAutoConfigurationInJar =
    tasks.register("verifyDecisionAutoConfigurationInJar") {
        description = "Fails unless the jar contains and imports the DecisionModel auto-configuration"
        group = "verification"
        val archive = tasks.named<Jar>("jar").flatMap { it.archiveFile }
        inputs.file(archive)
        doLast {
            val name = "com.integrallis.models.spring.boot.decision.ModelsDecisionAutoConfiguration"
            val imports = "META-INF/spring/org.springframework.boot.autoconfigure.AutoConfiguration.imports"
            ZipFile(archive.get().asFile).use { zip ->
                if (zip.getEntry(name.replace('.', '/') + ".class") == null) {
                    throw GradleException("${archive.get().asFile.name} does not contain $name")
                }
                val listed = zip.getInputStream(zip.getEntry(imports)).bufferedReader().readLines()
                if (name !in listed.map { it.trim() }) {
                    throw GradleException("$imports does not list $name")
                }
                val metadata = zip.getInputStream(zip.getEntry("META-INF/spring-configuration-metadata.json"))
                    .bufferedReader().readText()
                listOf("integrallis.models.decision.enabled", "integrallis.models.chat-template").forEach {
                    if ("\"$it\"" !in metadata) {
                        throw GradleException("spring-configuration-metadata.json does not describe $it")
                    }
                }
            }
        }
    }
tasks.named("check") {
    dependsOn(springAi21TestTask, verifyDecisionAutoConfigurationInJar)
}

// The auto-configured DecisionModel over a real model, through Spring Boot. Not part of `check`.
tasks.register<Test>("decisionAutoConfigurationIntegrationTest") {
    description = "Run the auto-configured DecisionModel end to end on the pinned Qwen3 1.7B artifact"
    group = "verification"
    testClassesDirs = springAi21Test.output.classesDirs
    classpath = springAi21Test.runtimeClasspath
    useJUnitPlatform {
        includeTags("integration")
    }
    systemProperty(
        "models.fixtures.directory",
        providers.systemProperty("models.fixtures.directory").orNull
            ?: Path.of(System.getProperty("user.home"), ".jvllm", "models").toString(),
    )
    providers.systemProperty("models.fixtures.decisionModel").orNull?.let {
        systemProperty("models.fixtures.decisionModel", it)
    }
    maxHeapSize = "6g"
    jvmArgs("--add-modules=jdk.incubator.vector", "--enable-native-access=ALL-UNNAMED")
}

tasks.withType<JacocoReportBase>().configureEach {
    dependsOn(springAi21TestTask)
    executionData(springAi21TestTask.get())
    sourceSets(springAi21)
}
