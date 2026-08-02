plugins {
    java
    application
    id("com.github.spotbugs")
}

val langchain4jVersion = providers.gradleProperty("langchain4jVersion").getOrElse("1.17.2")

java {
    toolchain {
        languageVersion = JavaLanguageVersion.of(25)
    }
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

tasks.withType<Test> {
    useJUnitPlatform()
    jvmArgs("--add-modules", "jdk.incubator.vector")
    systemProperty("models.repositoryRoot", rootProject.projectDir.absolutePath)
}

tasks.named<Test>("test") {
    useJUnitPlatform {
        excludeTags("slow")
    }
}

val nativeLibraryName =
    when {
        System.getProperty("os.name").startsWith("Mac", ignoreCase = true) ->
            "libjmodels_kernels.dylib"
        System.getProperty("os.name").startsWith("Windows", ignoreCase = true) ->
            "jmodels_kernels.dll"
        else -> "libjmodels_kernels.so"
    }
val nativeLibrary =
    project(":backend-native")
        .layout.buildDirectory.file("rust-target/release/$nativeLibraryName")

tasks.register<Test>("gemma426BA4BFrameworkSlowTest") {
    group = "verification"
    description =
        "Run the pinned Gemma 4 26B-A4B Q4_K_M plain-Java, LangChain4j, and Spring AI test"
    dependsOn(
        project(":backend-java").tasks.named("downloadGemma426BA4BQ4KMModel"),
        project(":backend-native").tasks.named("cargoBuildRelease"),
    )
    testClassesDirs = sourceSets["test"].output.classesDirs
    classpath = sourceSets["test"].runtimeClasspath
    useJUnitPlatform {
        includeTags("slow")
    }
    filter {
        includeTestsMatching("com.integrallis.models.rag.Gemma4FrameworkAdaptersSlowTest")
    }
    jvmArgs("--enable-native-access=ALL-UNNAMED")
    systemProperty("models.native.kernels.library", nativeLibrary.get().asFile.absolutePath)
    listOf(
        "models.fixtures.directory",
        "models.native.quantizedDecode",
        "models.native.loadWarmup",
        "models.native.kernels.threads",
    ).forEach { propertyName ->
        providers.systemProperty(propertyName).orNull?.let { value ->
            systemProperty(propertyName, value)
        }
    }
    outputs.upToDateWhen { false }
    maxParallelForks = 1
    maxHeapSize = "4g"
}

tasks.named("spotbugsTest") {
    enabled = false
}

application {
    mainClass = "com.integrallis.models.rag.RagBenchmarkCli"
    applicationDefaultJvmArgs = listOf("--add-modules", "jdk.incubator.vector")
}

tasks.withType<JavaExec>().configureEach {
    jvmArgs("--add-modules", "jdk.incubator.vector")
}

dependencies {
    implementation(project(":models-rag"))
    implementation(project(":models-runtime"))
    implementation(project(":backend-java"))
    implementation(project(":backend-native"))
    implementation(project(":models-langchain4j"))
    implementation(project(":models-spring-ai"))
    implementation("dev.langchain4j:langchain4j:$langchain4jVersion")
    implementation("org.apache.lucene:lucene-core:10.4.0")
    implementation("com.fasterxml.jackson.core:jackson-databind:2.21.4")
    implementation("org.springframework.ai:spring-ai-rag:2.0.0")

    testImplementation("org.junit.jupiter:junit-jupiter:5.11.4")
    testImplementation("org.assertj:assertj-core:3.27.2")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher:1.11.4")
}
