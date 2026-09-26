// models-bench — JMH benchmarks (not published as a library)

plugins {
    java
    application
    id("com.github.spotbugs")
    id("me.champeau.jmh") version "0.7.2"
}

java {
    toolchain {
        languageVersion = JavaLanguageVersion.of(25)
    }
}

val benchmarkJvmArgs =
    listOf(
        "--add-modules",
        "jdk.incubator.vector",
        "--enable-native-access=ALL-UNNAMED",
        "-XX:NativeMemoryTracking=summary",
    )

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
    jvmArgs(benchmarkJvmArgs)
}

tasks.named("spotbugsTest") {
    enabled = false
}

application {
    mainClass = "com.integrallis.models.bench.InferenceBenchmarkCli"
    applicationDefaultJvmArgs = benchmarkJvmArgs
}

val aggregateNativeRelease =
    providers.gradleProperty("modelsNativeArtifactDirectory").isPresent
val nativeBenchmarkRuntime =
    providers.gradleProperty("modelsBenchNative").map(String::toBoolean).getOrElse(false)

tasks.withType<JavaExec>().configureEach {
    jvmArgs(benchmarkJvmArgs)
    System.getProperty("models.native.kernels.library")?.let {
        systemProperty("models.native.kernels.library", it)
    }
}

dependencies {
    // The Rust PTX GPU arm. Present on every platform: the module's own fallback reports an
    // absent or ineligible device rather than failing, so a CPU-only host still builds and runs
    // the gate command (it reports accelerated=false, which is the G3 evidence).
    implementation(project(":backend-cuda"))
    implementation(project(":models-runtime"))
    implementation(project(":backend-java"))
    implementation(project(":models-router"))
    // The accelerator profile gate drives the Tornado backend directly. backend-tornado keeps
    // tornado-api off its public API, so nothing here needs it at compile time; the TornadoVM
    // launcher supplies the device runtime on a GPU host. Without one, TornadoBackend.open catches
    // the LinkageError and reports a Vector API fallback, which is what the report should say.
    implementation(project(":backend-tornado"))
    if (nativeBenchmarkRuntime && !aggregateNativeRelease) {
        // The platform artifact contains only the compiled library and metadata. The Java FFM
        // bridge lives in backend-native's ordinary JAR and must be present as well.
        runtimeOnly(project(":backend-native"))
        runtimeOnly(
            project(
                path = ":backend-native",
                configuration = "hostNativeRuntimeElements",
            )
        )
    }
    // JMH sources compile against the native FFM bridge classes even when the benchmark
    // runtime is not requested; the bridge JAR needs no Cargo build to compile.
    jmhImplementation(project(":backend-native"))
    implementation("com.integrallis:vectors-core:${providers.gradleProperty("vectorsVersion").get()}")
    implementation("com.fasterxml.jackson.core:jackson-databind:2.21.7")
    implementation("com.fasterxml.jackson.datatype:jackson-datatype-jdk8:2.21.4")
    testImplementation("org.junit.jupiter:junit-jupiter:5.11.4")
    testImplementation("org.assertj:assertj-core:3.27.2")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher:1.11.4")
}

if (nativeBenchmarkRuntime) {
    dependencies {
        testImplementation(project(":backend-native"))
    }
} else {
    sourceSets.named("test") {
        java.exclude("**/NativeBenchmarkRuntimeTest.java")
    }
}

jmh {
    jvmArgs.addAll(listOf(
        "--add-modules", "jdk.incubator.vector",
        "-Xms2g", "-Xmx8g",
        "-XX:+UseG1GC"
    ))

    (project.findProperty("bench.model") as String?)?.let {
        jvmArgs.add("-Dmodels.bench.model=$it")
    }
    (project.findProperty("jmh.includes") as String?)?.let {
        includes.set(listOf(it))
    }
    (project.findProperty("jmh.fork") as String?)?.let {
        fork.set(it.toInt())
    }
    (project.findProperty("jmh.warmup") as String?)?.let {
        warmupIterations.set(it.toInt())
    }
    (project.findProperty("jmh.iterations") as String?)?.let {
        iterations.set(it.toInt())
    }
    (project.findProperty("jmh.timeOnIteration") as String?)?.let {
        timeOnIteration.set(it)
    }
    resultFormat.set(project.findProperty("jmh.resultFormat") as String? ?: "JSON")
    val resultExtension = (resultFormat.get() as String).lowercase()
    resultsFile.set(project.file("build/results/jmh/results.$resultExtension"))
}
