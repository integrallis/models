// models-decisions — System One decisions: typed, bounded, calibrated judgements over model state.
//
// This module owns the decision contracts and the calibration mathematics only. It depends on the
// API tier so a head can read a hidden state, and deliberately pulls no backend: the evaluator that
// forks a shared physical prefix is wired by whichever backend implements the capability.

dependencies {
    api(project(":models-api"))

    // The capability probe drives the real pure-Java backend. Test-only: the module itself must
    // stay free of any backend so a decision head can be wired to whichever one is qualified.
    testImplementation(project(":backend-java"))
    // For JevBenchRunner's competitor-prompt arm, which renders a chat template rather than
    // guessing at one. Test-only; models-decisions itself stays free of the runtime.
    testImplementation(project(":models-runtime"))

    // The harvest tool drives the rust-ffm kernel, because the qualified answerability evidence is
    // on that backend and a figure taken on any other one would not be comparable with it.
    testImplementation(project(":backend-native"))

    testImplementation("org.junit.jupiter:junit-jupiter:5.11.4")
    testImplementation("org.assertj:assertj-core:3.27.2")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher:1.11.4")
}

// The capability probe is selected by DECISIONS_GRANITE_MODEL. Gradle does not treat an environment
// variable as a task input, so without this the probe can report a cached skip while the artifact is
// present. Declaring it means changing the variable invalidates the task, and a stale skip cannot
// masquerade as a run.
tasks.named<Test>("integrationTest") {
    inputs.property(
        "decisionsGraniteModel",
        providers.environmentVariable("DECISIONS_GRANITE_MODEL").orElse("absent")
    )
}

// The rust-ffm kernel is the qualified fast path and is 8x on this workload, but it only exists
// after the cargo build has run for the host platform. Every runnable task below depends on it and
// puts it on the classpath, because the alternative is a silent fallback that looks like a slow
// model rather than a missing build step.
val nativeResources = project(":backend-native").layout.buildDirectory
    .dir("generated/native-platform-resources")

fun JavaExec.withNativeKernel() {
    dependsOn(":backend-native:prepareNativePlatformResources")
    classpath += project.files(nativeResources)
}

// Cuts a release artifact from a harvest. The tool lives in the test source set because it drives
// a qualified backend, but the artifact it writes is a product of the main source set alone.
//
//   ./gradlew :models-decisions:release \
//       -Pharvest=/path/to/squad2.jsonl -Pcorpus=squad2 \
//       -Pbase=granite-4.1-3b -Pout=/path/to/model.idsn
tasks.register<JavaExec>("release") {
    group = "distribution"
    description = "Fit, calibrate, read the sealed split once, and write the decision artifact"
    classpath = sourceSets["test"].runtimeClasspath
    mainClass.set("com.integrallis.models.decisions.ReleaseTool")
    jvmArgs("--add-modules", "jdk.incubator.vector", "--enable-native-access=ALL-UNNAMED")
    args(
        providers.gradleProperty("harvest").get(),
        providers.gradleProperty("corpus").get(),
        providers.gradleProperty("base").get(),
        providers.gradleProperty("out").get(),
        providers.gradleProperty("baseFile").getOrElse(""),
    )
    withNativeKernel()
}

// Runs a released artifact over a JSONL of questions.
//
//   ./gradlew :models-decisions:decide \
//       -Partifact=model.idsn -Pbase=granite.gguf -Pin=questions.jsonl -Pout=verdicts.jsonl
tasks.register<JavaExec>("decide") {
    group = "application"
    description = "Answer a JSONL of questions with a released decision artifact"
    classpath = sourceSets["test"].runtimeClasspath
    mainClass.set("com.integrallis.models.decisions.DecideCli")
    jvmArgs("--add-modules", "jdk.incubator.vector", "--enable-native-access=ALL-UNNAMED")
    args(
        providers.gradleProperty("artifact").get(),
        providers.gradleProperty("base").get(),
        providers.gradleProperty("in").get(),
        providers.gradleProperty("out").get(),
    )
    withNativeKernel()
}

// Many decisions against one document, timed. The demo arm of the side-by-side.
tasks.register<JavaExec>("briefing") {
    group = "application"
    description = "Answer many questions about one document from a single shared prefix"
    classpath = sourceSets["test"].runtimeClasspath
    mainClass.set("com.integrallis.models.decisions.BriefingDemo")
    jvmArgs("--add-modules", "jdk.incubator.vector", "--enable-native-access=ALL-UNNAMED")
    args(
        providers.gradleProperty("artifact").get(),
        providers.gradleProperty("base").get(),
        providers.gradleProperty("doc").get(),
        providers.gradleProperty("questions").get(),
        providers.gradleProperty("timings").getOrElse("decisions-timing.json"),
    )
    withNativeKernel()
}

// Stages everything the demo needs to run off a bare JDK on another machine.
tasks.register<Sync>("demoDist") {
    group = "distribution"
    description = "Copy the demo's full runtime classpath into build/demo-dist"
    // Jars from the dependency graph, plus this module's own main and test classes. The filter
    // above dropped directories, and this module's main output is a directory, so without the
    // explicit includes the demo ships every dependency and none of its own code.
    from(sourceSets["test"].runtimeClasspath.filter { it.isFile })
    from(tasks.named("classes").map { sourceSets["main"].output })
    from(tasks.named("testClasses").map { sourceSets["test"].output })
    from(nativeResources)
    dependsOn(":backend-native:prepareNativePlatformResources")
    into(layout.buildDirectory.dir("demo-dist"))
    duplicatesStrategy = DuplicatesStrategy.EXCLUDE
}

// What a base costs before a head exists for it. The cheap half of a base swap.
tasks.register<JavaExec>("baseSpeed") {
    group = "verification"
    description = "Measure document prefill and question-tail cost for a candidate base"
    classpath = sourceSets["test"].runtimeClasspath
    mainClass.set("com.integrallis.models.decisions.BaseSpeedProbe")
    // Some architectures allocate session state for the whole declared context window up front,
    // so a small model can need more heap than a large one. The probe is about speed, and a heap
    // limit deciding which bases are measurable would silently narrow the comparison.
    // Session state is allocated for the declared context window, so a model that declares a very
    // long one cannot open a session at all; capping it keeps the comparison about speed.
    jvmArgs(
        "--add-modules", "jdk.incubator.vector",
        "--enable-native-access=ALL-UNNAMED",
        "-Xmx12g",
        "-Dmodels.purejava.maxContextLength=4096")
    // Tuning knobs are passed only when asked for. Supplying a default here would mean every run
    // silently carried a setting, and an unset knob would be indistinguishable from a chosen one.
    providers.gradleProperty("threads").orNull?.let { jvmArgs("-Dvectors.gguf.threads=$it") }
    providers.gradleProperty("parallelThreshold").orNull?.let {
        jvmArgs("-Dvectors.gguf.parallelThreshold=$it")
    }
    providers.gradleProperty("prefillBatch").orNull?.let {
        jvmArgs("-Dmodels.purejava.prefillBatchSize=$it")
    }
    args(
        providers.gradleProperty("base").get(),
        providers.gradleProperty("doc").get(),
    )
    withNativeKernel()
}

// Harvests one hidden state per item for a fixed question over varying state (Choice / Score).
tasks.register<JavaExec>("typedHarvest") {
    group = "verification"
    description = "Harvest hidden states for a typed corpus"
    classpath = sourceSets["test"].runtimeClasspath
    mainClass.set("com.integrallis.models.decisions.TypedHarvestTool")
    jvmArgs(
        "--add-modules", "jdk.incubator.vector",
        "--enable-native-access=ALL-UNNAMED",
        "-Xmx12g",
        "-Dmodels.purejava.maxContextLength=4096",
    )
    args(
        providers.gradleProperty("base").get(),
        providers.gradleProperty("corpus").get(),
        providers.gradleProperty("out").get(),
        providers.gradleProperty("n").get(),
        providers.gradleProperty("name").get(),
        providers.gradleProperty("suffix").getOrElse(""),
    )
    withNativeKernel()
}

// Fits a typed head on a harvest and writes the artifact.
tasks.register<JavaExec>("typedRelease") {
    group = "distribution"
    description = "Fit a Choice or Score head, read the sealed split once, write the artifact"
    classpath = sourceSets["test"].runtimeClasspath
    mainClass.set("com.integrallis.models.decisions.TypedReleaseTool")
    jvmArgs("--add-modules", "jdk.incubator.vector", "--enable-native-access=ALL-UNNAMED", "-Xmx12g")
    args(
        providers.gradleProperty("harvest").get(),
        providers.gradleProperty("corpus").get(),
        providers.gradleProperty("kind").get(),
        providers.gradleProperty("labels").get(),
        providers.gradleProperty("question").get(),
        providers.gradleProperty("base").get(),
        providers.gradleProperty("baseFile").getOrElse(""),
        providers.gradleProperty("l2").getOrElse("1.0"),
        providers.gradleProperty("out").get(),
    )
}

// Prints the test runtime classpath so a diagnostic harness can be compiled and run outside Gradle.
// A diagnostic that has to be a Gradle test task cannot be rerun with a changed system property
// without a rebuild, and a diagnostic nobody reruns stops being used.
tasks.register("printTestClasspath") {
    // A Provider rather than the FileCollection itself, so the task body captures no Project state
    // and stays usable if the configuration cache is ever switched on.
    val elements = sourceSets["test"].runtimeClasspath.elements
    doLast { println(elements.get().joinToString(":") { it.asFile.absolutePath }) }
}
