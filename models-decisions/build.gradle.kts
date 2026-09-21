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
    into(layout.buildDirectory.dir("demo-dist"))
    duplicatesStrategy = DuplicatesStrategy.EXCLUDE
}
