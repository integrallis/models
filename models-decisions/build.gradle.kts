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
