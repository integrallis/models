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
