// models-decisions — System One decisions: typed, bounded, calibrated judgements over model state.
//
// This module owns the decision contracts and the calibration mathematics only. It depends on the
// API tier so a head can read a hidden state, and deliberately pulls no backend: the evaluator that
// forks a shared physical prefix is wired by whichever backend implements the capability.

dependencies {
    api(project(":models-api"))

    testImplementation("org.junit.jupiter:junit-jupiter:5.11.4")
    testImplementation("org.assertj:assertj-core:3.27.2")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher:1.11.4")
}
