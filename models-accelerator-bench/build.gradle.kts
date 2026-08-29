// models-accelerator-bench — isolated accelerator experiments (never published)

plugins {
    java
    application
    id("com.github.spotbugs")
}

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
    jvmArgs("--add-modules", "jdk.incubator.vector", "--enable-native-access=ALL-UNNAMED")
}

tasks.withType<JavaExec> {
    jvmArgs("--add-modules", "jdk.incubator.vector", "--enable-native-access=ALL-UNNAMED")
}

tasks.named("spotbugsTest") {
    enabled = false
}

application {
    mainClass = "com.integrallis.models.accelerator.Q8ProjectionExperiment"
}

dependencies {
    implementation(project(":backend-java"))
    implementation(project(":backend-tornado"))
    implementation(project(":models-runtime"))
    implementation("com.integrallis:vectors-core:${providers.gradleProperty("vectorsVersion").get()}")
    implementation("io.github.beehive-lab:tornado-api:5.2.0-jdk25")
    implementation("io.github.beehive-lab:tornado-runtime:5.2.0-jdk25")
    testImplementation("org.junit.jupiter:junit-jupiter:5.11.4")
    testImplementation("org.assertj:assertj-core:3.27.2")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher:1.11.4")
}
