// models-bench — JMH benchmarks (not published as a library)

plugins {
    java
    id("me.champeau.jmh") version "0.7.2"
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
}

dependencies {
    implementation(project(":models-runtime"))
    implementation(project(":models-backend-purejava"))
}
