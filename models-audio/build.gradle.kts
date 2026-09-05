// models-audio — Java-native speech and audio model runtimes

dependencies {
    api(project(":models-api"))

    implementation(project(":models-runtime"))
    implementation(project(":backend-java"))
    implementation(project(":backend-native"))
}

val sopranoNativeLibraryName =
    when {
        System.getProperty("os.name").startsWith("Mac", ignoreCase = true) ->
            "libjmodels_kernels.dylib"
        System.getProperty("os.name").startsWith("Windows", ignoreCase = true) ->
            "jmodels_kernels.dll"
        else -> "libjmodels_kernels.so"
    }
val sopranoNativeLibrary =
    project(":backend-native")
        .layout.buildDirectory.file("rust-target/release/$sopranoNativeLibraryName")

tasks.register<Test>("sopranoIntegrationTest") {
    description = "Run end-to-end speech synthesis with the configured standalone Soprano GGUF"
    group = "verification"
    testClassesDirs = sourceSets["test"].output.classesDirs
    classpath = sourceSets["test"].runtimeClasspath
    useJUnitPlatform {
        includeTags("integration")
    }
    filter {
        includeTestsMatching(
            "com.integrallis.models.audio.SopranoTextToSpeechModelIntegrationTest",
        )
    }
    outputs.upToDateWhen { false }
    maxParallelForks = 1
    maxHeapSize = "2g"
    dependsOn(project(":backend-native").tasks.named("cargoBuildRelease"))
    jvmArgs("--enable-native-access=ALL-UNNAMED")
    systemProperty("models.soprano.test.nativeLibrary", sopranoNativeLibrary.get().asFile.absolutePath)
    systemProperty("models.native.quantizedDecode", "true")
    providers.environmentVariable("MODELS_SOPRANO_JFR").orNull?.let { recording ->
        jvmArgs("-XX:StartFlightRecording=filename=$recording,settings=profile,dumponexit=true")
    }
}
