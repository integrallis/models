import java.nio.charset.StandardCharsets
import java.security.MessageDigest
import java.util.HexFormat
import java.util.Properties

// backend-cuda - Models-owned GPU kernels, written in Rust, compiled to PTX, driven from Java FFM.
//
// The packaging mirrors backend-native: a build-time Rust compile, an artifact carried inside the
// jar under META-INF, a .properties descriptor holding a SHA-256 the Java loader recomputes, and
// an ABI field that must match the Java binding. Deliberately the same mechanism; a second one
// would be a second thing to get wrong.
//
// It is simpler in one respect. PTX is *device* code, so one artifact serves every host platform.
// backend-native ships six per-platform shared libraries and checks the host architecture at load;
// this ships one .ptx and checks the device's compute capability instead.
//
// CI can run every task here on a machine with no GPU and no CUDA toolkit. Only the Rust nightly
// toolchain pinned in the crate's rust-toolchain.toml is required.

apply(plugin = "maven-publish")

// Bump when a kernel name, parameter list or launch contract changes. Mirrors
// CudaKernelAbi.VERSION in Java and PTX_ABI_VERSION in the Rust crate.
val cudaAbi = 2

// The PTX virtual architecture. 8.0 is the campaign's floor: the pre-registration's candidate
// devices (A40, L40S) are 8.6 and 8.9, and PTX is forward compatible, so one sm_80 module serves
// every qualifying device. Recorded in the artifact descriptor and in every report JSON.
val ptxTarget = "sm_80"

dependencies {
    api(project(":models-api"))
    implementation(project(":backend-java"))
    implementation("com.integrallis:vectors-core:${providers.gradleProperty("vectorsVersion").get()}")
    testImplementation(project(":models-runtime"))
}

val rustProjectDirectory = layout.projectDirectory.dir("src/main/rust/models-cuda-kernels")
val rustManifest = rustProjectDirectory.file("Cargo.toml")
val rustSources = fileTree(rustProjectDirectory) {
    include("Cargo.toml", "Cargo.lock", "rust-toolchain.toml", "src/**/*.rs", "tests/**/*.rs")
}
val rustTargetDirectory = layout.buildDirectory.dir("rust-target")
val ptxFile = rustTargetDirectory.map { directory ->
    directory.file("nvptx64-nvidia-cuda/release/models_cuda_kernels.ptx")
}
val ptxResourceRoot = layout.buildDirectory.dir("generated/cuda-resources")
val ptxResourceDirectory = ptxResourceRoot.map { it.dir("META-INF/models/cuda") }

// The exact toolchain, pinned so a toolchain regression is distinguishable from our own change.
// Kept in step with src/main/rust/models-cuda-kernels/rust-toolchain.toml; verifyRustToolchain
// fails the build if the two drift.
val rustToolchain = "nightly-2026-09-17"

// -Zbuild-std is required, not a preference: without it `core` is prebuilt for the target's
// default CPU and rustc refuses the -C target-cpu=sm_80 mismatch. See backend-cuda/UPSTREAM.md,
// entry CU-001, for the reproduction and the alternative we rejected.
val ptxBuildArguments = listOf(
    "cargo",
    "+$rustToolchain",
    "build",
    "--release",
    "--target",
    "nvptx64-nvidia-cuda",
    "-Zbuild-std=core,compiler_builtins",
    "-Zbuild-std-features=compiler-builtins-mem",
    "--manifest-path",
)

val verifyRustToolchain by tasks.registering {
    group = "verification"
    description = "Fail if the pinned Rust toolchain and the crate's rust-toolchain.toml disagree"
    val toolchainFile = rustProjectDirectory.file("rust-toolchain.toml").asFile
    inputs.file(toolchainFile)
    inputs.property("expected", rustToolchain)
    doLast {
        val declared = toolchainFile.readLines()
            .firstOrNull { it.trimStart().startsWith("channel") }
            ?.substringAfter('=')
            ?.trim()
            ?.trim('"')
            ?: error("rust-toolchain.toml declares no channel")
        require(declared == rustToolchain) {
            "build.gradle.kts pins $rustToolchain but rust-toolchain.toml declares $declared"
        }
    }
}

val cargoTestHost by tasks.registering(Exec::class) {
    group = "verification"
    description = "Run the off-device kernel parity tests (no GPU, no CUDA toolkit required)"
    inputs.files(rustSources)
    environment("CARGO_TARGET_DIR", rustTargetDirectory.get().asFile.absolutePath)
    commandLine("cargo", "test", "--release", "--manifest-path", rustManifest.asFile.absolutePath)
}

val cargoClippy by tasks.registering(Exec::class) {
    group = "verification"
    description = "Lint the Models CUDA kernel crate"
    inputs.files(rustSources)
    environment("CARGO_TARGET_DIR", rustTargetDirectory.get().asFile.absolutePath)
    commandLine(
        "cargo",
        "clippy",
        "--release",
        "--all-targets",
        "--manifest-path",
        rustManifest.asFile.absolutePath,
        "--",
        "-D",
        "warnings",
    )
}

val compilePtx by tasks.registering(Exec::class) {
    group = "build"
    description = "Compile the Rust kernels to PTX for $ptxTarget"
    dependsOn(verifyRustToolchain)
    inputs.files(rustSources)
    inputs.property("target", ptxTarget)
    inputs.property("toolchain", rustToolchain)
    outputs.file(ptxFile)
    environment("CARGO_TARGET_DIR", rustTargetDirectory.get().asFile.absolutePath)
    environment("RUSTFLAGS", "-C target-cpu=$ptxTarget")
    commandLine(ptxBuildArguments + rustManifest.asFile.absolutePath)
}

val preparePtxResources by tasks.registering {
    group = "build"
    description = "Stage the PTX module and its integrity descriptor for packaging"
    dependsOn(compilePtx)
    inputs.file(ptxFile)
    inputs.property("abi", cudaAbi)
    inputs.property("target", ptxTarget)
    inputs.property("toolchain", rustToolchain)
    outputs.dir(ptxResourceRoot)
    doLast {
        val source = ptxFile.get().asFile
        require(source.isFile) { "cargo did not produce ${source.absolutePath}" }
        val outputRoot = ptxResourceRoot.get().asFile
        delete(outputRoot)
        val outputDirectory = ptxResourceDirectory.get().asFile
        outputDirectory.mkdirs()
        val moduleFile = outputDirectory.resolve("models-cuda-kernels.ptx")
        source.copyTo(moduleFile, overwrite = true)

        val ptx = moduleFile.readText(StandardCharsets.UTF_8)
        // The names the Java binding resolves. Checked here so a rename in Rust fails the build
        // instead of producing a module whose kernels are never found at runtime.
        val kernels = listOf(
            "models_q4k_decode_projection",
            "models_q6k_decode_projection",
            "models_gqa_decode_attention",
        )
        kernels.forEach { kernel ->
            require(ptx.contains(".visible .entry $kernel(")) {
                "PTX module does not export $kernel"
            }
        }
        require(ptx.contains(".target $ptxTarget")) {
            "PTX module targets something other than $ptxTarget"
        }
        // The shared-memory declaration must survive: link_section = \".shared\" silently lands
        // in global memory, so the kernels emit it through global_asm! instead. If this vanishes
        // the kernels still compile and still run, but race across CUDA blocks.
        require(ptx.contains(".shared .align 4 .b8 models_cuda_scratch")) {
            "PTX module has no .shared scratch; the kernels would race across blocks"
        }
        require(ptx.contains("st.shared.b32") && ptx.contains("ld.shared.b32")) {
            "PTX module never touches shared memory; the scratch declaration is inert"
        }
        // fma.rn.f32 is what makes the device bit-exact with the CPU K-quant path. A plain
        // mul+add would still produce plausible numbers and fail G1 on a real model.
        require(ptx.contains("fma.rn.f32")) {
            "PTX module contains no fused multiply-add; bit-exactness with the CPU path is lost"
        }

        val digest = HexFormat.of()
            .formatHex(MessageDigest.getInstance("SHA-256").digest(moduleFile.readBytes()))
        outputDirectory.resolve("cuda.properties").writeText(
            """
            abi=$cudaAbi
            module=models-cuda-kernels.ptx
            sha256=$digest
            target=$ptxTarget
            toolchain=$rustToolchain
            kernels=${kernels.joinToString(",")}
            """.trimIndent() + "\n",
            StandardCharsets.UTF_8,
        )
    }
}

sourceSets {
    named("main") {
        resources.srcDir(ptxResourceRoot)
    }
}

tasks.named("processResources") {
    dependsOn(preparePtxResources)
}

// withSourcesJar() (applied at the root) archives sourceSets["main"].allSource, which now
// includes ptxResourceRoot (registered as a resources srcDir above). Gradle only sees that as a
// task *input*, not a dependency, so a clean parallel build can race sourcesJar against
// preparePtxResources and either package a stale/missing PTX module or fail outright depending on
// scheduling. assemble (and anyone else who runs sourcesJar without first running check, which
// depends on preparePtxResources transitively via verifyPtxArtifact) can hit this.
tasks.named("sourcesJar") {
    dependsOn(preparePtxResources)
}

val verifyPtxArtifact by tasks.registering {
    group = "verification"
    description = "Recompute the packaged PTX digest and check it against its descriptor"
    dependsOn(preparePtxResources)
    inputs.dir(ptxResourceRoot)
    doLast {
        val directory = ptxResourceDirectory.get().asFile
        val metadata = Properties()
        directory.resolve("cuda.properties").inputStream().use { stream -> metadata.load(stream) }
        require(metadata.getProperty("abi") == cudaAbi.toString()) {
            "packaged PTX does not declare ABI $cudaAbi"
        }
        val moduleFile = directory.resolve(metadata.getProperty("module"))
        val actual = HexFormat.of()
            .formatHex(MessageDigest.getInstance("SHA-256").digest(moduleFile.readBytes()))
        require(metadata.getProperty("sha256") == actual) {
            "packaged PTX SHA-256 does not match its descriptor"
        }
    }
}

tasks.withType<Test>().configureEach {
    dependsOn(preparePtxResources)
    // The CUDA driver binding is FFM; the tests exercise the fallback path on hosts with no GPU.
    jvmArgs("--enable-native-access=ALL-UNNAMED")
}

tasks.named("check") {
    dependsOn(cargoTestHost, cargoClippy, verifyPtxArtifact)
}
