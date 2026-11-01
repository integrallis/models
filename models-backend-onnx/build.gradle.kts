// models-backend-onnx — in-process, pure-Java execution of ONNX graphs.
//
// This module reads ONNX protobuf files with its own wire-format decoder and executes the graph
// eagerly with Java kernels. It must never depend on or delegate to ONNX Runtime: the point of the
// module is that the artifact the catalog pins runs on the JVM alone.

description = "In-process, pure-Java ONNX graph execution"

dependencies {
    // No internal or external dependencies: the ONNX protobuf schema is decoded by hand.
}

// The kernels use the Java Vector API, which is still an incubating module; the root build adds it
// to every test JVM and to javadoc, and this is the compiler's side of the same requirement.
tasks.withType<JavaCompile>().configureEach {
    options.compilerArgs.addAll(listOf("--add-modules", "jdk.incubator.vector"))
}
