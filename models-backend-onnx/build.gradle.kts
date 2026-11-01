// models-backend-onnx — in-process, pure-Java execution of ONNX graphs.
//
// This module reads ONNX protobuf files with its own wire-format decoder and executes the graph
// eagerly with Java kernels. It must never depend on or delegate to ONNX Runtime: the point of the
// module is that the artifact the catalog pins runs on the JVM alone.

dependencies {
    // No internal or external dependencies: the ONNX protobuf schema is decoded by hand.
}
