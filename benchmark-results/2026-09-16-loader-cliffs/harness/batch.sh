set -x
$SCRATCH/exp/run.sh q3-java-forceScalar java $HOME/.jvllm/models/Qwen3-0.6B-Q4_0.gguf 4 -Dvectors.forceScalar=true
$SCRATCH/exp/run.sh q3-java-maxBits128 java $HOME/.jvllm/models/Qwen3-0.6B-Q4_0.gguf 4 -Dvectors.maxBits=128
$SCRATCH/exp/run.sh q3-java-shortPairwise java $HOME/.jvllm/models/Qwen3-0.6B-Q4_0.gguf 4 -Dmodels.purejava.q4Kernel=short-pairwise
$SCRATCH/exp/run.sh q3-java-shortPairwise-maxBits128 java $HOME/.jvllm/models/Qwen3-0.6B-Q4_0.gguf 4 -Dvectors.maxBits=128 -Dmodels.purejava.q4Kernel=short-pairwise
$SCRATCH/exp/run.sh q3-java-executor-dedicated java $HOME/.jvllm/models/Qwen3-0.6B-Q4_0.gguf 4 -Dvectors.gguf.executor=dedicated
$SCRATCH/exp/run.sh q3-java-parallel-false java $HOME/.jvllm/models/Qwen3-0.6B-Q4_0.gguf 4 -Dvectors.gguf.parallel=false
$SCRATCH/exp/run.sh granite41-native-baseline native $HOME/.jvllm/models/granite-4.1-3b-Q4_K_M.gguf 4
$SCRATCH/exp/run.sh granite41-native-groupedAttention-false native $HOME/.jvllm/models/granite-4.1-3b-Q4_K_M.gguf 4 -Dmodels.native.groupedAttention=false
$SCRATCH/exp/run.sh granite41-java-baseline java $HOME/.jvllm/models/granite-4.1-3b-Q4_K_M.gguf 4
$SCRATCH/exp/run.sh qwen35-native-gdn-true native $HOME/.jvllm/models/Qwen3.5-0.8B-Q4_K_M.gguf 4 -Dmodels.native.gatedDeltaNet=true
$SCRATCH/exp/run.sh qwen35-native-gdn-false native $HOME/.jvllm/models/Qwen3.5-0.8B-Q4_K_M.gguf 4 -Dmodels.native.gatedDeltaNet=false
$SCRATCH/exp/run.sh qwen35-java-baseline java $HOME/.jvllm/models/Qwen3.5-0.8B-Q4_K_M.gguf 4
