import sys
sys.path.insert(0, "/root")
from clm import Engine
from local_embedder import LocalQwenEmbedder

engine = Engine(embedder=LocalQwenEmbedder("Qwen/Qwen3-8B"))
print("heads:", sorted(engine.heads))
result = engine.rank(
    "What causes tides on Earth?",
    ["The Moon's gravitational pull.", "Photosynthesis in plants.", "Because the Earth is round."],
)
for row in result:
    print("  ", row)
print()
print("README expects: Moon at rank 1, prob ~0.997")
