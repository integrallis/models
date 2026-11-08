CPU float32 QuickGELU reference generated on Linux x86_64 with PyTorch 2.9.0+cpu, eight threads. Transformers 4.57.1 QuickGELUActivation uses `input * torch.sigmoid(1.702 * input)`.

Regenerate the little-endian fixtures with:

```python
import torch
torch.set_num_threads(8)
x = torch.linspace(-12, 12, 4097, dtype=torch.float32)
x.numpy().astype("<f4").tofile("input.f32")
(x * torch.sigmoid(1.702 * x)).numpy().astype("<f4").tofile("torch.f32")
```

The range exercises sigmoid tails, zero, and the curved region. The RMS bound detects the previous direct-division graph while allowing platform math rounding. These synthetic fixtures contain no model weights.
