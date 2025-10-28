# pysml/examples/ddp_pipeline_parallel.py
import pysml
import pysml.nn as nn
import pysml.nn.functional as F
from pysml.nn.optim import AdamW
from pysml.ddp.pipeline_manager import Pipeline, Stage
from pysml.ddp.device_manager import list_devices

devs = [d for d in list_devices() if d.startswith(("cuda","xpu"))]
if len(devs) < 2: devs = devs[:1] or ["cpu"]
print("Pipeline devices:", devs)

# Two simple stages
stage1 = nn.Sequential(nn.Linear(784, 1024), nn.GELU(), nn.Linear(1024, 512))
stage2 = nn.Sequential(nn.GELU(), nn.Linear(512, 10))

pipe = Pipeline([Stage(stage1, device=devs[0]),
                 Stage(stage2, device=devs[-1])])

opt = AdamW(pipe.parameters(), lr=1e-3, weight_decay=0.01)

for step in range(60):
    x = pysml.randn(128, 784)
    y = pysml.randn(128, 10)
    out = pipe(x)
    loss = F.mse_loss(out, y)
    loss.backward()
    opt.step()
    pipe.zero_grad()
    if step % 10 == 0:
        print("step", step, "loss", float(loss.item() if hasattr(loss,'item') else loss))

### Another example:

from pysml.ddp.pipeline_manager import Pipeline, Stage
from pysml.nn.models import Sequential, Linear, ReLU
from pysml.nn.optim import AdamW
from pysml.engine import set_device

# Define model parts
part1 = Sequential(Linear(1024, 2048), ReLU())
part2 = Sequential(Linear(2048, 1024), ReLU())

# Assign to devices
stage1 = Stage(part1, "cuda:0")
stage2 = Stage(part2, "cuda:1")

# Create pipeline
pipe = Pipeline([stage1, stage2], microbatch_size=8)
pipe.summary()

# Forward + backward
x = pipe.stages[0].backend.random.randn(32, 1024)
out = pipe.forward(x)
loss = pipe.stages[-1].backend.mean(out)
pipe.backward(loss)

# Optimizers
opt1 = AdamW(stage1.parameters(), lr=1e-3)
opt2 = AdamW(stage2.parameters(), lr=1e-3)
pipe.step([opt1, opt2])

