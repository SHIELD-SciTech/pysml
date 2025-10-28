# pysml/examples/ddp_data_parallel.py
import pysml
from pysml.nn.models import Classifier
from pysml.nn.optim import AdamW
import pysml.nn.functional as F
from pysml.ddp.data_parallel import DataParallel
from pysml.ddp.device_manager import list_devices

# Use all available accelerators of the same type, else CPU
devs = [d for d in list_devices() if d.startswith(("cuda","xpu"))]
if not devs: devs = ["cpu"]
print("Devices:", devs)

base = Classifier(input_dim=784, hidden=(512,256), num_classes=10)
model = DataParallel(base, devices=devs, bucket_bytes_cap=32*1024*1024)

opt = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

for step in range(50):
    x = pysml.randn(256, 784)
    y = pysml.randn(256, 10)
    pred = model(x)
    loss = F.mse_loss(pred, y)
    loss.backward()
    opt.step()
    model.zero_grad()
    if step % 10 == 0:
        print("step", step, "loss", float(loss.item() if hasattr(loss,"item") else loss))
