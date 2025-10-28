# pysml/examples/amp_train_classifier.py
import pysml
from pysml.nn.models import Classifier
from pysml.nn.optim import AdamW
import pysml.nn.functional as F

# Pick best device automatically
devices = pysml.engine.get_available_devices()
device = next((d for d in devices if d.startswith(("cuda","xpu"))), "cpu")
pysml.engine.set_device(device)
print("Using device:", pysml.engine.get_device())

# Toy data
B, Din, C = 128, 784, 10
def batch():
    x = pysml.randn(B, Din)
    y = pysml.randn(B, C)
    return x, y

model = Classifier(input_dim=Din, hidden=(512,256), num_classes=C)
opt = AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)

scaler = pysml.engine.get_scaler()  # may be None on CPU

for step in range(200):
    x, y = batch()
    # autocast if available
    with pysml.engine.autocast("float16"):
        pred = model(x)
        loss = F.mse_loss(pred, y)

    # scale if AMP
    loss_to_back = pysml.engine.scale_loss(loss)
    loss_to_back.backward()
    pysml.engine.optimizer_step(opt)
    model.zero_grad()

    if step % 20 == 0:
        print(f"step {step:4d} | loss {float(loss.item() if hasattr(loss,'item') else loss):.4f}")
