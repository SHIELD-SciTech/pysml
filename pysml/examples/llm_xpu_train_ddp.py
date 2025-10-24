
import sys, os
sys.path.append(os.getcwd())
import pysml
import pysml.nn as nn
import pysml.nn.functional as F
from pysml.ddp.pipeline_parallel import PipelineTransformer
from pysml.ddp import get_available_devices
from pysml import xpu

# === Detect devices and memory ===
devices = xpu.get_available_devices()
if len(devices) < 4:
    raise RuntimeError(f"Need 4 XPUs, found {len(devices)} -> {devices}")
devices = devices[:4]

print("Detected Intel XPUs:")
mem_info = []
for i, dev in enumerate(devices):
    props = xpu.get_device_properties(i)
    total_gb = props.get("global_mem_size", 0) / 1e9
    mem_info.append(total_gb)
    print(f"  {dev:6s} | {props.get('name', 'Unknown'):35s} | Total Memory: {total_gb:6.2f} GB")

# === Distribute layers proportionally by memory ===
total_mem = sum(mem_info)
num_layers = 36
layer_distribution = [max(1, int(num_layers * (m / total_mem))) for m in mem_info]

# Fix rounding (ensure sum == num_layers)
while sum(layer_distribution) < num_layers:
    layer_distribution[layer_distribution.index(max(layer_distribution))] += 1
while sum(layer_distribution) > num_layers:
    layer_distribution[layer_distribution.index(max(layer_distribution))] -= 1

print("\nLayer Distribution:")
for dev, layers in zip(devices, layer_distribution):
    print(f"  {dev}: {layers} layers")

# === Create pipeline-parallel transformer ===
model = PipelineTransformer(
    vocab_size=65536,               # 65536 - for multilingual
    d_model=2048,                   # 2048
    num_layers=num_layers,          # Layers  4 - 128 (recommend: 24, 32, 48, 64)
    num_heads=32,                   # d_model / 64 => 32 
    d_ff=8192,                      # 4*d_model => 8192
    max_seq_len=6144,               # Max length, recommend k * 2^n, ex: 1.5 * 2^12 = 6144
    devices=devices,
    dropout=0.1
)

# Manually assign blocks to match distribution
model.block_devices = []
current_idx = 0
for dev, layers in zip(devices, layer_distribution):
    for _ in range(layers):
        model.block_devices.append(dev)
        current_idx += 1

# === Show memory split ===
model.print_memory_breakdown()

# === Dummy data ===
batch_size, seq_len = 4, 128
x = pysml.randint(0, model.vocab_size, (batch_size, seq_len))
y = pysml.randint(0, model.vocab_size, (batch_size, seq_len))

# === Forward + loss ===
logits = model(x)
batch_size, seq_len, vocab_size = logits.shape
logits_flat = pysml.reshape(logits, (batch_size * seq_len, vocab_size))
y_flat = pysml.reshape(y, (batch_size * seq_len,))
loss = F.cross_entropy(logits_flat, y_flat)
print(f"Initial loss: {loss.item():.4f}")

# === Optimizer and training loop ===
optimizer = nn.AdamW(model.parameters(), lr=1e-4)
for epoch in range(3):
    optimizer.zero_grad()
    logits = model(x)
    logits_flat = pysml.reshape(logits, (batch_size * seq_len, vocab_size))
    y_flat = pysml.reshape(y, (batch_size * seq_len,))
    loss = F.cross_entropy(logits_flat, y_flat)
    loss.backward()
    optimizer.step()
    print(f"[Epoch {epoch+1}/3] Loss: {loss.item():.4f}")

print("✅ Non-uniform pipeline parallel training complete!")
