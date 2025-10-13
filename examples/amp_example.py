import sys, os
sys.path.insert(0, os.getcwd().split("examples")[0])
import numpy as np
import pysml
from pysml.nn.module import Transformer, SimpleCNN
from pysml.nn.optim import AdamW
from pysml.nn import functional as F
from pysml.amp import autocast, GradScaler, AMPContext, clip_grad_norm_
from pysml.backend.context import device
import time


#Example 1: Basic AMP Usage
print("="*70)
print("Example 1: Basic Mixed Precision Training")
print("="*70)

# Create model and data
model = SimpleCNN(num_classes=10)
optimizer = AdamW(model.parameters(), lr=0.001)
scaler = GradScaler()

# Dummy data
batch_size = 16
images = np.random.randn(batch_size, 1, 28, 28).astype(np.float32)
labels = np.random.randint(0, 10, batch_size)

print("Training with AMP...")
model.train()
optimizer.zero_grad()

# Forward pass with autocast
with autocast():
    x = pysml.Tensor(images)
    output = model(x)
    loss = F.cross_entropy(output, labels)
    print(f"Loss (in autocast): {loss.item():.4f}")
    print(f"Loss dtype: {loss.dtype}")

# Backward pass with gradient scaling
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

print(f"Training step completed with scale: {scaler.get_scale()}")
print()


# Example 2: Training Loop with AMP
print("="*70)
print("Example 2: Full Training Loop with Mixed Precision")
print("="*70)

# Model hyperparameters
VOCAB_SIZE = 50
D_MODEL = 64
N_HEADS = 4
NUM_LAYERS = 2
D_FF = 128
SEQ_LENGTH = 16
BATCH_SIZE = 8
EPOCHS = 10

# Create dummy data function
def create_dummy_data(batch_size, seq_len, vocab_size):
    src = np.random.randint(1, vocab_size, size=(batch_size, seq_len))
    tgt = src.copy()
    src_data = pysml.Tensor(src.astype(np.int32))
    tgt_data = pysml.Tensor(tgt.astype(np.int32))
    return src_data, tgt_data

# Initialize model and optimizer
print("Initializing model...")
model = Transformer(
    vocab_size=VOCAB_SIZE,
    d_model=D_MODEL,
    num_layers=NUM_LAYERS,
    n_heads=N_HEADS,
    d_ff=D_FF
)
optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scaler = GradScaler()

print(f"Total Parameters: {F.sum_params(model):,}")
print(f"Initial scale: {scaler.get_scale()}")
print()

# Training loop
print("Training with AMP...")
print("-"*70)
start_time = time.time()

for epoch in range(EPOCHS):
    model.train()
    
    # Generate batch
    src_data, tgt_data = create_dummy_data(BATCH_SIZE, SEQ_LENGTH, VOCAB_SIZE)
    
    optimizer.zero_grad()
    
    # Forward pass in mixed precision
    with autocast():
        output_logits = model(src_data)
        output_view = output_logits.view(BATCH_SIZE * SEQ_LENGTH, VOCAB_SIZE)
        target_view = tgt_data.data.reshape(-1)
        loss = F.cross_entropy(output_view, target_view)
    
    # Backward pass with gradient scaling
    scaler.scale(loss).backward()
    
    # Optional: Gradient clipping (recommended with AMP)
    scaler.unscale_(optimizer)
    clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    # Optimizer step with scaled gradients
    scaler.step(optimizer)
    scaler.update()
    
    if (epoch + 1) % 2 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}, "
              f"Scale: {scaler.get_scale():.0f}")

training_time = time.time() - start_time
print("-"*70)
print(f"Training completed in {training_time:.2f}s")
print()


# Example 3: AMP Context Manager
print("="*70)
print("Example 3: Using AMPContext (Simplified API)")
print("="*70)

model = SimpleCNN(num_classes=10)
optimizer = AdamW(model.parameters(), lr=0.001)

# Create AMP context
amp = AMPContext(enabled=True)

print("Training with AMPContext...")
for epoch in range(5):
    images = np.random.randn(16, 1, 28, 28).astype(np.float32)
    labels = np.random.randint(0, 10, 16)
    
    optimizer.zero_grad()
    
    # Forward pass with autocast
    with amp.autocast():
        x = pysml.Tensor(images)
        output = model(x)
        loss = F.cross_entropy(output, labels)
    
    # Backward and step
    amp.scale(loss).backward()
    amp.step(optimizer)
    amp.update()
    
    if (epoch + 1) % 2 == 0:
        print(f"Epoch [{epoch+1}/5], Loss: {loss.item():.4f}")

print("AMPContext training completed")
print()


# Example 4: Comparison - FP32 vs AMP
print("="*70)
print("Example 4: Performance Comparison (FP32 vs FP16)")
print("="*70)

model = Transformer(
    vocab_size=30,
    d_model=64,
    num_layers=2,
    n_heads=4,
    d_ff=128
)

# Training function
def train_epochs(use_amp=False, num_epochs=10):
    optimizer = AdamW(model.parameters(), lr=0.001)
    scaler = GradScaler() if use_amp else None
    
    start = time.time()
    for epoch in range(num_epochs):
        src, tgt = create_dummy_data(8, 16, 30)
        optimizer.zero_grad()
        
        if use_amp:
            with autocast():
                output = model(src)
                output_view = output.view(8 * 16, 30)
                loss = F.cross_entropy(output_view, tgt.data.reshape(-1))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(src)
            output_view = output.view(8 * 16, 30)
            loss = F.cross_entropy(output_view, tgt.data.reshape(-1))
            loss.backward()
            optimizer.step()
    
    return time.time() - start

# Run comparisons
print("Training with FP32 (baseline)...")
time_fp32 = train_epochs(use_amp=False, num_epochs=10)
print(f"FP32 Time: {time_fp32:.3f}s")

print("\nTraining with AMP (FP16)...")
time_amp = train_epochs(use_amp=True, num_epochs=10)
print(f"AMP Time: {time_amp:.3f}s")

speedup = time_fp32 / time_amp
print(f"\nSpeedup: {speedup:.2f}x")
print(f"Note: Speedup is more significant on GPU with tensor cores")
print()


# Example 5: Checkpointing with AMP
print("="*70)
print("Example 5: Saving and Loading AMP State")
print("="*70)

model = SimpleCNN(num_classes=10)
optimizer = AdamW(model.parameters(), lr=0.001)
scaler = GradScaler(init_scale=2**16)

# Train for a few steps
for step in range(3):
    images = np.random.randn(8, 1, 28, 28).astype(np.float32)
    labels = np.random.randint(0, 10, 8)
    
    optimizer.zero_grad()
    with autocast():
        x = pysml.Tensor(images)
        output = model(x)
        loss = F.cross_entropy(output, labels)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

# Save checkpoint with AMP state
checkpoint = {
    'epoch': 0,
    'model_state_dict': model.parameters(),
    'optimizer_state_dict': optimizer,
    'scaler_state_dict': scaler.state_dict()
}

print(f"Scaler state before save:")
print(f"  Scale: {scaler.get_scale()}")
print(f"  Growth tracker: {scaler._growth_tracker}")

# Simulate loading
new_scaler = GradScaler()
new_scaler.load_state_dict(checkpoint['scaler_state_dict'])

print(f"\nScaler state after load:")
print(f"  Scale: {new_scaler.get_scale()}")
print(f"  Growth tracker: {new_scaler._growth_tracker}")
print("Checkpoint save/load successful")
print()


# Example 6: Gradient Overflow Handling
print("="*70)
print("Example 6: Handling Gradient Overflow")
print("="*70)

model = SimpleCNN(num_classes=10)
optimizer = AdamW(model.parameters(), lr=0.001)
scaler = GradScaler(init_scale=2**20)  # Very high initial scale

print(f"Initial scale: {scaler.get_scale()}")

# Simulate training that causes overflow
for step in range(5):
    images = np.random.randn(8, 1, 28, 28).astype(np.float32) * 100  # Large values
    labels = np.random.randint(0, 10, 8)
    
    optimizer.zero_grad()
    with autocast():
        x = pysml.Tensor(images)
        output = model(x)
        loss = F.cross_entropy(output, labels)
    
    scaler.scale(loss).backward()
    
    # Check for overflow
    found_inf = scaler._check_inf_gradients(optimizer)
    if found_inf:
        print(f"Step {step}: Gradient overflow detected!")
    
    scaler.step(optimizer)
    scaler.update()
    
    if step > 0 and step % 2 == 0:
        print(f"Step {step}: Scale adjusted to {scaler.get_scale():.0f}")

print("Gradient overflow handling demonstrated")
print()


# Summary
print("="*70)
print("Summary: Mixed Precision Training Benefits")
print("="*70)
print("Faster training (especially on GPUs with tensor cores)")
print("Reduced memory usage (allows larger batch sizes)")
print("Automatic loss scaling prevents gradient underflow")
print("Dynamic scaling adapts to training stability")
print("Minimal code changes required")
print("\nBest Practices:")
print("  1. Use autocast() for forward pass")
print("  2. Use GradScaler for backward pass")
print("  3. Apply gradient clipping after unscaling")
print("  4. Save scaler state in checkpoints")
print("  5. Monitor loss scale during training")
print("="*70)