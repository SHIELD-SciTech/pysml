import numpy as np
import pysml
from pysml.nn.module import Transformer
from pysml.nn.optim import SGD, AdamW
from pysml.nn import functional as F
from pysml.backend.context import device

# Model Hyperparameters
VOCAB_SIZE = 10
D_MODEL = 32
N_HEADS = 4
NUM_LAYERS = 2
D_FF = 64
SEQ_LENGTH = 8

# Training Hyperparameters
EPOCHS = 50
BATCH_SIZE = 4

# Create Dummy Data
def create_dummy_data(n_samples, seq_len, vocab_size):
    src = np.random.randint(1, vocab_size, size=(n_samples, seq_len))
    tgt = src.copy()
    src_data = pysml.Tensor(src.astype(np.int32))
    tgt_data = pysml.Tensor(tgt.astype(np.int32))
    return src_data, tgt_data

print("Preparing Data")
src_data, tgt_data = create_dummy_data(BATCH_SIZE, SEQ_LENGTH, VOCAB_SIZE)

# Model, Optimizer, Loss
print("\nInitializing Model")
with device('cpu'):
    model = Transformer(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        n_heads=N_HEADS,
        d_ff=D_FF
    )
    
    # Count trainable parameters
    params = model.parameters()
    print(f"Total Parameters: {len(params)}")
    for i, p in enumerate(params[:3]):  # Show first 3
        print(f"  Param {i}: shape={p.shape}, requires_grad={p.requires_grad}, mean={np.mean(p.data):.4f}")
    
    # Try both optimizers
    print("\n" + "="*60)
    print("Testing SGD")
    print("="*60)
    
    optimizer_sgd = SGD(model.parameters(), lr=0.1)
    
    # Save initial parameter values
    initial_params = [p.data.copy() for p in model.parameters()]
    
    for epoch in range(5):
        model.train()
        optimizer_sgd.zero_grad()
        
        output_logits = model(src_data)
        output_view = output_logits.view(BATCH_SIZE * SEQ_LENGTH, VOCAB_SIZE)
        target_view = tgt_data.data.reshape(-1)
        
        loss = F.cross_entropy(output_view, target_view)
        
        # Check if loss requires grad and has context
        print(f"\nEpoch {epoch+1}:")
        print(f"  Loss: {loss.item():.4f}")
        print(f"  Loss requires_grad: {loss.requires_grad}")
        print(f"  Loss has _ctx: {loss._ctx is not None}")
        
        loss.backward()
        
        # Check gradients
        grad_count = 0
        grad_norms = []
        for p in model.parameters():
            if p.grad is not None:
                grad_count += 1
                grad_norms.append(np.linalg.norm(p.grad.data))
        
        print(f"  Params with gradients: {grad_count}/{len(params)}")
        print(f"  Max gradient norm: {max(grad_norms) if grad_norms else 0:.6f}")
        print(f"  Mean gradient norm: {np.mean(grad_norms) if grad_norms else 0:.6f}")
        
        optimizer_sgd.step()
        
        # Check if parameters actually changed
        changes = []
        for i, (p, init_p) in enumerate(zip(model.parameters(), initial_params)):
            change = np.max(np.abs(p.data - init_p))
            changes.append(change)
        
        print(f"  Max parameter change: {max(changes):.6f}")
        print(f"  Mean parameter change: {np.mean(changes):.6f}")
    
    print("\n" + "="*60)
    print("Testing AdamW")
    print("="*60)
    
    # Reinitialize model
    model = Transformer(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        n_heads=N_HEADS,
        d_ff=D_FF
    )
    
    optimizer_adamw = AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
    
    # Save initial parameter values
    initial_params = [p.data.copy() for p in model.parameters()]
    
    for epoch in range(5):
        model.train()
        optimizer_adamw.zero_grad()
        
        output_logits = model(src_data)
        output_view = output_logits.view(BATCH_SIZE * SEQ_LENGTH, VOCAB_SIZE)
        target_view = tgt_data.data.reshape(-1)
        
        loss = F.cross_entropy(output_view, target_view)
        
        print(f"\nEpoch {epoch+1}:")
        print(f"  Loss: {loss.item():.4f}")
        
        loss.backward()
        
        # Check gradients
        grad_count = 0
        grad_norms = []
        zero_grad_count = 0
        for p in model.parameters():
            if p.grad is not None:
                grad_count += 1
                norm = np.linalg.norm(p.grad.data)
                grad_norms.append(norm)
                if np.all(p.grad.data == 0):
                    zero_grad_count += 1
        
        print(f"  Params with gradients: {grad_count}/{len(params)}")
        print(f"  Params with zero gradients: {zero_grad_count}")
        print(f"  Max gradient norm: {max(grad_norms) if grad_norms else 0:.6f}")
        print(f"  Mean gradient norm: {np.mean(grad_norms) if grad_norms else 0:.6f}")
        
        # Debug: manually check what AdamW will do
        print(f"  AdamW timestep: {optimizer_adamw.t + 1}")
        
        optimizer_adamw.step()
        
        # Check if parameters actually changed
        changes = []
        for i, (p, init_p) in enumerate(zip(model.parameters(), initial_params)):
            change = np.max(np.abs(p.data - init_p))
            changes.append(change)
            if i < 3:  # Show first 3 params
                print(f"    Param {i} change: {change:.8f}")
        
        print(f"  Max parameter change: {max(changes):.6f}")
        print(f"  Mean parameter change: {np.mean(changes):.6f}")