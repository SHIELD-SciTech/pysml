"""
Memory-Optimized Training Loop for PySML
Demonstrates best practices for training large models efficiently
"""

import pysml
from pysml import Tensor
from pysml.nn import Module
import gc


def get_memory_stats(device='cuda'):
    """Get current memory usage statistics"""
    if device.startswith('cuda'):
        import torch.cuda as cuda
        return {
            'allocated_gb': cuda.memory_allocated() / 1e9,
            'reserved_gb': cuda.memory_reserved() / 1e9,
            'max_allocated_gb': cuda.max_memory_allocated() / 1e9
        }
    elif device.startswith('xpu'):
        # For Intel XPU, use dpctl if available
        try:
            import dpctl
            # Add XPU-specific memory tracking
            return {'status': 'xpu_tracking_available'}
        except:
            return {'status': 'no_tracking'}
    return {'status': 'cpu'}


def clear_memory_cache(device='cuda'):
    """Clear memory cache for the specified device"""
    if device.startswith('cuda'):
        import torch.cuda as cuda
        cuda.empty_cache()
        cuda.synchronize()
    elif device.startswith('xpu'):
        try:
            import dpctl
            # XPU-specific cache clearing if available
            pass
        except:
            pass
    
    # Python garbage collection
    gc.collect()


class MemoryEfficientTrainer:
    """
    Trainer class with aggressive memory management
    """
    
    def __init__(self, model, optimizer, criterion, device='cuda'):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        
        # Memory tracking
        self.log_memory = True
        self.clear_cache_every = 10  # Clear cache every N batches
    
    def train_epoch(self, dataloader, epoch):
        """Train for one epoch with memory optimization"""
        self.model.train()
        
        total_loss = 0
        num_batches = len(dataloader)
        
        for batch_idx, (data, target) in enumerate(dataloader):
            # Move data to device
            data = pysml.Tensor(data, device=self.device, requires_grad=False)
            target = pysml.Tensor(target, device=self.device, requires_grad=False)
            
            # CRITICAL: Zero gradients FIRST - with set_to_none for better memory
            self.optimizer.zero_grad(set_to_none=True)
            
            # Forward pass
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # Optimizer step
            self.optimizer.step()
            
            # Track loss
            loss_val = loss.item()
            total_loss += loss_val
            
            # CRITICAL: Explicitly delete intermediate tensors
            del loss, output, data, target
            
            # Periodic memory management
            if batch_idx % self.clear_cache_every == 0:
                clear_memory_cache(self.device)
                
                if self.log_memory and batch_idx % 50 == 0:
                    mem_stats = get_memory_stats(self.device)
                    print(f"Epoch {epoch}, Batch {batch_idx}/{num_batches}, "
                          f"Loss: {loss_val:.4f}, "
                          f"Memory: {mem_stats.get('allocated_gb', 0):.2f}GB")
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def evaluate(self, dataloader):
        """Evaluate model with minimal memory footprint"""
        self.model.eval()
        
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            # Move data to device (no gradients needed)
            data = pysml.Tensor(data, device=self.device, requires_grad=False)
            target = pysml.Tensor(target, device=self.device, requires_grad=False)
            
            # Forward pass only (no backward)
            output = self.model(data)
            loss = self.criterion(output, target)
            
            total_loss += loss.item()
            
            # Calculate accuracy (if classification)
            predictions = output.data.argmax(axis=-1)
            correct += (predictions == target.data).sum()
            total += len(target.data)
            
            # CRITICAL: Delete tensors immediately
            del loss, output, data, target
            
            # More frequent cache clearing in eval (no backward pass)
            if batch_idx % 5 == 0:
                clear_memory_cache(self.device)
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total if total > 0 else 0
        
        return avg_loss, accuracy


# ===== EXAMPLE USAGE =====

def train_model_example():
    """
    Example of memory-efficient training
    """
    
    # Set device
    device = 'cuda:0' if pysml.cuda.backend.AVAILABLE else 'cpu'
    pysml.set_device(device)
    
    # Create model (example)
    from pysml.nn import Linear, Sequential
    from pysml.nn.activations import ReLU
    
    model = Sequential(
        Linear(784, 512),
        ReLU(),
        Linear(512, 256),
        ReLU(),
        Linear(256, 10)
    )
    
    # Create optimizer
    from pysml.nn.optim import Adam
    optimizer = Adam(model.parameters(), lr=0.001)
    
    # Create loss function
    criterion = pysml.cross_entropy_loss
    
    # Create trainer
    trainer = MemoryEfficientTrainer(model, optimizer, criterion, device)
    
    # Training loop
    num_epochs = 10
    
    print("Starting training with memory optimization...")
    print(f"Device: {device}")
    print(f"Initial memory: {get_memory_stats(device)}")
    
    for epoch in range(num_epochs):
        # Train
        train_loss = trainer.train_epoch(train_dataloader, epoch)
        
        # Evaluate
        val_loss, val_acc = trainer.evaluate(val_dataloader)
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"  Memory: {get_memory_stats(device)}")
        
        # Full memory cleanup after each epoch
        clear_memory_cache(device)
    
    print("\nTraining complete!")
    print(f"Final memory: {get_memory_stats(device)}")


# ===== GRADIENT ACCUMULATION FOR LARGE MODELS =====

class GradientAccumulationTrainer(MemoryEfficientTrainer):
    """
    Trainer with gradient accumulation for very large models
    Allows effective batch size larger than memory permits
    """
    
    def __init__(self, model, optimizer, criterion, device='cuda', 
                 accumulation_steps=4):
        super().__init__(model, optimizer, criterion, device)
        self.accumulation_steps = accumulation_steps
    
    def train_epoch(self, dataloader, epoch):
        """Train with gradient accumulation"""
        self.model.train()
        
        total_loss = 0
        num_batches = len(dataloader)
        
        for batch_idx, (data, target) in enumerate(dataloader):
            # Move data to device
            data = pysml.Tensor(data, device=self.device, requires_grad=False)
            target = pysml.Tensor(target, device=self.device, requires_grad=False)
            
            # Forward pass
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Scale loss by accumulation steps
            scaled_loss = loss / self.accumulation_steps
            
            # Backward pass
            scaled_loss.backward()
            
            # Track loss
            loss_val = loss.item()
            total_loss += loss_val
            
            # CRITICAL: Delete tensors
            del loss, scaled_loss, output, data, target
            
            # Update weights every accumulation_steps
            if (batch_idx + 1) % self.accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                
                # Clear cache after optimizer step
                clear_memory_cache(self.device)
                
                if self.log_memory and batch_idx % 50 == 0:
                    mem_stats = get_memory_stats(self.device)
                    print(f"Epoch {epoch}, Batch {batch_idx}/{num_batches}, "
                          f"Loss: {loss_val:.4f}, "
                          f"Memory: {mem_stats.get('allocated_gb', 0):.2f}GB")
        
        # Final optimizer step for remaining gradients
        if (batch_idx + 1) % self.accumulation_steps != 0:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
        
        avg_loss = total_loss / num_batches
        return avg_loss


# ===== MEMORY PROFILING UTILITY =====

def profile_model_memory(model, input_shape, device='cuda'):
    """
    Profile memory usage of a model
    
    Args:
        model: The model to profile
        input_shape: Shape of input tensor (batch_size, ...)
        device: Device to run on
    
    Returns:
        Dictionary with memory statistics
    """
    import pysml
    
    # Reset peak memory stats
    if device.startswith('cuda'):
        import torch.cuda as cuda
        cuda.reset_peak_memory_stats()
    
    # Get initial memory
    initial_mem = get_memory_stats(device)
    
    # Create dummy input
    dummy_input = pysml.randn(*input_shape, device=device, requires_grad=False)
    
    # Forward pass
    model.train()
    output = model(dummy_input)
    
    forward_mem = get_memory_stats(device)
    
    # Create dummy loss and backward
    loss = output.sum()
    loss.backward()
    
    backward_mem = get_memory_stats(device)
    
    # Cleanup
    del loss, output, dummy_input
    model.zero_grad(set_to_none=True)
    clear_memory_cache(device)
    
    # Calculate memory usage
    num_params = sum(p.size for p in model.parameters())
    param_memory_mb = num_params * 4 / 1e6  # float32 = 4 bytes
    
    return {
        'num_parameters': num_params,
        'param_memory_mb': param_memory_mb,
        'forward_memory_gb': forward_mem.get('allocated_gb', 0),
        'backward_memory_gb': backward_mem.get('allocated_gb', 0),
        'peak_memory_gb': backward_mem.get('max_allocated_gb', 0)
    }


# ===== EXAMPLE: Check if model fits in memory =====

def check_model_fits(model, batch_size, input_shape, device='cuda', max_memory_gb=24):
    """
    Check if model and batch will fit in available memory
    """
    print(f"Profiling model memory usage...")
    
    full_input_shape = (batch_size,) + input_shape
    stats = profile_model_memory(model, full_input_shape, device)
    
    print(f"\nModel Statistics:")
    print(f"  Parameters: {stats['num_parameters']:,}")
    print(f"  Parameter Memory: {stats['param_memory_mb']:.2f} MB")
    print(f"  Forward Pass Memory: {stats['forward_memory_gb']:.2f} GB")
    print(f"  Backward Pass Memory: {stats['backward_memory_gb']:.2f} GB")
    print(f"  Peak Memory: {stats['peak_memory_gb']:.2f} GB")
    
    if stats['peak_memory_gb'] < max_memory_gb:
        print(f"\n✓ Model fits in {max_memory_gb}GB memory!")
        recommended_batch = int(batch_size * max_memory_gb / stats['peak_memory_gb'])
        print(f"  Recommended max batch size: {recommended_batch}")
        return True
    else:
        print(f"\n✗ Model exceeds {max_memory_gb}GB memory!")
        recommended_batch = int(batch_size * max_memory_gb / stats['peak_memory_gb'])
        print(f"  Reduce batch size to approximately: {recommended_batch}")
        return False