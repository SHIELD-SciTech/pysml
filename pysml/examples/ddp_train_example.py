"""
DDP Training Example for LLM with Mixed VRAM Intel Arc GPUs
Supports pipeline parallelism and gradient accumulation for efficient training
across 3 Arc GPUs: 16GB, 8GB, 8GB
"""

import time
from typing import List, Dict, Optional
import json
import numpy as np

import sys, os
sys.path.append(os.getcwd())
import pysml
from pysml import Tensor
from pysml.nn.module import Module
from pysml.nn.models import TransformerConfig
import pysml.nn.functional as F
from pysml.ddp.device_manager import DeviceManager, get_available_devices, print_device_info
from pysml.ddp.pipeline_parallel import PipelineTransformer, create_pipeline_model
from pysml.ddp.data_parallel import DistributedDataParallel
from pysml.ddp.utils import (
    clip_gradients, 
    count_parameters, 
    estimate_model_memory,
    print_memory_estimate,
    calculate_optimal_batch_split
)
from pysml.ddp.strategies import select_strategy, print_strategy_comparison


class LLMTrainer:
    """Trainer for distributed LLM training"""
    
    def __init__(
        self,
        model: Module,
        devices: List[str],
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 1e-4,
        log_interval: int = 10,
        max_grad_norm: float = 1.0
    ):
        self.model = model
        self.devices = devices
        self.grad_accum_steps = gradient_accumulation_steps
        self.lr = learning_rate
        self.log_interval = log_interval
        self.max_grad_norm = max_grad_norm
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.training_history = {
            'losses': [],
            'perplexities': [],
            'steps': []
        }
        
        # Simple learning rate (in production, would use proper optimizer)
        self.optimizer_state = {'step': 0}
        
    def compute_loss(self, logits: Tensor, labels: Tensor) -> Tensor:
        """Compute cross-entropy loss with proper gradient flow"""
        batch_size, seq_len, vocab_size = logits.shape
        
        # Reshape for loss computation
        logits_flat = pysml.reshape(logits, (batch_size * seq_len, vocab_size))
        labels_flat = pysml.reshape(labels, (batch_size * seq_len,))
        
        # Convert labels to numpy for one-hot encoding
        labels_data = labels_flat.data
        if hasattr(labels_data, 'asnumpy'):
            labels_data = labels_data.asnumpy()
        labels_data = labels_data.astype(np.int32)
        
        # Create one-hot encoded labels
        num_samples = labels_data.shape[0]
        one_hot = np.zeros((num_samples, vocab_size), dtype=np.float32)
        one_hot[np.arange(num_samples), labels_data] = 1.0
        
        # Convert to tensor (no grad needed for targets)
        targets = Tensor(one_hot, requires_grad=False)
        
        # Compute softmax
        logits_max = pysml.max(logits_flat, axis=-1, keepdims=True)
        logits_shifted = pysml.subtract(logits_flat, logits_max)
        exp_logits = pysml.exp(logits_shifted)
        sum_exp = pysml.sum(exp_logits, axis=-1, keepdims=True)
        probs = pysml.divide(exp_logits, sum_exp)
        
        # Clip for numerical stability
        probs_clipped = pysml.clip(probs, 1e-10, 1.0)
        
        # Log probabilities
        log_probs = pysml.log(probs_clipped)
        
        # Cross-entropy: -sum(targets * log_probs)
        loss_per_sample = pysml.negative(pysml.sum(pysml.multiply(targets, log_probs), axis=-1))
        
        # Mean loss
        loss = pysml.mean(loss_per_sample)
        
        return loss
    
    def train_step(self, input_ids: Tensor, labels: Tensor) -> Dict[str, float]:
        """Single training step with gradient accumulation"""
        
        # Forward pass
        logits = self.model(input_ids)
        
        # Compute loss
        loss = self.compute_loss(logits, labels)
        
        # Scale loss for gradient accumulation
        scaled_loss = loss / self.grad_accum_steps
        
        # Check if loss has grad enabled
        if not hasattr(scaled_loss, '_prev') or scaled_loss._prev is None:
            print(f"WARNING: Loss tensor has no gradient graph!")
            print(f"  Loss value: {scaled_loss.data}")
            print(f"  Loss requires_grad: {scaled_loss.requires_grad}")
        
        # Backward pass
        try:
            scaled_loss.backward()
        except Exception as e:
            print(f"ERROR in backward pass: {e}")
            import traceback
            traceback.print_exc()
        
        # Check if gradients were computed
        params_with_grad = 0
        params_without_grad = 0
        for param in self.model.parameters():
            if param.grad is not None:
                grad_data = param.grad.data
                if hasattr(grad_data, 'asnumpy'):
                    grad_data = grad_data.asnumpy()
                if np.any(grad_data != 0):
                    params_with_grad += 1
                else:
                    params_without_grad += 1
            else:
                params_without_grad += 1
        
        if params_with_grad == 0 and self.global_step == 0:
            print(f"WARNING: No parameters have non-zero gradients!")
            print(f"  Total parameters: {params_with_grad + params_without_grad}")
            print(f"  Parameters with gradients: {params_with_grad}")
            print(f"  Parameters without gradients: {params_without_grad}")
        
        # Calculate metrics
        loss_value = loss.data if hasattr(loss, 'data') else loss
        if hasattr(loss_value, 'item'):
            loss_value = loss_value.item()
        elif hasattr(loss_value, '__float__'):
            loss_value = float(loss_value)
        else:
            loss_value = float(np.mean(loss_value))
        
        perplexity = np.exp(min(loss_value, 20))  # Cap to prevent overflow
        
        metrics = {
            'loss': loss_value,
            'perplexity': perplexity
        }
        
        return metrics
    
    def optimizer_step(self):
        """Update model parameters (simplified optimizer)"""
        
        # Compute gradient norm first
        from pysml.ddp.utils import compute_gradient_norm
        total_norm = compute_gradient_norm(self.model.parameters())
        
        # Gradient clipping using utility
        clip_gradients(self.model.parameters(), max_norm=self.max_grad_norm)
        
        # Simple SGD update
        for param in self.model.parameters():
            if param.grad is not None:
                param.data = param.data - self.lr * param.grad.data
        
        self.optimizer_state['step'] += 1
        
        return total_norm
    
    def zero_grad(self):
        """Zero all gradients"""
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.data = pysml.zeros_like(param.grad.data)
    
    def train_epoch(self, dataloader, epoch: int):
        """Train for one epoch"""
        self.model.train()
        epoch_metrics = {'loss': 0.0, 'perplexity': 0.0}
        num_batches = 0
        
        self.zero_grad()
        
        print(f"\nTraining Epoch {epoch}...")
        print("-" * 80)
        
        for step, batch in enumerate(dataloader):
            # Training step
            metrics = self.train_step(batch['input_ids'], batch['labels'])
            
            # Accumulate metrics
            for key, value in metrics.items():
                epoch_metrics[key] += value
            num_batches += 1
            
            # Update weights after gradient accumulation
            if (step + 1) % self.grad_accum_steps == 0:
                # Optimizer step
                grad_norm = self.optimizer_step()
                self.zero_grad()
                
                self.global_step += 1
                
                # Logging
                if self.global_step % self.log_interval == 0:
                    avg_loss = epoch_metrics['loss'] / num_batches
                    avg_ppl = epoch_metrics['perplexity'] / num_batches
                    
                    print(f"[Epoch {epoch:2d} | Step {self.global_step:4d}] "
                          f"Loss: {avg_loss:.4f} | PPL: {avg_ppl:.2f} | "
                          f"GradNorm: {grad_norm:.4f}")
                    
                    # Store in history
                    self.training_history['losses'].append(avg_loss)
                    self.training_history['perplexities'].append(avg_ppl)
                    self.training_history['steps'].append(self.global_step)
                    
                    # Reset metrics
                    epoch_metrics = {'loss': 0.0, 'perplexity': 0.0}
                    num_batches = 0
        
        return epoch_metrics
    
    def save_checkpoint(self, path: str):
        """Save training checkpoint"""
        checkpoint = {
            'global_step': self.global_step,
            'epoch': self.epoch,
            'optimizer_state': self.optimizer_state,
            'training_history': self.training_history,
        }
        
        # Save checkpoint
        with open(path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        print(f"\nCheckpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint"""
        with open(path, 'r') as f:
            checkpoint = json.load(f)
        
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
        self.optimizer_state = checkpoint['optimizer_state']
        self.training_history = checkpoint['training_history']
        
        print(f"Checkpoint loaded from {path}")


class DummyDataLoader:
    """Dummy dataloader for testing"""
    
    def __init__(self, vocab_size: int, seq_len: int, batch_size: int, num_batches: int):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_batches = num_batches
        
    def __iter__(self):
        for _ in range(self.num_batches):
            # Generate random input IDs and labels
            input_ids = np.random.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
            labels = np.random.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
            
            yield {
                'input_ids': Tensor(input_ids, requires_grad=False),
                'labels': Tensor(labels, requires_grad=False)
            }
    
    def __len__(self):
        return self.num_batches


def print_training_summary(trainer: LLMTrainer, total_time: float, num_epochs: int):
    """Print training summary with statistics"""
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"Total Steps: {trainer.global_step}")
    print(f"Total Time: {total_time:.2f}s ({total_time/60:.2f}m)")
    print(f"Time per Epoch: {total_time/num_epochs:.2f}s")
    print(f"Steps per Second: {trainer.global_step/total_time:.2f}")
    
    if trainer.training_history['losses']:
        final_loss = trainer.training_history['losses'][-1]
        final_ppl = trainer.training_history['perplexities'][-1]
        print(f"\nFinal Loss: {final_loss:.4f}")
        print(f"Final Perplexity: {final_ppl:.2f}")
    
    print("=" * 80)


def main():
    """Main training loop"""
    
    print("=" * 80)
    print("PYSML DISTRIBUTED LLM TRAINING")
    print("Mixed VRAM Multi-GPU Training Example")
    print("=" * 80)
    
    # ============================================================================
    # CONFIGURATION
    # ============================================================================
    
    # Model config - Using SMALL preset as base
    model_config = TransformerConfig.SMALL.copy()
    model_config.update({
        'vocab_size': 32000,
        'num_layers': 12,
        'd_model': 512,
        'num_heads': 8,
        'd_ff': 2048,
        'max_seq_len': 512,
        'dropout': 0.1
    })
    
    # Training config
    batch_size = 4  # Small batch size per device for 8GB cards
    gradient_accumulation_steps = 8  # Effective batch size = 32
    num_epochs = 3
    num_batches_per_epoch = 100
    learning_rate = 3e-4
    
    # Device configuration
    print("\n" + "=" * 80)
    print("DEVICE SETUP")
    print("=" * 80)
    
    # Get available devices
    print("\nScanning for available devices...")
    print_device_info()
    
    # Define device configuration for mixed VRAM
    device_configs = [
        {'device': 'xpu:0', 'memory_gb': 16},  # Arc 16GB
        {'device': 'xpu:1', 'memory_gb': 8},   # Arc 8GB
        {'device': 'xpu:2', 'memory_gb': 8},   # Arc 8GB
    ]
    
    devices = [config['device'] for config in device_configs]
    
    print(f"\nConfigured Devices:")
    for i, config in enumerate(device_configs):
        print(f"  Device {i}: {config['device']} - {config['memory_gb']}GB VRAM")
    
    # ============================================================================
    # MODEL INITIALIZATION
    # ============================================================================
    
    print("\n" + "=" * 80)
    print("MODEL INITIALIZATION")
    print("=" * 80)
    
    print(f"\nModel Configuration:")
    print(f"  Vocabulary Size: {model_config['vocab_size']:,}")
    print(f"  Model Dimension: {model_config['d_model']}")
    print(f"  Layers: {model_config['num_layers']}")
    print(f"  Attention Heads: {model_config['num_heads']}")
    print(f"  FFN Dimension: {model_config['d_ff']}")
    print(f"  Max Sequence Length: {model_config['max_seq_len']}")
    
    # Create pipeline parallel model
    print("\nCreating pipeline-parallel transformer...")
    model = PipelineTransformer(
        vocab_size=model_config['vocab_size'],
        d_model=model_config['d_model'],
        num_layers=model_config['num_layers'],
        num_heads=model_config['num_heads'],
        d_ff=model_config['d_ff'],
        max_seq_len=model_config['max_seq_len'],
        devices=devices,
        dropout=model_config['dropout']
    )
    
    # Verify parameters have requires_grad
    print("\nVerifying parameter gradients are enabled...")
    total_params = 0
    grad_enabled_params = 0
    for name, param in model.named_parameters():
        total_params += 1
        if param.requires_grad:
            grad_enabled_params += 1
    print(f"  Total parameters: {total_params}")
    print(f"  Parameters with requires_grad=True: {grad_enabled_params}")
    
    if grad_enabled_params == 0:
        print("  WARNING: No parameters have requires_grad enabled!")
        print("  Enabling gradients on all parameters...")
        for param in model.parameters():
            param.requires_grad = True
    
    # Print memory breakdown
    model.print_memory_breakdown()
    
    # Calculate total parameters
    total_params, trainable_params = count_parameters(model)
    print(f"\nTotal Model Parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Trainable Parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    
    # Estimate memory usage
    print("\nEstimating memory requirements...")
    print_memory_estimate(model)
    
    # ============================================================================
    # TRAINING SETUP
    # ============================================================================
    
    print("\n" + "=" * 80)
    print("TRAINING CONFIGURATION")
    print("=" * 80)
    
    print(f"\nBatch Configuration:")
    print(f"  Micro-batch Size: {batch_size}")
    print(f"  Gradient Accumulation Steps: {gradient_accumulation_steps}")
    print(f"  Effective Batch Size: {batch_size * gradient_accumulation_steps}")
    
    print(f"\nOptimization:")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Max Gradient Norm: 1.0")
    
    print(f"\nTraining Schedule:")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batches per Epoch: {num_batches_per_epoch}")
    print(f"  Total Steps: {num_batches_per_epoch * num_epochs // gradient_accumulation_steps}")
    
    # Initialize trainer
    trainer = LLMTrainer(
        model=model,
        devices=devices,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        log_interval=10,
        max_grad_norm=1.0
    )
    
    # Create dataloader
    print("\nCreating dataloader...")
    dataloader = DummyDataLoader(
        vocab_size=model_config['vocab_size'],
        seq_len=model_config['max_seq_len'],
        batch_size=batch_size,
        num_batches=num_batches_per_epoch
    )
    
    # ============================================================================
    # TRAINING LOOP
    # ============================================================================
    
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)
    
    start_time = time.time()
    
    try:
        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            
            # Train epoch
            trainer.train_epoch(dataloader, epoch)
            
            epoch_time = time.time() - epoch_start
            print(f"\nEpoch {epoch} completed in {epoch_time:.2f}s")
            
            # Save checkpoint
            checkpoint_path = f"checkpoint_epoch_{epoch}.json"
            trainer.epoch = epoch
            trainer.save_checkpoint(checkpoint_path)
            
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
    
    total_time = time.time() - start_time
    
    # ============================================================================
    # TRAINING SUMMARY
    # ============================================================================
    
    print_training_summary(trainer, total_time, num_epochs)
    
    print("\n" + "=" * 80)
    print("Training completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()