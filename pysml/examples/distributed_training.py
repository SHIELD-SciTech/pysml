#!/usr/bin/env python3
# Distributed Training Examples for PySML
# Shows pipeline parallel, data parallel, and hybrid configurations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pysml
from pysml import Tensor, nn, optim
import pysml.dtype as dtype
from pysml import ddp


def example_pipeline():
    print('=== Pipeline Parallel ===')
    
    # Create model stages - each would go on different GPU
    stage0 = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 128))
    stage1 = nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 128))
    stage2 = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 32))
    
    # Wrap in pipeline (using CPU for demo)
    model = ddp.PipelineParallel(
        modules=[stage0, stage1, stage2],
        devices=['cpu', 'cpu', 'cpu'],
        chunks=4,
        schedule=ddp.PipelineSchedule.GPIPE,
    )
    
    print(f'Stages: {model.num_stages}')
    print(f'Parameters: {model.num_parameters():,}')
    
    x = Tensor(np.random.randn(16, 64).astype(np.float32), dtype=dtype.fp32())
    y = model(x)
    
    print(f'Input: {x.shape}')
    print(f'Output: {y.shape}')
    print()


def example_config():
    print('=== Parallel Configuration ===')
    
    # 2-way data parallel x 4-way pipeline parallel = 8 GPUs
    config = ddp.ParallelConfig(
        data_parallel_size=2,
        pipeline_parallel_size=4,
        num_micro_batches=8,
        pipeline_schedule=ddp.PipelineSchedule.ONE_F_ONE_B,
    )
    
    print(f'World size: {config.world_size}')
    print(f'Data parallel groups:')
    for rank in range(config.world_size):
        print(f'  Rank {rank}: {config.get_data_parallel_group(rank)}')
    
    print(f'Pipeline groups:')
    for rank in range(config.world_size):
        print(f'  Rank {rank}: {config.get_pipeline_parallel_group(rank)}')
    print()


def example_sampler():
    print('=== Distributed Sampler ===')
    
    dataset_size = 1000
    num_gpus = 4
    
    for rank in range(num_gpus):
        sampler = ddp.DistributedSampler(
            dataset_size=dataset_size,
            num_replicas=num_gpus,
            rank=rank,
            shuffle=True,
        )
        indices = list(sampler)
        print(f'Rank {rank}: {len(indices)} samples, first 5: {indices[:5]}')
    print()


def example_data_parallel():
    print('=== Data Parallel Pattern ===')
    
    # This shows the pattern - actual DDP needs multi-process setup
    
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 32)
            self.fc2 = nn.Linear(32, 10)
        
        def forward(self, x):
            x = pysml.relu(self.fc1(x))
            return self.fc2(x)
    
    model = SimpleModel()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    print(f'Parameters: {model.num_parameters()}')
    print('Training pattern:')
    print('  1. Each GPU gets full model copy')
    print('  2. Each GPU processes different batch')
    print('  3. Gradients are all-reduced')
    print('  4. All GPUs update with same gradients')
    
    for step in range(5):
        x = Tensor(np.random.randn(8, 10).astype(np.float32), dtype=dtype.fp32())
        y = Tensor(np.random.randn(8, 10).astype(np.float32), dtype=dtype.fp32())
        
        pred = model(x)
        loss = criterion(pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        # In DDP: gradients would be all-reduced here
        optimizer.step()
        
        print(f'  Step {step + 1}: Loss = {loss.item():.4f}')
    print()


def example_launch():
    print('=== Launch Configuration ===')
    
    # Configure for 4 GPUs (mixed CUDA + XPU)
    config = ddp.LaunchConfig(
        world_size=4,
        master_addr='localhost',
        master_port=29500,
        devices=['cuda:0', 'cuda:1', 'xpu:0', 'xpu:1'],
    )
    
    print(f'World size: {config.world_size}')
    print(f'Master: {config.master_addr}:{config.master_port}')
    print(f'Devices: {config.devices}')
    print()
    print('To launch:')
    print('  ddp.launch(train_fn, config, args=(model_cls,))')
    print()


def main():
    print('PySML - Distributed Training')
    print('=' * 45)
    print()
    
    example_pipeline()
    example_config()
    example_sampler()
    example_data_parallel()
    example_launch()
    
    print('All distributed examples complete!')
    print()
    print('For multi-GPU training:')
    print('  python -m pysml.ddp.launch --nproc_per_node=4 train_script.py')


if __name__ == '__main__':
    main()
