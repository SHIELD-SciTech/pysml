#!/usr/bin/env python3
# Training Script for RWKV 80M Chatbot
# Trains on conversational data for text generation

import sys
import os
import time
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pysml
from pysml import Tensor, nn, optim
import pysml.dtype as dtype

from rwkv_80m import RWKV80M, RWKVConfig


class TextDataset:
    # Character-level dataset for training
    
    def __init__(self, text: str, seq_len: int = 256, vocab_size: int = None):
        self.text = text
        self.seq_len = seq_len
        
        if vocab_size is None:
            chars = sorted(list(set(text)))
            self.vocab_size = len(chars)
            self.char_to_idx = {c: i for i, c in enumerate(chars)}
            self.idx_to_char = {i: c for i, c in enumerate(chars)}
        else:
            self.vocab_size = vocab_size
            self.char_to_idx = {chr(i): i for i in range(vocab_size)}
            self.idx_to_char = {i: chr(i) if i < 128 else '?' for i in range(vocab_size)}
        
        self.data = [self.char_to_idx.get(c, 0) for c in text]
        print(f'Dataset: {len(self.data):,} tokens, vocab: {self.vocab_size}')
    
    def __len__(self):
        return max(1, len(self.data) - self.seq_len - 1)
    
    def get_batch(self, batch_size: int, device: str = 'cpu') -> tuple:
        max_start = len(self) - 1
        if max_start < 1:
            max_start = 1
        
        indices = np.random.randint(0, max_start, size=(batch_size,))
        
        x_batch = [self.data[i:i + self.seq_len] for i in indices]
        y_batch = [self.data[i + 1:i + self.seq_len + 1] for i in indices]
        
        x = Tensor(np.array(x_batch, dtype=np.float32), dtype=dtype.fp32(), device=device)
        y = Tensor(np.array(y_batch, dtype=np.float32), dtype=dtype.fp32(), device=device)
        
        return x, y
    
    def decode(self, tokens) -> str:
        if isinstance(tokens, (list, np.ndarray)):
            return ''.join([self.idx_to_char.get(int(t), '?') for t in tokens])
        return self.idx_to_char.get(int(tokens), '?')
    
    def encode(self, text: str) -> list:
        return [self.char_to_idx.get(c, 0) for c in text]


def load_training_data() -> str:
    # Conversational training data
    training_text = '''
User: Hello, how are you?
Assistant: Hello! I'm doing well, thank you for asking. How can I help you today?

User: What is machine learning?
Assistant: Machine learning is a subset of artificial intelligence where computers learn patterns from data without being explicitly programmed.

User: Can you explain neural networks?
Assistant: Neural networks are computing systems inspired by biological brains. They consist of layers of interconnected nodes that process information.

User: What is deep learning?
Assistant: Deep learning uses neural networks with many layers to learn complex patterns from large amounts of data.

User: Tell me about transformers.
Assistant: Transformers are a neural network architecture that uses self-attention to process sequences in parallel, making them efficient for language tasks.

User: What is RWKV?
Assistant: RWKV combines the best of RNNs and Transformers with linear complexity during inference while maintaining competitive performance.

User: How does attention work?
Assistant: Attention mechanisms let models focus on relevant parts of input by computing weighted sums based on how relevant each position is.

User: What is Python?
Assistant: Python is a high-level programming language known for readability and versatility, widely used in data science and AI.

User: Can you write code?
Assistant: Yes, I can help write and explain code in various programming languages and assist with debugging.

User: What are you?
Assistant: I am an AI language model designed for helpful conversations. I can answer questions and assist with various tasks.

User: How do you learn?
Assistant: I learn through training on large amounts of text data. My parameters are adjusted to better predict and generate coherent responses.

User: What is backpropagation?
Assistant: Backpropagation calculates how much each weight contributed to the error and adjusts weights accordingly during training.

User: Explain gradient descent.
Assistant: Gradient descent finds the minimum of a function by iteratively moving in the direction of steepest descent to minimize loss.

User: What is overfitting?
Assistant: Overfitting occurs when a model learns training data too well, including noise, causing poor performance on new data.

User: What is a GPU?
Assistant: A GPU is a specialized processor for parallel computations, essential for training deep learning models efficiently.

User: Goodbye!
Assistant: Goodbye! Feel free to come back anytime you have more questions!

'''
    return training_text * 50


def train(
    model: RWKV80M,
    dataset: TextDataset,
    epochs: int = 10,
    batch_size: int = 4,
    lr: float = 3e-4,
    warmup_steps: int = 100,
    device: str = 'cuda',
    log_interval: int = 10,
):
    print(f'\n=== Training RWKV Chatbot ===')
    print(f'Device: {device}')
    print(f'Parameters: {model.num_parameters():,}')
    print(f'Vocab: {dataset.vocab_size}, Seq len: {dataset.seq_len}')
    print(f'Batch size: {batch_size}, LR: {lr}')
    print()
    
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    criterion = nn.CrossEntropyLoss()
    
    # LR schedule
    def get_lr(step, total_steps):
        if step < warmup_steps:
            return lr * step / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return lr * 0.5 * (1 + math.cos(math.pi * progress))
    
    total_steps = 0
    best_loss = float('inf')
    steps_per_epoch = max(1, len(dataset) // batch_size)
    total_training_steps = epochs * steps_per_epoch
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_tokens = 0
        epoch_start = time.time()
        
        for step in range(steps_per_epoch):
            current_lr = get_lr(total_steps, total_training_steps)
            for pg in optimizer.param_groups:
                pg['lr'] = current_lr
            
            x, y = dataset.get_batch(batch_size, device)
            logits, _ = model(x)
            
            batch_size_actual, seq_len, vocab_size = logits.shape
            loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))
            
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            max_grad_norm = 1.0
            for param in model.parameters():
                if param.grad is not None:
                    grad_data = param.grad.data if isinstance(param.grad, Tensor) else param.grad
                    backend = param._backend
                    grad_norm = float(backend.sqrt(backend.sum(backend.multiply(grad_data, grad_data))))
                    if grad_norm > max_grad_norm:
                        scale = max_grad_norm / (grad_norm + 1e-8)
                        if isinstance(param.grad, Tensor):
                            param.grad.data = backend.multiply(grad_data, scale)
                        else:
                            param.grad = backend.multiply(grad_data, scale)
            
            optimizer.step()
            
            loss_val = loss.item()
            epoch_loss += loss_val
            epoch_tokens += batch_size_actual * seq_len
            total_steps += 1
            
            if total_steps % log_interval == 0:
                elapsed = time.time() - epoch_start
                tokens_per_sec = epoch_tokens / max(elapsed, 0.001)
                avg_loss = epoch_loss / (step + 1)
                ppl = math.exp(min(avg_loss, 20))
                
                print(f'Epoch {epoch + 1}/{epochs} | Step {step + 1}/{steps_per_epoch} | '
                      f'Loss: {loss_val:.4f} | Avg: {avg_loss:.4f} | PPL: {ppl:.1f} | '
                      f'LR: {current_lr:.2e} | Tok/s: {tokens_per_sec:.0f}')
        
        avg_epoch_loss = epoch_loss / max(steps_per_epoch, 1)
        ppl = math.exp(min(avg_epoch_loss, 20))
        elapsed = time.time() - epoch_start
        
        print(f'\n--- Epoch {epoch + 1} Complete ---')
        print(f'Average Loss: {avg_epoch_loss:.4f}, Perplexity: {ppl:.2f}, Time: {elapsed:.1f}s')
        
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            print('New best loss!')
        
        print('\n--- Sample ---')
        generate_sample(model, dataset, device)
        print()
    
    return model


def generate_sample(model: RWKV80M, dataset: TextDataset, device: str):
    model.eval()
    
    prompt = "User: What is deep learning?\nAssistant:"
    tokens = dataset.encode(prompt)
    idx = Tensor(np.array([tokens], dtype=np.float32), dtype=dtype.fp32(), device=device)
    
    output = model.generate(idx, max_new_tokens=100, temperature=0.8, top_p=0.9)
    
    output_data = output.data
    backend = output._backend
    if hasattr(backend, 'asnumpy'):
        output_np = backend.asnumpy(output_data)
    else:
        output_np = np.asarray(output_data)
    
    output_tokens = output_np[0].astype(int).tolist()
    response = dataset.decode(output_tokens)
    
    if 'User:' in response:
        response = response.split('User:')[0]
    
    print(f'Prompt: "{prompt}"')
    print(f'Response: "{response.strip()}"')
    
    model.train()


def interactive_chat(model: RWKV80M, dataset: TextDataset, device: str):
    print('\n=== Interactive Chat ===')
    print('Type "quit" to exit\n')
    
    model.eval()
    
    while True:
        try:
            user_input = input('You: ').strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                print('Goodbye!')
                break
            
            if not user_input:
                continue
            
            prompt = f"User: {user_input}\nAssistant:"
            tokens = dataset.encode(prompt)
            idx = Tensor(np.array([tokens], dtype=np.float32), dtype=dtype.fp32(), device=device)
            
            output = model.generate(idx, max_new_tokens=150, temperature=0.8, top_p=0.9)
            
            output_data = output.data
            backend = output._backend
            if hasattr(backend, 'asnumpy'):
                output_np = backend.asnumpy(output_data)
            else:
                output_np = np.asarray(output_data)
            
            output_tokens = output_np[0, len(tokens):].astype(int).tolist()
            response = dataset.decode(output_tokens)
            
            if 'User:' in response:
                response = response.split('User:')[0]
            if '\n\n' in response:
                response = response.split('\n\n')[0]
            
            print(f'Assistant: {response.strip()}\n')
            
        except KeyboardInterrupt:
            print('\nGoodbye!')
            break


def main():
    print('=' * 50)
    print('RWKV Chatbot Training')
    print('=' * 50)
    
    # Detect device
    device = 'cpu'
    try:
        import cupy
        device = 'cuda'
        print('Using CUDA')
    except ImportError:
        print('CUDA not available, using CPU')
    
    # Load data
    training_text = load_training_data()
    dataset = TextDataset(training_text, seq_len=128)
    
    # Create model
    print('\nCreating model...')
    config = RWKVConfig(
        vocab_size=dataset.vocab_size,
        d_model=384,
        n_layers=8,
        ctx_len=512,
        device=device,
    )
    model = RWKV80M(config)
    print(f'Parameters: {model.num_parameters():,}')
    
    # Train
    model = train(
        model=model,
        dataset=dataset,
        epochs=20,
        batch_size=8,
        lr=3e-4,
        warmup_steps=50,
        device=device,
        log_interval=20,
    )
    
    # Chat
    interactive_chat(model, dataset, device)


if __name__ == '__main__':
    main()
