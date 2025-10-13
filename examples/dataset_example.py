import sys, os
sys.path.insert(0,os.getcwd().split("examples")[0])
import numpy as np
import pysml
from pysml.data import (
    TensorDataset, DataLoader, train_test_split,
    Compose, Normalize, ToTensor
)
from pysml.nn.module import Module
from pysml.nn.conv import Conv2d, MaxPool2d, Flatten, Dropout
from pysml.nn.linear import Linear
from pysml.nn.optim import AdamW
from pysml.nn import functional as F
import pysml.store as store


class SimpleCNN(Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = Conv2d(1, 32, 3, padding=1)
        self.conv2 = Conv2d(32, 64, 3, padding=1)
        self.pool = MaxPool2d(2, 2)
        self.flatten = Flatten()
        self.fc1 = Linear(64 * 7 * 7, 128)
        self.dropout = Dropout(0.5)
        self.fc2 = Linear(128, num_classes)
    
    def forward(self, x):
        from pysml.nn.autograd import ReLU
        
        x = self.conv1(x)
        x = ReLU.apply(ReLU, x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = ReLU.apply(ReLU, x)
        x = self.pool(x)
        
        x = self.flatten(x)
        x = self.fc1(x)
        x = ReLU.apply(ReLU, x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


def create_synthetic_mnist_data(num_samples=1000):
    images = np.random.randn(num_samples, 1, 28, 28).astype(np.float32)
    labels = np.random.randint(0, 10, size=num_samples)
    return images, labels


def train_epoch(model, train_loader, optimizer, epoch, device='cpu'):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        # Convert to PySML tensors
        data = pysml.Tensor(data)
        
        # Move to device if needed
        if device != 'cpu':
            data = data.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(data)
        
        # Compute loss
        batch_size = data.shape[0]
        loss = F.cross_entropy(output.view(batch_size, -1), target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        predictions = np.argmax(output.data, axis=1)
        correct += np.sum(predictions == target)
        total += len(target)
        
        # Print progress every 10 batches
        if (batch_idx + 1) % 10 == 0:
            print(f'  Batch [{batch_idx+1}/{len(train_loader)}], '
                  f'Loss: {loss.item():.4f}, '
                  f'Acc: {100*correct/total:.2f}%')
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def evaluate(model, test_loader, device='cpu'):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    for data, target in test_loader:
        # Convert to PySML tensors
        data = pysml.Tensor(data)
        
        # Move to device if needed
        if device != 'cpu':
            data = data.to(device)
        
        # Forward pass (no gradients needed)
        output = model(data)
        
        # Compute loss
        batch_size = data.shape[0]
        loss = F.cross_entropy(output.view(batch_size, -1), target)
        
        # Statistics
        total_loss += loss.item()
        predictions = np.argmax(output.data, axis=1)
        correct += np.sum(predictions == target)
        total += len(target)
    
    avg_loss = total_loss / len(test_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def main():
    print("="*70)
    print("PySML Complete Training Example with DataLoader")
    print("="*70)
    
    # Hyperparameters
    NUM_EPOCHS = 10
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 0.0001
    NUM_CLASSES = 10
    DEVICE = 'cpu'  # Change to 'cuda:0' or 'xpu:0' if available
    
    # Create synthetic data
    print("\n[1/6] Creating dataset...")
    images, labels = create_synthetic_mnist_data(num_samples=1000)
    print(f"  Data shape: {images.shape}")
    print(f"  Labels shape: {labels.shape}")
    
    # Create dataset
    dataset = TensorDataset(images, labels)
    
    # Split into train and test
    print("\n[2/6] Splitting dataset...")
    train_dataset, test_dataset = train_test_split(
        dataset,
        test_size=0.2,
        random_state=42
    )
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    
    # Create data loaders
    print("\n[3/6] Creating data loaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # Create model
    print("\n[4/6] Creating model...")
    model = SimpleCNN(num_classes=NUM_CLASSES)
    
    # Count parameters
    model_info = store.get_model_size(model)
    print(f"  Total parameters: {model_info['total_params']:,}")
    print(f"  Model size: {model_info['memory_mb']:.2f} MB")
    
    # Create optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    
    # Training loop
    print("\n[5/6] Training model...")
    print("-"*70)
    
    best_test_acc = 0
    train_losses = []
    test_losses = []
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}]")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, epoch, DEVICE
        )
        train_losses.append(train_loss)
        
        # Evaluate
        test_loss, test_acc = evaluate(model, test_loader, DEVICE)
        test_losses.append(test_loss)
        
        print(f"\n  Results:")
        print(f"    Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"    Test Loss:  {test_loss:.4f}, Test Acc:  {test_acc:.2f}%")
        
        # Save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            store.save_checkpoint(
                model, optimizer,
                'best_model.pysml',
                epoch=epoch,
                loss=test_loss,
                metadata={'test_accuracy': test_acc}
            )
            print(f"    ✓ New best model saved! (Test Acc: {test_acc:.2f}%)")
    
    print("-"*70)
    
    # Final evaluation
    print("\n[6/6] Final evaluation...")
    final_loss, final_acc = evaluate(model, test_loader, DEVICE)
    print(f"  Final Test Loss: {final_loss:.4f}")
    print(f"  Final Test Accuracy: {final_acc:.2f}%")
    print(f"  Best Test Accuracy: {best_test_acc:.2f}%")
    
    # Save final model
    store.save_state_dict(model, 'final_model_weights.pysml')
    print("\n  Model saved to 'final_model_weights.pysml'")
    
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    
    return model, train_losses, test_losses


def inference_example():
    print("\n" + "="*70)
    print("Inference Example")
    print("="*70)
    
    # Load model
    print("\n[1/3] Loading saved model...")
    model = SimpleCNN(num_classes=10)
    store.load_state_dict(model, 'final_model_weights.pysml')
    model.eval()
    print("  ✓ Model loaded successfully")
    
    # Create test sample
    print("\n[2/3] Running inference...")
    test_image = np.random.randn(1, 1, 28, 28).astype(np.float32)
    test_tensor = pysml.Tensor(test_image)
    
    # Run inference
    output = model(test_tensor)
    
    # Get prediction
    prediction = np.argmax(output.data)
    probabilities = output.data[0]
    
    print(f"  Predicted class: {prediction}")
    print(f"  Confidence: {probabilities[prediction]:.4f}")
    print(f"\n  Top 3 predictions:")
    top_3_indices = np.argsort(probabilities)[-3:][::-1]
    for idx in top_3_indices:
        print(f"    Class {idx}: {probabilities[idx]:.4f}")
    
    print("\n[3/3] Batch inference...")
    batch_images = np.random.randn(5, 1, 28, 28).astype(np.float32)
    batch_tensor = pysml.Tensor(batch_images)
    
    batch_output = model(batch_tensor)
    batch_predictions = np.argmax(batch_output.data, axis=1)
    
    print(f"  Batch predictions: {batch_predictions}")
    print(f"  Batch shape: {batch_output.shape}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    # Run training
    model, train_losses, test_losses = main()
    
    # Run inference example
    inference_example()
    
    # Print summary statistics
    print("\n" + "="*70)
    print("Training Summary")
    print("="*70)
    print(f"  Initial train loss: {train_losses[0]:.4f}")
    print(f"  Final train loss:   {train_losses[-1]:.4f}")
    print(f"  Loss reduction:     {train_losses[0] - train_losses[-1]:.4f}")
    print(f"\n  Initial test loss:  {test_losses[0]:.4f}")
    print(f"  Final test loss:    {test_losses[-1]:.4f}")
    print(f"  Loss reduction:     {test_losses[0] - test_losses[-1]:.4f}")
    print("\n" + "="*70)