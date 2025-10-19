"""
PySML Preset Model Training Examples
Complete training examples using preset architectures
"""

import sys, os
sys.path.append(os.getcwd())
import pysml
import pysml.nn as nn
import pysml.nn.functional as F
import numpy as np

print("=" * 80)
print("PySML Preset Model Training Examples")
print("=" * 80)

# Set device
pysml.set_device('cpu')


# ===== Example 1: MNIST-like Classifier =====
print("\n" + "=" * 80)
print("Example 1: Training Classifier on Synthetic MNIST-like Data")
print("=" * 80)

# Create classifier from preset
model = nn.Classifier.from_preset('SIMPLE_MLP')
print(f"Model:\n{model}")
print(f"Total parameters: {nn.count_parameters(model):,}")

# Generate synthetic data (simulating flattened 28x28 MNIST images)
np.random.seed(42)
n_samples = 1000
n_classes = 10

# Create synthetic features and labels
X_train = np.random.randn(n_samples, 784).astype(np.float32) * 0.5
y_train = np.random.randint(0, n_classes, n_samples).astype(np.int32)

# Convert to one-hot
y_train_onehot = np.zeros((n_samples, n_classes), dtype=np.float32)
for i in range(n_samples):
    y_train_onehot[i, y_train[i]] = 1.0

# Convert to tensors
X_tensor = pysml.Tensor(X_train, requires_grad=False)
y_tensor = pysml.Tensor(y_train_onehot, requires_grad=False)

# Setup optimizer
optimizer = nn.Adam(model.parameters(), lr=0.001)

# Training loop
print("\nTraining classifier...")
model.train()
batch_size = 32
n_epochs = 20

for epoch in range(n_epochs):
    total_loss = 0.0
    n_batches = 0
    
    # Mini-batch training
    for i in range(0, n_samples, batch_size):
        end_idx = min(i + batch_size, n_samples)
        
        # Get batch
        X_batch = pysml.Tensor(X_train[i:end_idx], requires_grad=False)
        y_batch = pysml.Tensor(y_train_onehot[i:end_idx], requires_grad=False)
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = pysml.mse_loss(predictions, y_batch)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    avg_loss = total_loss / n_batches
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1:3d}/{n_epochs}, Loss: {avg_loss:.6f}")

# Evaluation
print("\nEvaluating on training data...")
model.eval()
predictions = model(X_tensor)
pred_classes = np.argmax(predictions.data, axis=1)
accuracy = np.mean(pred_classes == y_train)
print(f"Training Accuracy: {accuracy * 100:.2f}%")


# ===== Example 2: Autoencoder for Dimensionality Reduction =====
print("\n" + "=" * 80)
print("Example 2: Training Autoencoder for Dimensionality Reduction")
print("=" * 80)

# Create autoencoder from preset
autoencoder = nn.AutoEncoder.from_preset('SMALL')
print(f"Autoencoder:\n{autoencoder}")
print(f"Total parameters: {nn.count_parameters(autoencoder):,}")

# Generate synthetic high-dimensional data
np.random.seed(123)
n_samples = 500
X_ae = np.random.randn(n_samples, 784).astype(np.float32) * 0.3 + 0.5
X_ae = np.clip(X_ae, 0, 1)  # Normalize to [0, 1]

X_ae_tensor = pysml.Tensor(X_ae, requires_grad=False)

# Setup optimizer
optimizer_ae = nn.Adam(autoencoder.parameters(), lr=0.001)

# Training loop
print("\nTraining autoencoder...")
autoencoder.train()
n_epochs = 30
batch_size = 50

for epoch in range(n_epochs):
    total_loss = 0.0
    n_batches = 0
    
    for i in range(0, n_samples, batch_size):
        end_idx = min(i + batch_size, n_samples)
        
        # Get batch
        X_batch = pysml.Tensor(X_ae[i:end_idx], requires_grad=False)
        
        # Forward pass - autoencoder reconstructs input
        optimizer_ae.zero_grad()
        reconstruction = autoencoder(X_batch)
        loss = pysml.mse_loss(reconstruction, X_batch)
        
        # Backward pass
        loss.backward()
        optimizer_ae.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    avg_loss = total_loss / n_batches
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1:3d}/{n_epochs}, Reconstruction Loss: {avg_loss:.6f}")

# Test reconstruction
print("\nTesting reconstruction...")
autoencoder.eval()
test_sample = pysml.Tensor(X_ae[:5], requires_grad=False)
reconstructed = autoencoder(test_sample)
reconstruction_error = pysml.mse_loss(reconstructed, test_sample)
print(f"Reconstruction error on test samples: {reconstruction_error.item():.6f}")


# ===== Example 3: Simple VAE Training =====
print("\n" + "=" * 80)
print("Example 3: Training VAE for Generative Modeling")
print("=" * 80)

# Create VAE from preset
vae = nn.VAE.from_preset('MNIST')
print(f"VAE Latent Dimension: {vae.latent_dim}")
print(f"Total parameters: {nn.count_parameters(vae):,}")

# Generate synthetic data
np.random.seed(456)
n_samples = 400
X_vae = np.random.randn(n_samples, 784).astype(np.float32) * 0.3 + 0.5
X_vae = np.clip(X_vae, 0, 1)

# Setup optimizer
optimizer_vae = nn.Adam(vae.parameters(), lr=0.001)

# VAE loss function
def vae_loss(recon_x, x, mu, logvar):
    """VAE loss = reconstruction loss + KL divergence"""
    # Reconstruction loss (MSE)
    recon_loss = pysml.mse_loss(recon_x, x, reduction='sum')
    
    # KL divergence: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    # KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    mu_sq = pysml.multiply(mu, mu)
    exp_logvar = pysml.exp(logvar)
    
    kl_div = -0.5 * pysml.sum(
        1 + logvar - mu_sq - exp_logvar
    )
    
    return recon_loss + kl_div

# Training loop
print("\nTraining VAE...")
vae.train()
n_epochs = 25
batch_size = 50

for epoch in range(n_epochs):
    total_loss = 0.0
    n_batches = 0
    
    for i in range(0, n_samples, batch_size):
        end_idx = min(i + batch_size, n_samples)
        
        # Get batch
        X_batch = pysml.Tensor(X_vae[i:end_idx], requires_grad=False)
        
        # Forward pass
        optimizer_vae.zero_grad()
        reconstruction, mu, logvar = vae(X_batch)
        loss = vae_loss(reconstruction, X_batch, mu, logvar)
        
        # Backward pass
        loss.backward()
        optimizer_vae.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    avg_loss = total_loss / n_batches
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1:3d}/{n_epochs}, Loss: {avg_loss:.2f}")

# Generate new samples
print("\nGenerating new samples from VAE...")
vae.eval()
generated = vae.generate(num_samples=5)
print(f"Generated samples shape: {generated.shape}")
print(f"Generated samples (first 10 values): {generated.data[0, :10]}")


# ===== Example 4: Multi-Layer Network Training =====
print("\n" + "=" * 80)
print("Example 4: Training Deep Classifier with Learning Rate Scheduling")
print("=" * 80)

# Create deep classifier
deep_model = nn.Classifier.from_preset('DEEP')
print(f"Deep Model:\n{deep_model}")
print(f"Total parameters: {nn.count_parameters(deep_model):,}")

# Generate more complex synthetic data
np.random.seed(789)
n_samples = 800
X_deep = np.random.randn(n_samples, 784).astype(np.float32)
# Create non-linear patterns
X_deep = np.tanh(X_deep * 0.5)
y_deep = np.random.randint(0, 10, n_samples).astype(np.int32)

# One-hot encode
y_deep_onehot = np.zeros((n_samples, 10), dtype=np.float32)
for i in range(n_samples):
    y_deep_onehot[i, y_deep[i]] = 1.0

# Setup optimizer with learning rate scheduler
optimizer_deep = nn.Adam(deep_model.parameters(), lr=0.01)
scheduler = nn.StepLR(optimizer_deep, step_size=10, gamma=0.5)

# Training loop with scheduler
print("\nTraining deep classifier with LR scheduling...")
deep_model.train()
n_epochs = 30
batch_size = 64

for epoch in range(n_epochs):
    total_loss = 0.0
    correct = 0
    n_batches = 0
    
    for i in range(0, n_samples, batch_size):
        end_idx = min(i + batch_size, n_samples)
        
        X_batch = pysml.Tensor(X_deep[i:end_idx], requires_grad=False)
        y_batch = pysml.Tensor(y_deep_onehot[i:end_idx], requires_grad=False)
        
        # Forward pass
        optimizer_deep.zero_grad()
        predictions = deep_model(X_batch)
        loss = pysml.mse_loss(predictions, y_batch)
        
        # Backward pass
        loss.backward()
        optimizer_deep.step()
        
        total_loss += loss.item()
        
        # Calculate accuracy
        pred_classes = np.argmax(predictions.data, axis=1)
        correct += np.sum(pred_classes == y_deep[i:end_idx])
        n_batches += 1
    
    # Step the scheduler
    scheduler.step()
    
    avg_loss = total_loss / n_batches
    accuracy = correct / n_samples
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1:3d}/{n_epochs}, Loss: {avg_loss:.6f}, "
              f"Accuracy: {accuracy*100:.2f}%, LR: {optimizer_deep.lr:.6f}")


# ===== Example 5: Comparison of Different Optimizers =====
print("\n" + "=" * 80)
print("Example 5: Comparing Different Optimizers")
print("=" * 80)

# Create three identical models
model_sgd = nn.Classifier.from_preset('SIMPLE_MLP')
model_adam = nn.Classifier.from_preset('SIMPLE_MLP')
model_rmsprop = nn.Classifier.from_preset('SIMPLE_MLP')

# Generate simple XOR-like problem
np.random.seed(999)
X_compare = np.random.randn(200, 784).astype(np.float32) * 0.3
y_compare = np.random.randint(0, 2, 200).astype(np.int32)
y_compare_onehot = np.zeros((200, 10), dtype=np.float32)
for i in range(200):
    y_compare_onehot[i, y_compare[i]] = 1.0

X_compare_tensor = pysml.Tensor(X_compare, requires_grad=False)
y_compare_tensor = pysml.Tensor(y_compare_onehot, requires_grad=False)

# Setup different optimizers
opt_sgd = nn.SGD(model_sgd.parameters(), lr=0.01)
opt_adam = nn.Adam(model_adam.parameters(), lr=0.001)
opt_rmsprop = nn.RMSprop(model_rmsprop.parameters(), lr=0.001)

optimizers = [
    ('SGD', model_sgd, opt_sgd),
    ('Adam', model_adam, opt_adam),
    ('RMSprop', model_rmsprop, opt_rmsprop)
]

print("\nTraining with different optimizers...")
n_epochs = 20

for opt_name, model, optimizer in optimizers:
    print(f"\n{opt_name}:")
    model.train()
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        predictions = model(X_compare_tensor)
        loss = pysml.mse_loss(predictions, y_compare_tensor)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:2d}/{n_epochs}, Loss: {loss.item():.6f}")


# ===== Summary =====
print("\n" + "=" * 80)
print("Training Examples Summary")
print("=" * 80)
print("\n✓ Example 1: Trained classifier with mini-batch SGD")
print("✓ Example 2: Trained autoencoder for reconstruction")
print("✓ Example 3: Trained VAE with KL divergence")
print("✓ Example 4: Used learning rate scheduling")
print("✓ Example 5: Compared different optimizers")
print("\nAll preset models support:")
print("  • Full gradient computation via backpropagation")
print("  • Mini-batch training")
print("  • Multiple optimizer choices (SGD, Adam, RMSprop, etc.)")
print("  • Learning rate scheduling")
print("  • Training/evaluation modes")
print("=" * 80)