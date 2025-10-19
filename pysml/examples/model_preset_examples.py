"""
PySML Model Preset Examples
Complete guide to using all preset models with training examples
"""

import sys, os
sys.path.append(os.getcwd())
import pysml
import pysml.nn as nn
import pysml.nn.functional as F
import numpy as np

print("=" * 80)
print("PySML Model Preset Examples - Complete Guide")
print("=" * 80)

# Set device
pysml.set_device('cpu')


# ===== Example 1: TransformerLM for Character-Level Language Modeling =====
print("\n" + "=" * 80)
print("Example 1: TransformerLM - Character-Level Language Modeling")
print("=" * 80)

# Build vocabulary
special_tokens = ["<pad>", "<eos>"]
chars = list("abcdefghijklmnopqrstuvwxyz ")
vocab = {}
idx = 0
for token in special_tokens:
    vocab[token] = idx
    idx += 1
for char in chars:
    vocab[char] = idx
    idx += 1
vocab_index = {v: k for k, v in vocab.items()}
vocab_size = len(vocab)

print(f"Vocabulary size: {vocab_size}")

def encode_text(text, max_len=32):
    """Encode text to token indices"""
    tokens = []
    for char in text[:max_len-1]:
        tokens.append(vocab.get(char, vocab["<pad>"]))
    tokens.append(vocab["<eos>"])
    while len(tokens) < max_len:
        tokens.append(vocab["<pad>"])
    return np.array(tokens, dtype=np.int32)

def decode_text(tokens):
    """Decode token indices to text"""
    if isinstance(tokens, pysml.Tensor):
        tokens = tokens.data
    if hasattr(tokens, 'asnumpy'):
        tokens = tokens.asnumpy()
    
    text = ""
    for idx in tokens:
        idx = int(idx)
        if idx >= vocab_size:
            continue
        char = vocab_index[idx]
        if char == "<eos>":
            break
        if char == "<pad>":
            continue
        text += char
    return text

# Create TransformerLM with custom vocab size
print("\nCreating TransformerLM from TINY preset...")
transformer = nn.TransformerLM.from_preset('TINY', vocab_size=vocab_size)
print(f"Total parameters: {sum(p.size for p in transformer.parameters()):,}")

# Training data
training_pairs = [
    ("hello", "hello world"),
    ("good", "good morning"),
    ("thank", "thank you"),
    ("how", "how are you"),
    ("nice", "nice day"),
]

print("\nPreparing training data...")
train_inputs = []
train_targets = []
for inp, tgt in training_pairs:
    train_inputs.append(encode_text(inp))
    train_targets.append(encode_text(tgt))
    print(f"  '{inp}' -> '{tgt}'")

train_inputs = np.array(train_inputs)
train_targets = np.array(train_targets)

# Setup optimizer
optimizer = nn.AdamW(transformer.parameters(), lr=0.001, weight_decay=0.01)

# Training loop
print("\nTraining TransformerLM...")
transformer.train()
epochs = 30

for epoch in range(epochs):
    total_loss = 0
    
    for i in range(len(train_inputs)):
        input_ids = pysml.Tensor(train_inputs[i:i+1], requires_grad=False)
        target_ids = pysml.Tensor(train_targets[i:i+1], requires_grad=False)
        
        optimizer.zero_grad()
        output = transformer(input_ids)
        
        # One-hot encode targets
        target_one_hot = F.one_hot(target_ids, vocab_size)
        loss = pysml.mse_loss(output, target_one_hot, reduction='mean')
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_inputs)
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3d}/{epochs}, Loss: {avg_loss:.6f}")

# Test generation
print("\nTesting generation...")
transformer.eval()
test_prompts = ["hello", "good", "thank"]

for prompt in test_prompts:
    input_ids = pysml.Tensor(encode_text(prompt).reshape(1, -1), requires_grad=False)
    output = transformer(input_ids)
    
    output_data = output.data
    if hasattr(output_data, 'asnumpy'):
        output_data = output_data.asnumpy()
    
    predicted = np.argmax(output_data, axis=-1)[0]
    decoded = decode_text(predicted)
    print(f"  Input: '{prompt}' -> Output: '{decoded}'")


# ===== Example 2: Classifier for MNIST-like Classification =====
print("\n" + "=" * 80)
print("Example 2: Classifier - Image Classification")
print("=" * 80)

# Create classifier
print("Creating Classifier from SIMPLE_MLP preset...")
classifier = nn.Classifier.from_preset('SIMPLE_MLP')
print(f"Total parameters: {sum(p.size for p in classifier.parameters()):,}")

# Generate synthetic MNIST-like data
np.random.seed(42)
n_samples = 500
n_classes = 10

X_train = np.random.randn(n_samples, 784).astype(np.float32) * 0.5
y_train = np.random.randint(0, n_classes, n_samples)

# One-hot encode
y_train_onehot = np.zeros((n_samples, n_classes), dtype=np.float32)
for i in range(n_samples):
    y_train_onehot[i, y_train[i]] = 1.0

print(f"Training data: {X_train.shape}, Labels: {y_train.shape}")

# Setup optimizer
optimizer = nn.Adam(classifier.parameters(), lr=0.001)

# Training
print("\nTraining Classifier...")
classifier.train()
batch_size = 32
epochs = 20

for epoch in range(epochs):
    total_loss = 0
    n_batches = 0
    
    for i in range(0, n_samples, batch_size):
        end_idx = min(i + batch_size, n_samples)
        
        X_batch = pysml.Tensor(X_train[i:end_idx], requires_grad=False)
        y_batch = pysml.Tensor(y_train_onehot[i:end_idx], requires_grad=False)
        
        optimizer.zero_grad()
        predictions = classifier(X_batch)
        loss = pysml.mse_loss(predictions, y_batch)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1:3d}/{epochs}, Loss: {total_loss/n_batches:.6f}")

# Evaluation
print("\nEvaluating...")
classifier.eval()
X_test = pysml.Tensor(X_train[:100], requires_grad=False)
predictions = classifier(X_test)
pred_classes = np.argmax(predictions.data, axis=1)
accuracy = np.mean(pred_classes == y_train[:100])
print(f"Accuracy: {accuracy*100:.2f}%")


# ===== Example 3: VAE for Generative Modeling =====
print("\n" + "=" * 80)
print("Example 3: VAE - Generative Modeling")
print("=" * 80)

# Create VAE
print("Creating VAE from MNIST preset...")
vae = nn.VAE.from_preset('MNIST')
print(f"Total parameters: {sum(p.size for p in vae.parameters()):,}")
print(f"Latent dimension: {vae.latent_dim}")

# Generate synthetic data
np.random.seed(123)
n_samples = 300
X_vae = np.random.randn(n_samples, 784).astype(np.float32) * 0.3 + 0.5
X_vae = np.clip(X_vae, 0, 1)

# VAE loss function
def vae_loss(recon_x, x, mu, logvar):
    """VAE loss = reconstruction + KL divergence"""
    recon_loss = pysml.mse_loss(recon_x, x, reduction='sum')
    
    # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_div = -0.5 * pysml.sum(1 + logvar - mu * mu - pysml.exp(logvar))
    
    return recon_loss + kl_div

# Setup optimizer
optimizer = nn.Adam(vae.parameters(), lr=0.001)

# Training
print("\nTraining VAE...")
vae.train()
batch_size = 50
epochs = 25

for epoch in range(epochs):
    total_loss = 0
    n_batches = 0
    
    for i in range(0, n_samples, batch_size):
        end_idx = min(i + batch_size, n_samples)
        
        X_batch = pysml.Tensor(X_vae[i:end_idx], requires_grad=False)
        
        optimizer.zero_grad()
        reconstruction, mu, logvar = vae(X_batch)
        loss = vae_loss(reconstruction, X_batch, mu, logvar)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1:3d}/{epochs}, Loss: {total_loss/n_batches:.2f}")

# Generate samples
print("\nGenerating new samples...")
vae.eval()
generated = vae.generate(num_samples=5)
print(f"Generated shape: {generated.shape}")
print(f"Sample values (first 10): {generated.data[0, :10]}")


# ===== Example 4: GAN for Image Generation =====
print("\n" + "=" * 80)
print("Example 4: GAN - Adversarial Training")
print("=" * 80)

# Create GAN
print("Creating GAN from SIMPLE preset...")
gan = nn.GAN.from_preset('SIMPLE')
print(f"Generator parameters: {sum(p.size for p in gan.generator.parameters()):,}")
print(f"Discriminator parameters: {sum(p.size for p in gan.discriminator.parameters()):,}")

# Generate synthetic real data
np.random.seed(456)
n_samples = 200
real_data = np.random.randn(n_samples, 784).astype(np.float32) * 0.3

# Setup optimizers
optimizer_g = nn.Adam(gan.generator.parameters(), lr=0.0002)
optimizer_d = nn.Adam(gan.discriminator.parameters(), lr=0.0002)

# Training
print("\nTraining GAN...")
batch_size = 32
epochs = 20

for epoch in range(epochs):
    total_d_loss = 0
    total_g_loss = 0
    n_batches = 0
    
    for i in range(0, n_samples, batch_size):
        end_idx = min(i + batch_size, n_samples)
        batch_real = pysml.Tensor(real_data[i:end_idx], requires_grad=False)
        current_batch_size = end_idx - i
        
        # Train Discriminator
        optimizer_d.zero_grad()
        
        # Real samples
        real_labels = pysml.ones(current_batch_size, 1, requires_grad=False)
        real_output = gan.discriminate(batch_real)
        d_loss_real = pysml.mse_loss(real_output, real_labels)
        
        # Fake samples
        fake_samples = gan.generate(current_batch_size)
        fake_labels = pysml.zeros(current_batch_size, 1, requires_grad=False)
        fake_output = gan.discriminate(fake_samples)
        d_loss_fake = pysml.mse_loss(fake_output, fake_labels)
        
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_d.step()
        
        # Train Generator
        optimizer_g.zero_grad()
        
        fake_samples = gan.generate(current_batch_size)
        fake_output = gan.discriminate(fake_samples)
        g_loss = pysml.mse_loss(fake_output, real_labels)
        
        g_loss.backward()
        optimizer_g.step()
        
        total_d_loss += d_loss.item()
        total_g_loss += g_loss.item()
        n_batches += 1
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1:3d}/{epochs}, D_Loss: {total_d_loss/n_batches:.4f}, "
              f"G_Loss: {total_g_loss/n_batches:.4f}")

print("\nGenerating samples with trained GAN...")
generated_imgs = gan.generate(batch_size=5)
print(f"Generated images shape: {generated_imgs.shape}")


# ===== Example 5: AutoEncoder for Dimensionality Reduction =====
print("\n" + "=" * 80)
print("Example 5: AutoEncoder - Dimensionality Reduction")
print("=" * 80)

# Create AutoEncoder
print("Creating AutoEncoder from SMALL preset...")
autoencoder = nn.AutoEncoder.from_preset('SMALL')
print(f"Total parameters: {sum(p.size for p in autoencoder.parameters()):,}")

# Generate synthetic data
np.random.seed(789)
n_samples = 400
X_ae = np.random.randn(n_samples, 784).astype(np.float32) * 0.3 + 0.5
X_ae = np.clip(X_ae, 0, 1)

# Setup optimizer
optimizer = nn.Adam(autoencoder.parameters(), lr=0.001)

# Training
print("\nTraining AutoEncoder...")
autoencoder.train()
batch_size = 50
epochs = 25

for epoch in range(epochs):
    total_loss = 0
    n_batches = 0
    
    for i in range(0, n_samples, batch_size):
        end_idx = min(i + batch_size, n_samples)
        
        X_batch = pysml.Tensor(X_ae[i:end_idx], requires_grad=False)
        
        optimizer.zero_grad()
        reconstruction = autoencoder(X_batch)
        loss = pysml.mse_loss(reconstruction, X_batch)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1:3d}/{epochs}, Loss: {total_loss/n_batches:.6f}")

# Test reconstruction
print("\nTesting reconstruction...")
autoencoder.eval()
test_sample = pysml.Tensor(X_ae[:5], requires_grad=False)
reconstructed = autoencoder(test_sample)
recon_error = pysml.mse_loss(reconstructed, test_sample)
print(f"Reconstruction error: {recon_error.item():.6f}")

# Test encoding
encoded = autoencoder.encode(test_sample)
print(f"Encoded representation shape: {encoded.shape}")


# ===== Example 6: Comparing Different Classifier Presets =====
print("\n" + "=" * 80)
print("Example 6: Comparing Different Classifier Presets")
print("=" * 80)

# Generate test data
np.random.seed(999)
n_test = 200
X_compare = np.random.randn(n_test, 784).astype(np.float32) * 0.3
y_compare = np.random.randint(0, 10, n_test)
y_compare_onehot = np.zeros((n_test, 10), dtype=np.float32)
for i in range(n_test):
    y_compare_onehot[i, y_compare[i]] = 1.0

X_compare_tensor = pysml.Tensor(X_compare, requires_grad=False)
y_compare_tensor = pysml.Tensor(y_compare_onehot, requires_grad=False)

# Test different presets
presets = ['SIMPLE_MLP', 'DEEP', 'WIDE']

for preset_name in presets:
    print(f"\nTesting {preset_name}...")
    
    model = nn.Classifier.from_preset(preset_name)
    optimizer = nn.Adam(model.parameters(), lr=0.001)
    
    print(f"  Parameters: {sum(p.size for p in model.parameters()):,}")
    
    # Quick training
    model.train()
    for epoch in range(10):
        optimizer.zero_grad()
        predictions = model(X_compare_tensor)
        loss = pysml.mse_loss(predictions, y_compare_tensor)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/10, Loss: {loss.item():.6f}")


# ===== Example 7: Using Different Optimizers =====
print("\n" + "=" * 80)
print("Example 7: Comparing Optimizers (SGD vs Adam vs AdamW)")
print("=" * 80)

# Create three identical classifiers
model_sgd = nn.Classifier.from_preset('SIMPLE_MLP')
model_adam = nn.Classifier.from_preset('SIMPLE_MLP')
model_adamw = nn.Classifier.from_preset('SIMPLE_MLP')

# Setup different optimizers
opt_sgd = nn.SGD(model_sgd.parameters(), lr=0.01, momentum=0.9)
opt_adam = nn.Adam(model_adam.parameters(), lr=0.001)
opt_adamw = nn.AdamW(model_adamw.parameters(), lr=0.001, weight_decay=0.01)

# Generate simple data
np.random.seed(111)
X_opt = pysml.Tensor(np.random.randn(100, 784).astype(np.float32) * 0.3, requires_grad=False)
y_opt_labels = np.random.randint(0, 10, 100)
y_opt = np.zeros((100, 10), dtype=np.float32)
for i in range(100):
    y_opt[i, y_opt_labels[i]] = 1.0
y_opt_tensor = pysml.Tensor(y_opt, requires_grad=False)

optimizers = [
    ('SGD (momentum=0.9)', model_sgd, opt_sgd),
    ('Adam', model_adam, opt_adam),
    ('AdamW', model_adamw, opt_adamw)
]

for opt_name, model, optimizer in optimizers:
    print(f"\nTraining with {opt_name}...")
    model.train()
    
    for epoch in range(15):
        optimizer.zero_grad()
        predictions = model(X_opt)
        loss = pysml.mse_loss(predictions, y_opt_tensor)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/15, Loss: {loss.item():.6f}")


# ===== Summary =====
print("\n" + "=" * 80)
print("Summary - All Preset Models Demonstrated")
print("=" * 80)

print("\n✓ Example 1: TransformerLM - Character-level language modeling")
print("  • Used TINY preset with custom vocab size")
print("  • Trained with AdamW optimizer")
print("  • Generated text from prompts")

print("\n✓ Example 2: Classifier - Image classification")
print("  • Used SIMPLE_MLP preset")
print("  • Mini-batch training with Adam")
print("  • Evaluated accuracy on test set")

print("\n✓ Example 3: VAE - Generative modeling")
print("  • Used MNIST preset")
print("  • Implemented full VAE loss (reconstruction + KL)")
print("  • Generated new samples from latent space")

print("\n✓ Example 4: GAN - Adversarial training")
print("  • Used SIMPLE preset")
print("  • Alternating discriminator and generator training")
print("  • Generated synthetic images")

print("\n✓ Example 5: AutoEncoder - Dimensionality reduction")
print("  • Used SMALL preset")
print("  • Reconstruction-based training")
print("  • Extracted latent representations")

print("\n✓ Example 6: Preset comparison")
print("  • Compared SIMPLE_MLP, DEEP, and WIDE classifiers")
print("  • Different architectures, same task")

print("\n✓ Example 7: Optimizer comparison")
print("  • Tested SGD, Adam, and AdamW")
print("  • Same model, different optimization strategies")

print("\n" + "=" * 80)
print("All preset models support:")
print("  • Full gradient computation via backpropagation")
print("  • Multiple optimizer choices")
print("  • Training/evaluation modes")
print("  • Easy customization with from_preset()")
print("=" * 80)