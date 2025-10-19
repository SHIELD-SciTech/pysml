"""
PySML Preset Models
Pre-configured architectures for common tasks
"""

import pysml
import pysml.nn as nn
import pysml.nn.functional as F
from typing import Optional, List


# ===== TRANSFORMER PRESETS =====

class TransformerConfig:
    """Configuration presets for Transformers"""
    
    # Small model for experimentation
    TINY = {
        'vocab_size': 10000,
        'd_model': 128,
        'num_layers': 2,
        'num_heads': 4,
        'd_ff': 512,
        'max_seq_len': 512,
        'dropout': 0.1,
    }
    
    # Small model (GPT-2 small-like)
    SMALL = {
        'vocab_size': 50257,
        'd_model': 768,
        'num_layers': 12,
        'num_heads': 12,
        'd_ff': 3072,
        'max_seq_len': 1024,
        'dropout': 0.1,
    }
    
    # Medium model (GPT-2 medium-like)
    MEDIUM = {
        'vocab_size': 50257,
        'd_model': 1024,
        'num_layers': 24,
        'num_heads': 16,
        'd_ff': 4096,
        'max_seq_len': 1024,
        'dropout': 0.1,
    }
    
    # Large model (GPT-2 large-like)
    LARGE = {
        'vocab_size': 50257,
        'd_model': 1280,
        'num_layers': 36,
        'num_heads': 20,
        'd_ff': 5120,
        'max_seq_len': 1024,
        'dropout': 0.1,
    }
    
    # Custom BERT-like
    BERT_BASE = {
        'vocab_size': 30522,
        'd_model': 768,
        'num_layers': 12,
        'num_heads': 12,
        'd_ff': 3072,
        'max_seq_len': 512,
        'dropout': 0.1,
    }


class LayerNorm(nn.Module):
    """Layer Normalization - Fixed for PySML"""
    
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.normalized_shape = normalized_shape if isinstance(normalized_shape, (list, tuple)) else (normalized_shape,)
        
        # Learnable parameters
        self.gamma = pysml.ones(*self.normalized_shape, requires_grad=True)
        self.beta = pysml.zeros(*self.normalized_shape, requires_grad=True)
    
    def forward(self, x):
        # Compute mean and variance over the last dimension(s)
        mean = pysml.mean(x, axis=-1, keepdims=True)
        var = pysml.var(x, axis=-1, keepdims=True)
        
        # Create epsilon tensor for proper broadcasting
        eps_tensor = pysml.Tensor(self.eps, backend=x.backend, device=x.device)
        
        # Normalize: (x - mean) / sqrt(var + eps)
        x_centered = x - mean
        std = pysml.sqrt(var + eps_tensor)
        x_norm = x_centered / std
        
        # Scale and shift
        return self.gamma * x_norm + self.beta


class TransformerBlock(nn.Module):
    """Single Transformer Block - Fixed for PySML"""
    
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        
        # Simplified attention (just a linear projection for now)
        self.attention = nn.Linear(d_model, d_model)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        
        # Layer normalization
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        
        self.dropout = dropout
    
    def forward(self, x):
        # Self-attention with residual connection
        residual = x
        x = self.attention(x)
        
        if self.training and self.dropout > 0:
            x = pysml.dropout(x, p=self.dropout, training=True)
        
        x = residual + x
        x = self.norm1(x)
        
        # Feed-forward with residual connection
        residual = x
        x = self.ffn(x)
        
        if self.training and self.dropout > 0:
            x = pysml.dropout(x, p=self.dropout, training=True)
        
        x = residual + x
        x = self.norm2(x)
        
        return x


class TransformerLM(nn.Module):
    """Pre-configured Transformer Language Model - Fixed for PySML"""
    
    @classmethod
    def from_preset(cls, preset='SMALL', vocab_size=None):
        """
        Create model from preset configuration
        
        Args:
            preset: 'TINY', 'SMALL', 'MEDIUM', 'LARGE', or 'BERT_BASE'
            vocab_size: Optional override for vocabulary size
        
        Returns:
            Configured TransformerLM instance
        """
        config = getattr(TransformerConfig, preset).copy()
        
        # Allow vocabulary size override
        if vocab_size is not None:
            config['vocab_size'] = vocab_size
        
        return cls(**config)
    
    def __init__(self, vocab_size, d_model, num_layers, num_heads, 
                 d_ff, max_seq_len, dropout=0.1):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.dropout_p = dropout
        
        print(f"Initializing TransformerLM:")
        print(f"  vocab_size={vocab_size}, d_model={d_model}")
        print(f"  num_layers={num_layers}, num_heads={num_heads}")
        print(f"  d_ff={d_ff}, max_seq_len={max_seq_len}")
        
        # Token embeddings - stored as a parameter matrix
        self.token_embedding = pysml.randn(vocab_size, d_model, requires_grad=True) * 0.02
        
        # Positional embeddings
        self.pos_embedding = pysml.randn(max_seq_len, d_model, requires_grad=True) * 0.02
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.ln_f = LayerNorm(d_model)
        
        # Language model head
        self.lm_head = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Token indices of shape (batch_size, seq_len)
        
        Returns:
            Logits of shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = x.shape
        
        # Validate input
        x_data = x.data
        if hasattr(x_data, 'asnumpy'):
            x_data = x_data.asnumpy()
        
        if x_data.min() < 0 or x_data.max() >= self.vocab_size:
            raise ValueError(
                f"Input indices out of bounds: min={x_data.min()}, "
                f"max={x_data.max()}, vocab_size={self.vocab_size}"
            )
        
        # Embedding lookup using F.embedding
        token_emb = F.embedding(x, self.token_embedding)
        
        # Add positional embeddings
        pos_emb = self.pos_embedding[:seq_len]
        pos_emb_expanded = pysml.reshape(pos_emb, (1, seq_len, self.d_model))
        x = token_emb + pos_emb_expanded
        
        # Apply dropout to embeddings
        if self.training and self.dropout_p > 0:
            x = pysml.dropout(x, p=self.dropout_p, training=True)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Project to vocabulary
        # Flatten for linear layer
        x_flat = pysml.reshape(x, (batch_size * seq_len, self.d_model))
        logits = self.lm_head(x_flat)
        
        # Reshape back
        logits = pysml.reshape(logits, (batch_size, seq_len, self.vocab_size))
        
        return logits
    
    def parameters(self):
        """Return all trainable parameters"""
        params = [self.token_embedding, self.pos_embedding]
        
        # Add parameters from all blocks
        for block in self.blocks:
            params.extend(block.parameters())
        
        # Add final layer norm parameters
        params.extend(self.ln_f.parameters())
        
        # Add LM head parameters
        params.extend(self.lm_head.parameters())
        
        return params
    
    def generate(self, start_tokens, max_new_tokens=50, temperature=1.0):
        """
        Generate text given starting tokens
        
        Args:
            start_tokens: Initial token indices, shape (1, seq_len)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
        
        Returns:
            Generated token indices
        """
        self.eval()
        
        import numpy as np
        
        current_tokens = start_tokens
        
        for _ in range(max_new_tokens):
            # Get predictions
            logits = self(current_tokens)
            
            # Get last token predictions
            next_token_logits = logits[:, -1, :]  # (1, vocab_size)
            
            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # Convert to probabilities
            probs = pysml.softmax(next_token_logits, axis=-1)
            
            # Sample from distribution
            probs_data = probs.data
            if hasattr(probs_data, 'asnumpy'):
                probs_data = probs_data.asnumpy()
            
            # Simple argmax sampling (can be improved with temperature sampling)
            next_token = np.argmax(probs_data, axis=-1)
            
            # Append to sequence
            next_token_tensor = pysml.Tensor(next_token.reshape(1, 1), requires_grad=False)
            current_tokens = pysml.concatenate([current_tokens, next_token_tensor], axis=1)
            
            # Check if we've exceeded max sequence length
            if current_tokens.shape[1] >= self.max_seq_len:
                break
        
        return current_tokens



# ===== CLASSIFIER PRESETS =====

class ClassifierConfig:
    """Configuration presets for Classifiers"""
    
    # Simple MLP
    SIMPLE_MLP = {
        'input_dim': 784,
        'hidden_dims': [256, 128],
        'num_classes': 10,
        'activation': 'relu',
        'dropout': 0.2,
    }
    
    # Deep classifier
    DEEP = {
        'input_dim': 784,
        'hidden_dims': [512, 512, 256, 128],
        'num_classes': 10,
        'activation': 'gelu',
        'dropout': 0.3,
    }
    
    # Wide classifier
    WIDE = {
        'input_dim': 784,
        'hidden_dims': [1024, 1024],
        'num_classes': 10,
        'activation': 'relu',
        'dropout': 0.4,
    }


class Classifier(nn.Module):
    """Pre-configured Classifier for various tasks"""
    
    @classmethod
    def from_preset(cls, preset='SIMPLE_MLP'):
        """
        Create classifier from preset
        
        Args:
            preset: 'SIMPLE_MLP', 'DEEP', or 'WIDE'
        """
        config = getattr(ClassifierConfig, preset)
        return cls(**config)
    
    def __init__(self, input_dim, hidden_dims, num_classes, 
                 activation='relu', dropout=0.2):
        super().__init__()
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Activation
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'gelu':
                layers.append(nn.GELU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            
            # Dropout
            if dropout > 0:
                # Store dropout for manual application
                self.dropout_p = dropout
            
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
        self.dropout_p = dropout
    
    def forward(self, x):
        # Flatten input if needed
        if len(x.shape) > 2:
            batch_size = x.shape[0]
            x = x.reshape(batch_size, -1)
        
        # Pass through network with dropout
        for i, layer in enumerate(self.network.layers):
            x = layer(x)
            # Apply dropout after activations
            if self.training and self.dropout_p > 0 and isinstance(layer, (nn.ReLU, nn.GELU, nn.Tanh)):
                x = pysml.dropout(x, p=self.dropout_p, training=True)
        
        return x


# ===== IMAGE GENERATION PRESETS =====

class VAEConfig:
    """Configuration presets for VAE"""
    
    MNIST = {
        'input_dim': 784,
        'latent_dim': 20,
        'hidden_dims': [512, 256],
    }
    
    CIFAR10 = {
        'input_dim': 3072,
        'latent_dim': 128,
        'hidden_dims': [1024, 512, 256],
    }
    
    LARGE = {
        'input_dim': 12288,  # 64x64x3
        'latent_dim': 256,
        'hidden_dims': [2048, 1024, 512],
    }


class VAE(nn.Module):
    """Variational Autoencoder for image generation"""
    
    @classmethod
    def from_preset(cls, preset='MNIST'):
        """Create VAE from preset"""
        config = getattr(VAEConfig, preset)
        return cls(**config)
    
    def __init__(self, input_dim, latent_dim, hidden_dims):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
        
        # Decoder
        decoder_layers = []
        decoder_layers.append(nn.Linear(latent_dim, hidden_dims[-1]))
        decoder_layers.append(nn.ReLU())
        
        for i in range(len(hidden_dims) - 1, 0, -1):
            decoder_layers.extend([
                nn.Linear(hidden_dims[i], hidden_dims[i-1]),
                nn.ReLU()
            ])
        
        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
        decoder_layers.append(nn.Sigmoid())
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x):
        """Encode input to latent space"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = pysml.exp(logvar * 0.5)
        eps = pysml.randn(*mu.shape)
        return mu + eps * std
    
    def decode(self, z):
        """Decode latent vector to output"""
        return self.decoder(z)
    
    def forward(self, x):
        """Full forward pass"""
        # Flatten input
        batch_size = x.shape[0]
        x_flat = x.reshape(batch_size, -1)
        
        # Encode
        mu, logvar = self.encode(x_flat)
        
        # Sample
        z = self.reparameterize(mu, logvar)
        
        # Decode
        reconstruction = self.decode(z)
        
        return reconstruction, mu, logvar
    
    def generate(self, num_samples):
        """Generate new samples"""
        z = pysml.randn(num_samples, self.latent_dim)
        return self.decode(z)


class GANConfig:
    """Configuration presets for GAN"""
    
    SIMPLE = {
        'latent_dim': 100,
        'image_dim': 784,
        'hidden_dim': 256,
    }
    
    DCGAN_SMALL = {
        'latent_dim': 100,
        'image_dim': 3072,  # 32x32x3
        'hidden_dim': 512,
    }
    
    LARGE = {
        'latent_dim': 256,
        'image_dim': 12288,  # 64x64x3
        'hidden_dim': 1024,
    }


class GAN(nn.Module):
    """Generative Adversarial Network"""
    
    @classmethod
    def from_preset(cls, preset='SIMPLE'):
        """Create GAN from preset"""
        config = getattr(GANConfig, preset)
        return cls(**config)
    
    def __init__(self, latent_dim, image_dim, hidden_dim):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.image_dim = image_dim
        
        # Generator
        self.generator = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, image_dim),
            nn.Tanh()
        )
        
        # Discriminator
        self.discriminator = nn.Sequential(
            nn.Linear(image_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def generate(self, batch_size):
        """Generate fake images"""
        z = pysml.randn(batch_size, self.latent_dim)
        return self.generator(z)
    
    def discriminate(self, x):
        """Discriminate real vs fake"""
        return self.discriminator(x)


# ===== SEQUENCE MODELS =====

class RNNConfig:
    """Configuration presets for RNN/LSTM"""
    
    SMALL = {
        'input_size': 128,
        'hidden_size': 256,
        'num_layers': 2,
        'output_size': 10,
    }
    
    MEDIUM = {
        'input_size': 256,
        'hidden_size': 512,
        'num_layers': 3,
        'output_size': 10,
    }
    
    LARGE = {
        'input_size': 512,
        'hidden_size': 1024,
        'num_layers': 4,
        'output_size': 10,
    }


class SimpleLSTM(nn.Module):
    """Simplified LSTM for sequence tasks"""
    
    @classmethod
    def from_preset(cls, preset='SMALL'):
        """Create LSTM from preset"""
        config = getattr(RNNConfig, preset)
        return cls(**config)
    
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Simplified: stack of Linear layers for each LSTM cell
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else hidden_size
            self.layers.append(nn.Linear(layer_input_size + hidden_size, hidden_size * 4))
        
        # Output layer
        self.fc_out = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        """
        Simplified forward pass
        x: (batch, seq_len, input_size)
        """
        batch_size, seq_len, _ = x.shape
        
        # Initialize hidden states (simplified)
        h = pysml.zeros(batch_size, self.hidden_size)
        c = pysml.zeros(batch_size, self.hidden_size)
        
        # Process sequence
        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            
            # LSTM cell (simplified)
            for layer in self.layers:
                combined = pysml.concatenate([x_t, h], axis=1)
                gates = layer(combined)
                
                # Split into gates (simplified)
                gate_size = self.hidden_size
                i_gate = pysml.sigmoid(gates[:, :gate_size])
                f_gate = pysml.sigmoid(gates[:, gate_size:gate_size*2])
                o_gate = pysml.sigmoid(gates[:, gate_size*2:gate_size*3])
                g_gate = pysml.tanh(gates[:, gate_size*3:])
                
                # Update cell state
                c = f_gate * c + i_gate * g_gate
                h = o_gate * pysml.tanh(c)
                
                x_t = h
            
            outputs.append(h)
        
        # Stack outputs
        output = pysml.stack(outputs, axis=1)
        
        # Final output
        return self.fc_out(output[:, -1, :])


# ===== AUTOENCODER PRESETS =====

class AutoEncoderConfig:
    """Configuration presets for AutoEncoders"""
    
    SMALL = {
        'input_dim': 784,
        'encoding_dims': [256, 128, 64],
    }
    
    DEEP = {
        'input_dim': 784,
        'encoding_dims': [512, 256, 128, 64, 32],
    }
    
    WIDE = {
        'input_dim': 3072,
        'encoding_dims': [1024, 512, 256],
    }


class AutoEncoder(nn.Module):
    """Standard AutoEncoder"""
    
    @classmethod
    def from_preset(cls, preset='SMALL'):
        """Create AutoEncoder from preset"""
        config = getattr(AutoEncoderConfig, preset)
        return cls(**config)
    
    def __init__(self, input_dim, encoding_dims):
        super().__init__()
        
        self.input_dim = input_dim
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for dim in encoding_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU()
            ])
            prev_dim = dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        decoding_dims = list(reversed(encoding_dims[:-1])) + [input_dim]
        
        for dim in decoding_dims:
            decoder_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU() if dim != input_dim else nn.Sigmoid()
            ])
            prev_dim = dim
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x):
        """Encode input"""
        return self.encoder(x)
    
    def decode(self, z):
        """Decode latent representation"""
        return self.decoder(z)
    
    def forward(self, x):
        """Full forward pass"""
        batch_size = x.shape[0]
        x_flat = x.reshape(batch_size, -1)
        z = self.encode(x_flat)
        reconstruction = self.decode(z)
        return reconstruction


# ===== UTILITY FUNCTION =====

def list_presets(model_type='all'):
    """
    List all available presets
    
    Args:
        model_type: 'transformer', 'classifier', 'vae', 'gan', 'rnn', 'autoencoder', or 'all'
    """
    presets = {
        'transformer': ['TINY', 'SMALL', 'MEDIUM', 'LARGE', 'BERT_BASE'],
        'classifier': ['SIMPLE_MLP', 'DEEP', 'WIDE'],
        'vae': ['MNIST', 'CIFAR10', 'LARGE'],
        'gan': ['SIMPLE', 'DCGAN_SMALL', 'LARGE'],
        'rnn': ['SMALL', 'MEDIUM', 'LARGE'],
        'autoencoder': ['SMALL', 'DEEP', 'WIDE'],
    }
    
    if model_type == 'all':
        return presets
    else:
        return {model_type: presets.get(model_type, [])}


# ===== EXAMPLE USAGE =====

if __name__ == "__main__":
    print("=" * 70)
    print("PySML Preset Models Demo")
    print("=" * 70)
    print()
    
    # List all presets
    print("Available Presets:")
    print("-" * 70)
    for model_type, preset_list in list_presets().items():
        print(f"{model_type.upper()}:")
        for preset in preset_list:
            print(f"  - {preset}")
    print()
    
    # ===== Transformer Example =====
    print("=" * 70)
    print("1. Transformer LM (SMALL preset)")
    print("=" * 70)
    
    model = TransformerLM.from_preset('SMALL')
    print(f"Parameters: {sum(p.size for p in model.parameters()):,}")
    print(f"Config: vocab_size={model.vocab_size}, d_model={model.d_model}")
    print()
    
    # ===== Classifier Example =====
    print("=" * 70)
    print("2. Classifier (DEEP preset)")
    print("=" * 70)
    
    classifier = Classifier.from_preset('DEEP')
    print(classifier)
    print(f"Parameters: {sum(p.size for p in classifier.parameters()):,}")
    print()
    
    # Test
    x = pysml.randn(32, 784)
    output = classifier(x)
    print(f"Input: {x.shape} -> Output: {output.shape}")
    print()
    
    # ===== VAE Example =====
    print("=" * 70)
    print("3. VAE (MNIST preset)")
    print("=" * 70)
    
    vae = VAE.from_preset('MNIST')
    print(f"Parameters: {sum(p.size for p in vae.parameters()):,}")
    print(f"Latent dim: {vae.latent_dim}")
    print()
    
    # Test generation
    samples = vae.generate(num_samples=10)
    print(f"Generated samples shape: {samples.shape}")
    print()
    
    # ===== GAN Example =====
    print("=" * 70)
    print("4. GAN (SIMPLE preset)")
    print("=" * 70)
    
    gan = GAN.from_preset('SIMPLE')
    print(f"Generator parameters: {sum(p.size for p in gan.generator.parameters()):,}")
    print(f"Discriminator parameters: {sum(p.size for p in gan.discriminator.parameters()):,}")
    print()
    
    # Test generation
    fake_images = gan.generate(batch_size=16)
    print(f"Generated images shape: {fake_images.shape}")
    print()
    
    # ===== AutoEncoder Example =====
    print("=" * 70)
    print("5. AutoEncoder (SMALL preset)")
    print("=" * 70)
    
    ae = AutoEncoder.from_preset('SMALL')
    print(f"Parameters: {sum(p.size for p in ae.parameters()):,}")
    print()
    
    # Test
    x = pysml.randn(32, 784)
    reconstruction = ae(x)
    print(f"Input: {x.shape} -> Reconstruction: {reconstruction.shape}")
    print()
    
    # ===== Quick Training Example =====
    print("=" * 70)
    print("6. Quick Training Example")
    print("=" * 70)
    
    # Create a classifier
    model = Classifier.from_preset('SIMPLE_MLP')
    model.train()
    
    print("Training simple classifier...")
    for epoch in range(3):
        x = pysml.randn(64, 784)
        y = pysml.randn(64, 10)
        
        pred = model(x)
        loss = pysml.mse_loss(pred, y)
        loss.backward()
        
        # Simple SGD
        for p in model.parameters():
            if p.grad is not None:
                p.data = p.data - 0.01 * p.grad.data
                p.zero_grad()
        
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
    
    print()
    print("=" * 70)
    print("All preset models demonstrated!")
    print("=" * 70)