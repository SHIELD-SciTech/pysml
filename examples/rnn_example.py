import sys, os
sys.path.insert(0, os.getcwd().split("examples")[0])
import numpy as np
import pysml
from pysml.nn.rnn import RNN, LSTM, GRU
from pysml.nn.linear import Linear
from pysml.nn.module import Module
from pysml.nn.optim import AdamW
from pysml.nn import functional as F
from pysml.backend.context import device


# ==================== Example 1: Simple RNN ====================
print("="*70)
print("Example 1: Basic RNN for Sequence Classification")
print("="*70)

class SimpleRNNClassifier(Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.rnn = RNN(input_size, hidden_size, num_layers=1, batch_first=True)
        self.fc = Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        output, h_n = self.rnn(x)
        # Use last hidden state for classification
        last_hidden = h_n[-1]  # (batch, hidden_size)
        logits = self.fc(last_hidden)
        return logits

# Create model
model = SimpleRNNClassifier(input_size=10, hidden_size=32, num_classes=5)

# Dummy data
batch_size, seq_len, input_size = 8, 15, 10
x = pysml.Tensor(np.random.randn(batch_size, seq_len, input_size).astype(np.float32))
labels = np.random.randint(0, 5, batch_size)

# Forward pass
output = model(x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")

# Training step
optimizer = AdamW(model.parameters(), lr=0.001)
optimizer.zero_grad()
loss = F.cross_entropy(output, labels)
print(f"Loss: {loss.item():.4f}")
loss.backward()
optimizer.step()
print("Training step completed\n")


# ==================== Example 2: LSTM for Sequence Prediction ====================
print("="*70)
print("Example 2: LSTM for Next Token Prediction")
print("="*70)

class LSTMPredictor(Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers=2):
        super().__init__()
        from pysml.nn.module import Embedding
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.lstm = LSTM(embedding_dim, hidden_size, num_layers=num_layers, 
                         batch_first=True, dropout=0.2)
        self.fc = Linear(hidden_size, vocab_size)
    
    def forward(self, x):
        # x: (batch, seq_len) - token indices
        embeds = self.embedding(x)  # (batch, seq_len, embedding_dim)
        output, (h_n, c_n) = self.lstm(embeds)
        # output: (batch, seq_len, hidden_size)
        logits = self.fc(output)  # (batch, seq_len, vocab_size)
        return logits

# Create model
vocab_size = 50
model = LSTMPredictor(vocab_size=vocab_size, embedding_dim=16, hidden_size=32, num_layers=2)

# Dummy sequence data
batch_size, seq_len = 4, 10
x = pysml.Tensor(np.random.randint(0, vocab_size, (batch_size, seq_len)).astype(np.int32))
targets = pysml.Tensor(np.random.randint(0, vocab_size, (batch_size, seq_len)).astype(np.int32))

# Forward pass
output = model(x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")

# Training
optimizer = AdamW(model.parameters(), lr=0.001)
model.train()

for epoch in range(5):
    optimizer.zero_grad()
    output = model(x)
    
    # Reshape for loss
    output_flat = output.view(batch_size * seq_len, vocab_size)
    targets_flat = targets.data.reshape(-1)
    
    loss = F.cross_entropy(output_flat, targets_flat)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 2 == 0:
        print(f"Epoch [{epoch+1}/5], Loss: {loss.item():.4f}")

print("LSTM training completed\n")


# ==================== Example 3: Bidirectional GRU ====================
print("="*70)
print("Example 3: Bidirectional GRU for Sentiment Analysis")
print("="*70)

class BiGRUSentiment(Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes):
        super().__init__()
        from pysml.nn.module import Embedding
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.gru = GRU(embedding_dim, hidden_size, num_layers=1, 
                       batch_first=True, bidirectional=True)
        # Bidirectional doubles the hidden size
        self.fc = Linear(hidden_size * 2, num_classes)
    
    def forward(self, x):
        embeds = self.embedding(x)
        output, h_n = self.gru(embeds)
        # Concatenate final forward and backward hidden states
        # h_n shape: (2, batch, hidden_size) for bidirectional single layer
        hidden_cat = pysml.Tensor(np.concatenate([h_n.data[0], h_n.data[1]], axis=-1))
        logits = self.fc(hidden_cat)
        return logits

# Create model
model = BiGRUSentiment(vocab_size=100, embedding_dim=32, hidden_size=64, num_classes=2)

# Dummy data
batch_size, seq_len = 16, 20
x = pysml.Tensor(np.random.randint(0, 100, (batch_size, seq_len)).astype(np.int32))
labels = np.random.randint(0, 2, batch_size)

# Forward pass
output = model(x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")

# Training
optimizer = AdamW(model.parameters(), lr=0.001)
optimizer.zero_grad()
loss = F.cross_entropy(output, labels)
print(f"Loss: {loss.item():.4f}")
loss.backward()
optimizer.step()
print("Bidirectional GRU completed\n")


# ==================== Example 4: Sequence-to-Sequence with LSTM ====================
print("="*70)
print("Example 4: Simple Seq2Seq with LSTM")
print("="*70)

class Seq2SeqLSTM(Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super().__init__()
        from pysml.nn.module import Embedding
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.encoder = LSTM(embedding_dim, hidden_size, num_layers=1, batch_first=True)
        self.decoder = LSTM(embedding_dim, hidden_size, num_layers=1, batch_first=True)
        self.fc = Linear(hidden_size, vocab_size)
    
    def forward(self, src, tgt):
        # Encode
        src_embeds = self.embedding(src)
        _, (h_n, c_n) = self.encoder(src_embeds)
        
        # Decode
        tgt_embeds = self.embedding(tgt)
        output, _ = self.decoder(tgt_embeds, (h_n, c_n))
        
        # Generate predictions
        logits = self.fc(output)
        return logits

# Create model
model = Seq2SeqLSTM(vocab_size=50, embedding_dim=32, hidden_size=64)

# Dummy data (e.g., translation)
batch_size = 4
src_len, tgt_len = 8, 10
src = pysml.Tensor(np.random.randint(0, 50, (batch_size, src_len)).astype(np.int32))
tgt = pysml.Tensor(np.random.randint(0, 50, (batch_size, tgt_len)).astype(np.int32))
tgt_labels = np.random.randint(0, 50, (batch_size, tgt_len))

# Forward pass
output = model(src, tgt)
print(f"Source shape: {src.shape}")
print(f"Target shape: {tgt.shape}")
print(f"Output shape: {output.shape}")

# Training
optimizer = AdamW(model.parameters(), lr=0.001)
optimizer.zero_grad()

output_flat = output.view(batch_size * tgt_len, 50)
loss = F.cross_entropy(output_flat, tgt_labels.reshape(-1))
print(f"Loss: {loss.item():.4f}")
loss.backward()
optimizer.step()
print("Seq2Seq training completed\n")


# ==================== Example 5: Multi-layer LSTM ====================
print("="*70)
print("Example 5: Deep LSTM (3 layers) with Dropout")
print("="*70)

# Deep LSTM
lstm = LSTM(input_size=20, hidden_size=64, num_layers=3, 
            batch_first=True, dropout=0.3)

x = pysml.Tensor(np.random.randn(8, 15, 20).astype(np.float32))
output, (h_n, c_n) = lstm(x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Final hidden state shape: {h_n.shape}")
print(f"Final cell state shape: {c_n.shape}")
print(f"Number of parameters: {F.sum_params(lstm):,}")
print("Multi-layer LSTM completed\n")


# ==================== Summary ====================
print("="*70)
print("Summary")
print("="*70)
print("RNN: Good for simple sequence tasks")
print("LSTM: Handles long-term dependencies better")
print("GRU: Faster than LSTM, similar performance")
print("Bidirectional: Processes sequence in both directions")
print("Multi-layer: Increases model capacity")
print("\nAll RNN examples completed successfully!")
print("="*70)