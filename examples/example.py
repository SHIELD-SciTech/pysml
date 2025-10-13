import sys, os
sys.path.insert(0,os.getcwd().split("examples")[0])
import pysml


# Example usage of PySML - Runs on CPU by default
t1 = pysml.Tensor([[1, 2], [3, 4]])
t2 = pysml.Tensor([[5, 6], [7, 8]])

result_add = pysml.add(t1, t2)
result_mul = pysml.multiply(t1, t2)
result_matmul = pysml.matmul(t1, t2)

print("Addition Result:\n", result_add)
print("Multiplication Result:\n", result_mul)
print("Matrix Multiplication Result:\n", result_matmul)

# Use different precissions
t_fp16_1 = pysml.Tensor([[1, 2], [3, 4]], dtype=pysml.tensor.dtype.float16)
t_fp16_2 = pysml.Tensor([[1, 2], [3, 4]], dtype=pysml.tensor.dtype.float16)
result_matmul_fp16 = pysml.matmul(t_fp16_1, t_fp16_2)
print("Matrix Multiplication Result with float16:\n", result_matmul_fp16)

# Check CUDA availability and switch backend if available
if pysml.cuda.is_available():
    pysml.cuda.init()
    pysml.operations.update_backend('cuda') # Or pysml.operations.update_backend() if cuda.init() is called, 'cuda' is optional
    print("Current Backend:", pysml.TensorType.backend)  # Check new backend

    # Alternative, use GPU directly in calculations
    t3 = pysml.Tensor([[1, 2], [3, 4]])
    t4 = pysml.Tensor([[5, 6], [7, 8]])
    result_add_gpu = pysml.cuda.add(t3, t4)
    result_mul_gpu = pysml.cuda.multiply(t3, t4)
    result_matmul_gpu = pysml.cuda.matmul(t3, t4)

    print("GPU Addition Result:\n", result_add_gpu)
    print("GPU Multiplication Result:\n", result_mul_gpu)
    print("GPU Matrix Multiplication Result:\n", result_matmul_gpu)

    # Move tensor
    t5 = pysml.Tensor([[1, 2], [3, 4]])
    t5_gpu = t5.to('cuda:0')
    print("Tensor moved to GPU:\n", t5_gpu)
    t5_cpu = t5_gpu.to('cpu')
    print("Tensor moved back to CPU:\n", t5_cpu)

if pysml.xpu.is_available():
    pysml.xpu.init()
    pysml.operations.update_backend('xpu') # Or pysml.operations.update_backend() if xpu.init() is called, 'xpu' is optional
    print("Current Backend:", pysml.TensorType.backend)  # Check new backend

    # Alternative, use XPU directly in calculations
    t6 = pysml.Tensor([[1, 2], [3, 4]])
    t7 = pysml.Tensor([[5, 6], [7, 8]])
    result_add_xpu = pysml.xpu.add(t6, t7)
    result_mul_xpu = pysml.xpu.multiply(t6, t7)
    result_matmul_xpu = pysml.xpu.matmul(t6, t7)

    print("XPU Addition Result:\n", result_add_xpu)
    print("XPU Multiplication Result:\n", result_mul_xpu)
    print("XPU Matrix Multiplication Result:\n", result_matmul_xpu)

    # Move tensor
    t8 = pysml.Tensor([[1, 2], [3, 4]])
    t8_xpu = t8.to('xpu:0')
    print("Tensor moved to XPU:\n", t8_xpu)
    t8_cpu = t8_xpu.to('cpu')
    print("Tensor moved back to CPU:\n", t8_cpu)

# Example of context manager for backend switching
from pysml.backend.context import device
with device('cpu'):
    t9 = pysml.Tensor([[1, 2], [3, 4]]).to('cpu')
    t10 = pysml.Tensor([[5, 6], [7, 8]]).to('cpu')
    result_add_context = pysml.add(t9, t10)
    print("Addition Result in CPU context:\n", result_add_context)

# Example of context manager for backend switching
if pysml.xpu.is_available():
    with device('xpu:0'):
        t11 = pysml.Tensor([[1, 2], [3, 4]]).to('xpu:0')
        t12 = pysml.Tensor([[5, 6], [7, 8]]).to('xpu:0')
        result_mm_context = pysml.mm(t11, t12)
        print("Matrix Multiplication Result in XPU context:\n", result_mm_context)

# Set standard dtype for new tensors
pysml.tensor.STANDARD_DTYPE = pysml.tensor.dtype.float16
t13 = pysml.Tensor([[1, 2], [3, 4]])
print("New Tensor with standard dtype float16:\n", t13)

from pysml.nn import functional as F
logits = pysml.Tensor([1.0, 2.0, 3.0])
probs = F.softmax(logits)
print("Softmax Probabilities:\n", probs)

import numpy as np
import pysml
from pysml.nn.module import Transformer
from pysml.nn.optim import SGD, AdamW
from pysml.nn import functional as F
from pysml.backend.context import device

# Model Hyperparameters
VOCAB_SIZE = 10
D_MODEL = 32     # Embedding dimension
N_HEADS = 4      # Number of attention heads
NUM_LAYERS = 2   # Number of transformer blocks
D_FF = 64        # Feed-forward hidden dimension
SEQ_LENGTH = 8   # (Max) Length of input sequences

# Training Hyperparameters
EPOCHS = 50
LEARNING_RATE = 0.1
BATCH_SIZE = 4
print_mod = 10

# Create Dummy Data
def create_dummy_data(n_samples, seq_len, vocab_size):
    src = np.random.randint(1, vocab_size, size=(n_samples, seq_len))
    tgt = src.copy()
    src_data = pysml.Tensor(src.astype(np.int32))
    tgt_data = pysml.Tensor(tgt.astype(np.int32))
    return src_data, tgt_data

print("Preparing Data")
src_data, tgt_data = create_dummy_data(BATCH_SIZE, SEQ_LENGTH, VOCAB_SIZE)
print("Source Shape:", src_data.shape)
print("Target Shape:", tgt_data.shape)

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
    print(f"Total Parameters: {F.sum_params(model)}")
                                                    # To avoid overfitting
    optimizer = SGD(model.parameters(), lr=LEARNING_RATE) # Or use AdamW: optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    # Training Loop
    print("\nStarting Training on CPU with Full Backpropagation")
    for epoch in range(EPOCHS):
        src_data, tgt_data = create_dummy_data(BATCH_SIZE, SEQ_LENGTH, VOCAB_SIZE)
        model.train()
        optimizer.zero_grad()
        
        output_logits = model(src_data)
        
        output_view = output_logits.view(BATCH_SIZE * SEQ_LENGTH, VOCAB_SIZE)
        target_view = tgt_data.data.reshape(-1)

        loss = F.cross_entropy(output_view, target_view)
        loss.backward()
        
        optimizer.step()

        if (epoch + 1) % print_mod == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}")

    #Inference Example
    print("\nStarting Inference")
    model.eval()

    test_input = pysml.Tensor(np.array([[5, 2, 8, 1, 4, 3, 9, 6]], dtype=np.int32))
    print("Input Sequence: ", test_input.data)

    with device('cpu'):
        prediction_logits = model(test_input)
        predicted_indices = np.argmax(prediction_logits.data, axis=-1)

    print("Predicted Sequence:", predicted_indices)
    print("Target Sequence:   ", test_input.data)
    print(f"Accuracy: {np.mean(predicted_indices == test_input.data)*100:.2f}%")

if pysml.xpu.is_available():
    print("\nInitializing Model on XPU")
    with device('xpu:0'):
        model = Transformer(
            vocab_size=VOCAB_SIZE,
            d_model=D_MODEL,
            num_layers=NUM_LAYERS,
            n_heads=N_HEADS,
            d_ff=D_FF
        )
        print(f"Total Parameters: {F.sum_params(model)}")
                                                        # To avoid overfitting
        optimizer = SGD(model.parameters(), lr=LEARNING_RATE) # Or use AdamW: optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

        # Training Loop
        print("\nStarting Training on XPU with Full Backpropagation")
        for epoch in range(EPOCHS):
            src_data, tgt_data = create_dummy_data(BATCH_SIZE, SEQ_LENGTH, VOCAB_SIZE)
            src_data = src_data.to("xpu:0")
            model.train()
            optimizer.zero_grad()
            
            output_logits = model(src_data)
            
            output_view = output_logits.view(BATCH_SIZE * SEQ_LENGTH, VOCAB_SIZE)
            target_view = tgt_data.data.reshape(-1)

            loss = F.cross_entropy(output_view, target_view)
            loss.backward()
            
            optimizer.step()

            if (epoch + 1) % print_mod == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}")

        #Inference Example
        print("\nStarting Inference")
        model.eval()

        test_input = pysml.Tensor(np.array([[5, 2, 8, 1, 4, 3, 9, 6]], dtype=np.int32))
        print("Input Sequence: ", test_input.data)

        with device('xpu:0'):
            test_input_xpu = test_input.to("xpu:0")
            prediction_logits = model(test_input_xpu)
            predicted_indices = np.argmax(prediction_logits.to("cpu").data, axis=-1)
            test_input_cpu = test_input_xpu.to("cpu").data

        print("Predicted Sequence:", predicted_indices)
        print("Target Sequence:   ", test_input_cpu)
        print(f"Accuracy: {np.mean(predicted_indices == test_input_cpu)*100:.2f}%")

if pysml.cuda.is_available():
    print("\nInitializing Model on Cuda")
    with device('cuda:0'):
        model = Transformer(
            vocab_size=VOCAB_SIZE,
            d_model=D_MODEL,
            num_layers=NUM_LAYERS,
            n_heads=N_HEADS,
            d_ff=D_FF
        )
        print(f"Total Parameters: {F.sum_params(model)}")
                                                        # To avoid overfitting
        optimizer = SGD(model.parameters(), lr=LEARNING_RATE) # Or use AdamW: optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

        # Training Loop
        print("\nStarting Training on XPU with Full Backpropagation")
        for epoch in range(EPOCHS):
            src_data, tgt_data = create_dummy_data(BATCH_SIZE, SEQ_LENGTH, VOCAB_SIZE)
            src_data = src_data.to("cuda:0")
            model.train()
            optimizer.zero_grad()
            
            output_logits = model(src_data)
            
            output_view = output_logits.view(BATCH_SIZE * SEQ_LENGTH, VOCAB_SIZE)
            target_view = tgt_data.data.reshape(-1)

            loss = F.cross_entropy(output_view, target_view)
            loss.backward()
            
            optimizer.step()

            if (epoch + 1) % print_mod == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}")

        #Inference Example
        print("\nStarting Inference")
        model.eval()

        test_input = pysml.Tensor(np.array([[5, 2, 8, 1, 4, 3, 9, 6]], dtype=np.int32))
        print("Input Sequence: ", test_input.data)

        with device('cuda:0'):
            test_input_cuda = test_input.to("cuda:0")
            prediction_logits = model(test_input_cuda)
            predicted_indices = np.argmax(prediction_logits.to("cpu").data, axis=-1)
            test_input_cpu = test_input_cuda.to("cpu").data

        print("Predicted Sequence:", predicted_indices)
        print("Target Sequence:   ", test_input_cpu)
        print(f"Accuracy: {np.mean(predicted_indices == test_input_cpu)*100:.2f}%")



from pysml.nn.optim import AdamW
from pysml.nn.module import Transformer

# Model Hyperparameters
VOCAB_SIZE = 100
D_MODEL = 128
N_HEADS = 16
NUM_LAYERS = 12
D_FF = 512
SEQ_LENGTH = 128

model = Transformer(
    vocab_size=VOCAB_SIZE,
    d_model=D_MODEL,
    num_layers=NUM_LAYERS,
    n_heads=N_HEADS,
    d_ff=D_FF
)
optimizer = AdamW(model.parameters(), lr=0.001)

from pysml.store import save_state_dict, load_state_dict, save_model, load_model, save_checkpoint, load_checkpoint, get_model_size
# Or use pysml.save_state_dict or pysml.save_model, pysml.load_checkpoint, ...

pysml.save_state_dict(model, "model_weights.pysml") # You can use any extention, recommended extentions are .pysml and .pm (PySML Model)
pysml.load_state_dict(model, "model_weights.pysml") # Use the model as model

pysml.save_checkpoint(model, optimizer, "checkpoint.pysml", epoch=50, loss=0.3, metadata={"best_accuracy": 0.95}) # Save checkpoint in training
info = pysml.load_checkpoint(model, optimizer, "checkpoint.pysml") # model is updated, optimizer is updated
start_epoch = info["epoch"] + 1
model.train() # Continue training the model with "model" and "optimizer"

pysml.save_model(model, "model.pysml")
model = pysml.load_model("model.pysml")

info = pysml.get_model_size(model)
print(f"Total parameters: {info['total_params']:,}")
print(f"Memory usage: {info['memory_mb']:.2f} MB")


# CNN example with PySML
import numpy as np
import pysml
from pysml.nn.module import Module, SimpleCNN, AdvancedCNN
from pysml.nn.linear import Linear
from pysml.nn.optim import AdamW
from pysml.nn import functional as F

def create_dummy_data(batch_size, num_classes=10):
    # Random images (batch_size, 1, 28, 28)
    images = np.random.randn(batch_size, 1, 28, 28).astype(np.float32)
    # Random labels
    labels = np.random.randint(0, num_classes, size=(batch_size,))
    
    return pysml.Tensor(images), labels

# Hyperparameters
BATCH_SIZE = 32
NUM_CLASSES = 10
EPOCHS = 5
LEARNING_RATE = 0.001

# Create model
print("\nCreating SimpleCNN model...")
model = SimpleCNN(num_classes=NUM_CLASSES)

# Count parameters
from pysml.nn import functional as F
total_params = F.sum_params(model)
print(f"Total parameters: {total_params:,}")

# Create optimizer
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0001)

# Training loop
print(f"\nTraining for {EPOCHS} epochs...")
print("-"*60)

for epoch in range(EPOCHS):
    model.train()

    images, labels = create_dummy_data(BATCH_SIZE, NUM_CLASSES)
    
    optimizer.zero_grad()
    outputs = model(images)
    
    loss = F.cross_entropy(outputs.view(BATCH_SIZE, NUM_CLASSES), labels)
    loss.backward()
    optimizer.step()
    
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}")

# Evaluation
print("\nEvaluating model...")
model.eval()

test_images, test_labels = create_dummy_data(BATCH_SIZE, NUM_CLASSES)

with pysml.backend.context.device('cpu'):
    predictions = model(test_images)
    predicted_classes = np.argmax(predictions.data, axis=1)
    accuracy = np.mean(predicted_classes == test_labels) * 100

print(f"Test Accuracy: {accuracy:.2f}%")

# Same can be doen with the AdvancedCNN

#Test Datasets
from pysml.data import TensorDataset, DataLoader, train_test_split


data = [pysml.Tensor([[1, 2], [3, 4]]), pysml.Tensor([[5, 6], [7, 8]])] # A list of tensors (containing data)
labels = [pysml.Tensor([[2, 3], [4, 5]]), pysml.Tensor([[6, 7], [8, 9]])] # A list of tensors (containing data)

# Create dataset
dataset = TensorDataset(data, labels)
train_data, test_data = train_test_split(dataset, test_size=0.2)

# Create loaders
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Training loop
epochs = 0 # Select epochs...
for epoch in range(epochs):
    for batch_x, batch_y in train_loader:
        # training code here
        pass

# RNN/LSTM/GRU
print("\nRNN/LSTM/GRU Examples --")

from pysml.nn.rnn import RNN, LSTM, GRU

# Basic RNN
rnn = RNN(input_size=10, hidden_size=20, num_layers=2, batch_first=True)
x_rnn = pysml.Tensor(np.random.randn(4, 5, 10).astype(np.float32))  # (batch, seq, features)
output_rnn, h_n = rnn(x_rnn)
print(f"RNN Output shape: {output_rnn.shape}, Hidden: {h_n.shape}")

# LSTM for sequence prediction
lstm = LSTM(input_size=16, hidden_size=32, num_layers=2, batch_first=True, dropout=0.2)
x_lstm = pysml.Tensor(np.random.randn(8, 10, 16).astype(np.float32))
output_lstm, (h_n, c_n) = lstm(x_lstm)
print(f"LSTM Output shape: {output_lstm.shape}, Hidden: {h_n.shape}, Cell: {c_n.shape}")

# Bidirectional GRU
gru = GRU(input_size=20, hidden_size=64, num_layers=1, batch_first=True, bidirectional=True)
x_gru = pysml.Tensor(np.random.randn(4, 8, 20).astype(np.float32))
output_gru, h_n = gru(x_gru)
print(f"BiGRU Output shape: {output_gru.shape} (note: hidden_size * 2 for bidirectional)")

# Simple text classifier with LSTM
class TextClassifier(Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes):
        super().__init__()
        from pysml.nn.module import Embedding
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.lstm = LSTM(embedding_dim, hidden_size, batch_first=True)
        self.fc = Linear(hidden_size, num_classes)
    
    def forward(self, x):
        embeds = self.embedding(x)
        _, (h_n, _) = self.lstm(embeds)
        return self.fc(h_n[-1])  # Use last layer's hidden state

text_model = TextClassifier(vocab_size=100, embedding_dim=32, hidden_size=64, num_classes=3)
text_input = pysml.Tensor(np.random.randint(0, 100, (8, 15)).astype(np.int32))
text_output = text_model(text_input)
print(f"Text Classifier Output: {text_output.shape}")

print("RNN/LSTM/GRU examples completed!")


### AMP (Mixed Precision Example)
# Add this to the end of your example.py file

# Mixed Precision Training (AMP) 
print("\nmixed Precision Training Example")

from pysml.amp import autocast, GradScaler, AMPContext, clip_grad_norm_

# Method 1: Manual autocast + GradScaler
print("Method 1: Manual AMP")
model = SimpleCNN(num_classes=10)
optimizer = AdamW(model.parameters(), lr=0.001)
scaler = GradScaler()

for epoch in range(3):
    optimizer.zero_grad()
    
    # Forward pass in FP16
    with autocast():
        x = pysml.Tensor(np.random.randn(8, 1, 28, 28).astype(np.float32))
        output = model(x)
        loss = F.cross_entropy(output, np.random.randint(0, 10, 8))
    
    # Backward with gradient scaling
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    clip_grad_norm_(model.parameters(), max_norm=1.0)  # Optional gradient clipping
    scaler.step(optimizer)
    scaler.update()
    
    if (epoch + 1) % 2 == 0:
        print(f"  Epoch {epoch+1}: Loss={loss.item():.4f}, Scale={scaler.get_scale():.0f}")

# Method 2: AMPContext (Simplified)
print("\nMethod 2: AMPContext")
model = SimpleCNN(num_classes=10)
optimizer = AdamW(model.parameters(), lr=0.001)
amp = AMPContext()

for epoch in range(3):
    optimizer.zero_grad()
    
    with amp.autocast():
        x = pysml.Tensor(np.random.randn(8, 1, 28, 28).astype(np.float32))
        output = model(x)
        loss = F.cross_entropy(output, np.random.randint(0, 10, 8))
    
    amp.scale(loss).backward()
    amp.step(optimizer)
    amp.update()

print("Mixed precision training completed!")

# Checkpoint with AMP state
print("\nSaving checkpoint with AMP state...")
checkpoint = {
    'model': model.parameters(),
    'optimizer': optimizer,
    'amp': amp.state_dict()
}
print(f"  Current scale: {amp.scaler.get_scale()}")
print("AMP examples completed!")

