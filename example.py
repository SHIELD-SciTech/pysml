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

    # --- Training Loop ---
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

        # --- Training Loop ---
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

        # --- Training Loop ---
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

