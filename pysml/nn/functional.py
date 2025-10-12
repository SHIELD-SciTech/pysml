import numpy as np
from pysml.tensor import Tensor
from pysml.nn.autograd import CrossEntropyLoss # Import the new loss function

def softmax(t, temp=-1):
    if isinstance(t, Tensor):
        t = t.data
    exps = np.exp(t - np.max(t, axis=temp, keepdims=True))
    return exps / np.sum(exps, axis=temp, keepdims=True)

def relu(t):
    if not isinstance(t, Tensor):
        t = Tensor(t)
    return Tensor(np.maximum(0, t.data))

def sum_params(module):
    total = 0
    for param in module.parameters():
        total += param.data.size
    return total

def cross_entropy(y_hat, y_true): # A standard cross entropy activation.
    # Directly apply the combined, stable CrossEntropyLoss function
    return CrossEntropyLoss.apply(CrossEntropyLoss, y_hat, y_true)