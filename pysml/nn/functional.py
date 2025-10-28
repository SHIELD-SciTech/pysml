"""
PySML Functional — v0.4.8-final
Optimized: fused activations, log-softmax, loss functions built on engine ops.
"""

from .. import engine

# --- Activations ---
relu = engine.relu
sigmoid = engine.sigmoid
tanh = engine.tanh
gelu = engine.gelu
softmax = engine.softmax

def log_softmax(x, axis=-1):
    m = engine.max(x, axis=axis, keepdims=True)
    e = engine.exp(x - m)
    s = engine.sum(e, axis=axis, keepdims=True)
    return x - m - engine.log(s)

# --- Loss functions ---
def mse_loss(pred, target):
    diff = pred - target
    return engine.mean(diff * diff)

def cross_entropy_loss(logits, targets):
    return engine.cross_entropy_loss(logits, targets)

def binary_cross_entropy(pred, target):
    return engine.binary_cross_entropy_loss(pred, target)

# --- Normalization ---
def batch_norm(x, eps=1e-5):
    return engine.batch_norm(x, eps=eps)

def layer_norm(x, eps=1e-5):
    return engine.layer_norm(x, eps=eps)

def dropout(x, p=0.5):
    return engine.dropout(x, p=p)
