"""
PySML Neural Network Module
PyTorch-like neural network API
"""

# Import functional API
from . import functional as F

# Import core modules
from .module import Module, Sequential, ModuleList, ModuleDict

# Import layers
from .linear import (
    Linear, 
    Bilinear, 
    Identity, 
    Flatten, 
    Unflatten, 
    Embedding, 
    LazyLinear
)

# Import activations
from .activations import (
    ReLU,
    LeakyReLU,
    GELU,
    Sigmoid,
    Tanh,
    Softmax,
    LogSoftmax,
    ELU,
    Softplus,
    Mish,
    Swish,
    SiLU,
    Hardswish,
    PReLU
)

# Import optimizers
from .optim import (
    Optimizer,
    SGD,
    Adam,
    AdamW,
    RMSprop,
    Adagrad,
    Adadelta,
    LBFGS,
    get_optimizer,
    # Learning rate schedulers
    LRScheduler,
    StepLR,
    ExponentialLR,
    CosineAnnealingLR,
    ReduceLROnPlateau
)

# Import preset models (optional, may not exist yet)
try:
    from .models import (
        TransformerLM,
        TransformerConfig,
        MultiHeadAttention,
        Classifier,
        ClassifierConfig,
        VAE,
        VAEConfig,
        GAN,
        GANConfig,
        SimpleLSTM,
        RNNConfig,
        AutoEncoder,
        AutoEncoderConfig,
        list_presets
    )
    _has_models = True
except ImportError:
    _has_models = False


__all__ = [
    # Functional API
    'F',
    'functional',
    
    # Core modules
    'Module',
    'Sequential',
    'ModuleList',
    'ModuleDict',
    
    # Layers
    'Linear',
    'Bilinear',
    'Identity',
    'Flatten',
    'Unflatten',
    'Embedding',
    'LazyLinear',
    
    # Activations
    'ReLU',
    'LeakyReLU',
    'GELU',
    'Sigmoid',
    'Tanh',
    'Softmax',
    'LogSoftmax',
    'ELU',
    'Softplus',
    'Mish',
    'Swish',
    'SiLU',
    'Hardswish',
    'PReLU',
    
    # Optimizers
    'Optimizer',
    'SGD',
    'Adam',
    'AdamW',
    'RMSprop',
    'Adagrad',
    'Adadelta',
    'LBFGS',
    'get_optimizer',
    
    # Learning rate schedulers
    'LRScheduler',
    'StepLR',
    'ExponentialLR',
    'CosineAnnealingLR',
    'ReduceLROnPlateau',
    
    # Preset models
    'TransformerLM',
    'TransformerConfig',
    'MultiHeadAttention',
    'Classifier',
    'ClassifierConfig',
    'VAE',
    'VAEConfig',
    'GAN',
    'GANConfig',
    'SimpleLSTM',
    'RNNConfig',
    'AutoEncoder',
    'AutoEncoderConfig',
    'list_presets',
    
    # Autograd
    'Function',
    'CrossEntropyLoss',
]


# Version info
__version__ = '0.1.0'


# Convenience functions
def count_parameters(module):
    """Count total trainable parameters in a module"""
    return sum(p.size for p in module.parameters() if p.requires_grad)


def freeze(module):
    """Freeze all parameters in a module"""
    for param in module.parameters():
        param.requires_grad = False
    return module


def unfreeze(module):
    """Unfreeze all parameters in a module"""
    for param in module.parameters():
        param.requires_grad = True
    return module


def summary(module, input_shape=None):
    """
    Print a summary of the model architecture
    
    Args:
        module: Neural network module
        input_shape: Optional input shape for forward pass
    """
    print("=" * 80)
    print(f"Model: {module.__class__.__name__}")
    print("=" * 80)
    
    # Count parameters
    total_params = 0
    trainable_params = 0
    
    print("\nParameters:")
    print("-" * 80)
    
    for name, param in module.named_parameters():
        param_count = param.size
        total_params += param_count
        
        if param.requires_grad:
            trainable_params += param_count
            status = "trainable"
        else:
            status = "frozen"
        
        print(f"  {name:40s} {str(param.shape):20s} {param_count:>10,} {status}")
    
    print("-" * 80)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print("=" * 80)
    
    # Print module structure
    print("\nModule Structure:")
    print("-" * 80)
    print(module)
    print("=" * 80)


__all__.extend(['count_parameters', 'freeze', 'unfreeze', 'summary'])