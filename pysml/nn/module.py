import numpy as np
from pysml.tensor import Tensor, dtype, TensorType
from typing import List
import pysml.operations as ops
from pysml.nn.linear import Linear
import pysml.nn.functional as F
from pysml.nn.autograd import GetItem

### Transformer guide: https://medium.com/@amanatulla1606/transformer-architecture-explained-2c49e2257b4c


class Module:
    # Base class for neural network modules in PySML.
    def __init__(self):
        self.training = True

    def parameters(self):
        visited = set()
        return self._get_parameters(self, visited)

    def _get_parameters(self, obj, visited):
        params = []
        obj_id = id(obj)
        if obj_id in visited:
            return params
        visited.add(obj_id)

        if isinstance(obj, Tensor) and getattr(obj, "requires_grad", False):
            params.append(obj)
        elif isinstance(obj, Module):
            for attr_name in dir(obj):
                if not attr_name.startswith("_"):
                    try:
                        attr = getattr(obj, attr_name)
                        params.extend(self._get_parameters(attr, visited))
                    except AttributeError:
                        continue
        elif isinstance(obj, (list, tuple, set)):
            for item in obj:
                params.extend(self._get_parameters(item, visited))
        elif isinstance(obj, dict):
            for value in obj.values():
                params.extend(self._get_parameters(value, visited))
        
        return params

    def train(self, mode=True):
        self.training = mode
        for attr_name in dir(self):
            obj = getattr(self, attr_name)
            if isinstance(obj, Module):
                obj.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def to(self, device: str):
        TensorType.set_device(device)
        for attr_name in dir(self):
            obj = getattr(self, attr_name)
            if isinstance(obj, Tensor):
                setattr(self, attr_name, obj.to(device))
            elif isinstance(obj, Module):
                obj.to(device)
        return self

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Subclasses of Module must implement forward()")

    def __repr__(self):
        submodules = [
            f"  ({name}): {obj.__repr__().replace(chr(10), chr(10) + '  ')}"
            for name, obj in self.__dict__.items()
            if isinstance(obj, Module)
        ]
        sub_str = "\n".join(submodules)
        return f"{self.__class__.__name__}(\n{sub_str}\n)" if submodules else self.__class__.__name__

class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.weight = Tensor(ops.randn(num_embeddings, embedding_dim).data * 0.02, requires_grad=True)

    def forward(self, x):
        import numpy as np
        index_data = x.data
        if not np.issubdtype(index_data.dtype, np.integer):
            index_data = index_data.astype(np.int64)
        
        # Convert index_data to same backend as weight if needed
        if hasattr(self.weight.data, '__sycl_usm_array_interface__'):
            # Weight is on XPU, ensure indices are too
            import dpnp
            if not hasattr(index_data, '__sycl_usm_array_interface__'):
                # Convert numpy array to dpnp with same queue as weight
                index_data = dpnp.array(index_data, sycl_queue=self.weight.data.sycl_queue)

        # Use the autograd-aware GetItem function
        return GetItem.apply(GetItem, self.weight, index_data)

class LayerNorm(Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.gamma = Tensor((ops.zeros(features).data + 1.), requires_grad=True)
        self.beta = Tensor(ops.zeros(features).data, requires_grad=True)
        self.eps = eps

    def forward(self, x):
        mean = x.mean(axis=-1, keepdims=True)
        # Calculate variance using autograd-awrae operations
        var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
        std = (var + self.eps) ** 0.5
        return self.gamma * (x - mean) / std + self.beta
        
class FeedForward(Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = Linear(d_model, d_ff)
        self.linear2 = Linear(d_ff, d_model)

    def forward(self, x):
        h = self.linear1(x)
        # ReLU using autograd-aware operations
        from pysml.nn.autograd import ReLU
        h = ReLU.apply(ReLU, h)
        return self.linear2(h)

class TransformerBlock(Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.attention = MultiHeadSelfAttention(d_model, n_heads)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff)

    def forward(self, x, mask=None):
        attn_output = self.attention(x, mask)
        x = self.norm1(x + attn_output)
        ff_output = self.ff(x)
        x = self.norm2(x + ff_output)
        return x

class TransformerEncoder(Module):
    def __init__(self, num_layers, d_model, n_heads, d_ff):
        super().__init__()
        self.layers = [TransformerBlock(d_model, n_heads, d_ff) for _ in range(num_layers)]

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x

class Transformer(Module):
    def __init__(self, vocab_size, d_model, num_layers, n_heads, d_ff):
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder = TransformerEncoder(num_layers, d_model, n_heads, d_ff)
        self.fc_out = Linear(d_model, vocab_size)

    def forward(self, src, src_mask=None):
        src = self.embedding(src)
        src = self.pos_encoder(src)
        output = self.encoder(src, src_mask)
        output = self.fc_out(output)
        return output
    

from pysml.nn.attention import MultiHeadSelfAttention, PositionalEncoding
