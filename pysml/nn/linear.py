import math

from .module import Module, Parameter


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(self._initialize_weight())

        if bias:
            self.bias = Parameter(self._initialize_bias())
        else:
            self.bias = None

    def _initialize_weight(self):
        from .. import Tensor
        import numpy as np

        limit = math.sqrt(1.0 / self.in_features)
        data = np.random.uniform(-limit, limit, (self.out_features, self.in_features))
        return Tensor(data, requires_grad=True)

    def _initialize_bias(self):
        from .. import Tensor
        import numpy as np

        limit = math.sqrt(1.0 / self.in_features)
        data = np.random.uniform(-limit, limit, (self.out_features,))
        return Tensor(data, requires_grad=True)

    def forward(self, x):
        from .. import engine

        input_shape = x.shape
        if len(input_shape) == 1:
            x = x.reshape((1, -1))
            single_sample = True
        else:
            single_sample = False

        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"Input dimension mismatch: expected {self.in_features}, got {x.shape[-1]}"
            )

        original_shape = x.shape
        if len(original_shape) > 2:
            batch_size = 1
            for dim in original_shape[:-1]:
                batch_size *= dim
            x = x.reshape((batch_size, self.in_features))

        output = engine.matmul(x, self.weight.data.T())

        if self.bias is not None:
            output = engine.add(output, self.bias.data)

        if len(original_shape) > 2:
            output_shape = list(original_shape[:-1]) + [self.out_features]
            output = output.reshape(tuple(output_shape))

        if single_sample:
            output = output.reshape((self.out_features,))

        return output

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


class Bilinear(Module):
    def __init__(self, in1_features, in2_features, out_features, bias=True):
        super().__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features

        self.weight = Parameter(self._initialize_weight())

        if bias:
            self.bias = Parameter(self._initialize_bias())
        else:
            self.bias = None

    def _initialize_weight(self):
        from .. import Tensor
        import numpy as np

        limit = math.sqrt(6.0 / (self.in1_features + self.in2_features + self.out_features))
        data = np.random.uniform(
            -limit,
            limit,
            (self.out_features, self.in1_features, self.in2_features),
        )
        return Tensor(data, requires_grad=True)

    def _initialize_bias(self):
        from .. import Tensor
        import numpy as np

        data = np.zeros((self.out_features,))
        return Tensor(data, requires_grad=True)

    def forward(self, x1, x2):
        from .. import engine

        batch_size = x1.shape[0]

        if x1.shape[-1] != self.in1_features:
            raise ValueError(
                f"x1 dimension mismatch: expected {self.in1_features}, got {x1.shape[-1]}"
            )
        if x2.shape[-1] != self.in2_features:
            raise ValueError(
                f"x2 dimension mismatch: expected {self.in2_features}, got {x2.shape[-1]}"
            )

        backend = x1._backend
        x1_expanded = backend.expand_dims(x1.data, axis=-1)
        x2_expanded = backend.expand_dims(x2.data, axis=1)
        outer = backend.multiply(x1_expanded, x2_expanded)

        from .. import Tensor

        outer_tensor = Tensor.__new__(Tensor)
        outer_tensor._backend = backend
        outer_tensor._dtype = x1._dtype
        outer_tensor.device = x1.device
        outer_tensor.active_device = x1.active_device
        outer_tensor.data = backend.reshape(outer, (batch_size, -1))
        outer_tensor._requires_grad = False
        outer_tensor._grad = None

        weight_flat = self.weight.data.reshape((self.out_features, -1))
        output = engine.matmul(outer_tensor, weight_flat.T())

        if self.bias is not None:
            output = engine.add(output, self.bias.data)

        return output

    def extra_repr(self):
        return (
            f"in1_features={self.in1_features}, in2_features={self.in2_features}, "
            f"out_features={self.out_features}, bias={self.bias is not None}"
        )


class LazyLinear(Module):
    def __init__(self, out_features, bias=True):
        super().__init__()
        self.out_features = out_features
        self.in_features = None
        self._bias_enabled = bias

        self.weight = None
        self.bias = None

    def _initialize_parameters(self, in_features):
        from .. import Tensor
        import numpy as np

        self.in_features = in_features

        limit = math.sqrt(1.0 / in_features)
        weight_data = np.random.uniform(-limit, limit, (self.out_features, in_features))
        self.weight = Parameter(Tensor(weight_data, requires_grad=True))

        if self._bias_enabled:
            bias_data = np.random.uniform(-limit, limit, (self.out_features,))
            self.bias = Parameter(Tensor(bias_data, requires_grad=True))
        else:
            self.bias = None

    def forward(self, x):
        from .. import engine

        if self.weight is None:
            self._initialize_parameters(x.shape[-1])

        input_shape = x.shape
        if len(input_shape) == 1:
            x = x.reshape((1, -1))
            single_sample = True
        else:
            single_sample = False

        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"Input dimension changed: expected {self.in_features}, got {x.shape[-1]}"
            )

        original_shape = x.shape
        if len(original_shape) > 2:
            batch_size = 1
            for dim in original_shape[:-1]:
                batch_size *= dim
            x = x.reshape((batch_size, self.in_features))

        output = engine.matmul(x, self.weight.data.T())

        if self.bias is not None:
            output = engine.add(output, self.bias.data)

        if len(original_shape) > 2:
            output_shape = list(original_shape[:-1]) + [self.out_features]
            output = output.reshape(tuple(output_shape))

        if single_sample:
            output = output.reshape((self.out_features,))

        return output

    def extra_repr(self):
        in_features_str = self.in_features if self.in_features is not None else 'uninitialized'
        return (
            f"in_features={in_features_str}, out_features={self.out_features}, "
            f"bias={self._bias_enabled}"
        )


def init_xavier_uniform_(tensor, gain=1.0):
    import numpy as np

    fan_in, fan_out = tensor.shape[-2], tensor.shape[-1]
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    limit = math.sqrt(3.0) * std

    data = np.random.uniform(-limit, limit, tensor.shape)
    tensor.data.data = tensor.data._backend.asarray(data)


def init_xavier_normal_(tensor, gain=1.0):
    import numpy as np

    fan_in, fan_out = tensor.shape[-2], tensor.shape[-1]
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))

    data = np.random.normal(0, std, tensor.shape)
    tensor.data.data = tensor.data._backend.asarray(data)


def init_kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    import numpy as np

    fan = _calculate_fan(tensor.shape, mode)
    gain = _calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    limit = math.sqrt(3.0) * std

    data = np.random.uniform(-limit, limit, tensor.shape)
    tensor.data.data = tensor.data._backend.asarray(data)


def init_kaiming_normal_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    import numpy as np

    fan = _calculate_fan(tensor.shape, mode)
    gain = _calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)

    data = np.random.normal(0, std, tensor.shape)
    tensor.data.data = tensor.data._backend.asarray(data)


def _calculate_fan(shape, mode='fan_in'):
    if len(shape) < 2:
        raise ValueError("Shape must have at least 2 dimensions")

    fan_in = shape[-2]
    fan_out = shape[-1]

    if mode == 'fan_in':
        return fan_in
    if mode == 'fan_out':
        return fan_out
    if mode == 'fan_avg':
        return (fan_in + fan_out) / 2.0
    raise ValueError(f"Invalid mode: {mode}")


def _calculate_gain(nonlinearity, param=None):
    gains = {
        'linear': 1.0,
        'sigmoid': 1.0,
        'tanh': 5.0 / 3.0,
        'relu': math.sqrt(2.0),
        'leaky_relu': math.sqrt(2.0 / (1 + (param or 0.01) ** 2)),
        'selu': 3.0 / 4.0,
    }

    return gains.get(nonlinearity, 1.0)


__all__ = [
    'Linear',
    'Bilinear',
    'LazyLinear',
    'init_xavier_uniform_',
    'init_xavier_normal_',
    'init_kaiming_uniform_',
    'init_kaiming_normal_',
]
