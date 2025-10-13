import numpy as np
from pysml.tensor import Tensor, dtype
from pysml.nn.module import Module
from pysml.nn.linear import Linear
import pysml.operations as ops


class RNNCell(Module):
    def __init__(self, input_size: int, hidden_size: int, bias=True, 
                 nonlinearity='tanh', dtype=dtype.float32):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nonlinearity = nonlinearity
        
        # Input to hidden
        self.W_ih = Linear(input_size, hidden_size, bias=bias, dtype=dtype)
        # Hidden to hidden
        self.W_hh = Linear(hidden_size, hidden_size, bias=bias, dtype=dtype)
    
    def forward(self, x: Tensor, h: Tensor = None) -> Tensor:
        batch_size = x.shape[0]
        
        # Initialize hidden state if not provided
        if h is None:
            np_backend = x._get_backend_module()
            h = Tensor(np_backend.zeros((batch_size, self.hidden_size), dtype=np.float32))
        
        # h_t = activation(W_ih @ x + W_hh @ h)
        h_next = self.W_ih(x) + self.W_hh(h)
        
        # Apply activation
        if self.nonlinearity == 'tanh':
            from pysml.nn.autograd import Tanh
            h_next = Tanh.apply(Tanh, h_next)
        elif self.nonlinearity == 'relu':
            from pysml.nn.autograd import ReLU
            h_next = ReLU.apply(ReLU, h_next)
        else:
            raise ValueError(f"Unknown nonlinearity: {self.nonlinearity}")
        
        return h_next


class RNN(Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers=1,
                 bias=True, batch_first=False, dropout=0.0, bidirectional=False,
                 nonlinearity='tanh', dtype=dtype.float32):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.nonlinearity = nonlinearity
        self.num_directions = 2 if bidirectional else 1
        
        # Create RNN cells for each layer
        self.cells_forward = []
        self.cells_backward = [] if bidirectional else None
        
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size * self.num_directions
            
            # Forward direction
            cell = RNNCell(layer_input_size, hidden_size, bias, nonlinearity, dtype)
            self.cells_forward.append(cell)
            
            # Backward direction
            if bidirectional:
                cell_back = RNNCell(layer_input_size, hidden_size, bias, nonlinearity, dtype)
                self.cells_backward.append(cell_back)
        
        # Dropout layer
        if dropout > 0:
            from pysml.nn.conv import Dropout
            self.dropout_layer = Dropout(dropout)
        else:
            self.dropout_layer = None
    
    def forward(self, x: Tensor, h0: Tensor = None):
        # Handle batch_first
        if self.batch_first:
            x = x.transpose(0, 1)  # (batch, seq, feature) -> (seq, batch, feature)
        
        seq_len, batch_size, _ = x.shape
        np_backend = x._get_backend_module()
        
        # Initialize hidden states
        if h0 is None:
            h0_data = np_backend.zeros(
                (self.num_layers * self.num_directions, batch_size, self.hidden_size),
                dtype=np.float32
            )
            h0 = Tensor(h0_data)
        
        # Process each layer
        layer_output = x
        final_hiddens = []
        
        for layer in range(self.num_layers):
            # Get initial hidden states for this layer
            if self.bidirectional:
                h_forward = h0[layer * 2]
                h_backward = h0[layer * 2 + 1]
            else:
                h_forward = h0[layer]
            
            # Forward pass
            outputs_forward = []
            h = h_forward
            for t in range(seq_len):
                h = self.cells_forward[layer](layer_output[t], h)
                outputs_forward.append(h)
            
            final_hiddens.append(h)
            
            # Backward pass (if bidirectional)
            if self.bidirectional:
                outputs_backward = []
                h = h_backward
                for t in range(seq_len - 1, -1, -1):
                    h = self.cells_backward[layer](layer_output[t], h)
                    outputs_backward.insert(0, h)
                
                final_hiddens.append(h)
                
                # Concatenate forward and backward outputs
                layer_output_data = np_backend.concatenate([
                    np_backend.stack([o.data for o in outputs_forward]),
                    np_backend.stack([o.data for o in outputs_backward])
                ], axis=-1)
                layer_output = Tensor(layer_output_data)
            else:
                layer_output_data = np_backend.stack([o.data for o in outputs_forward])
                layer_output = Tensor(layer_output_data)
            
            # Apply dropout (except for last layer)
            if self.dropout_layer and layer < self.num_layers - 1:
                layer_output = self.dropout_layer(layer_output)
        
        # Stack final hidden states
        h_n_data = np_backend.stack([h.data for h in final_hiddens])
        h_n = Tensor(h_n_data)
        
        # Handle batch_first for output
        if self.batch_first:
            layer_output = layer_output.transpose(0, 1)
        
        return layer_output, h_n


class LSTMCell(Module):
    def __init__(self, input_size: int, hidden_size: int, bias=True, dtype=dtype.float32):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Input to hidden for all gates (i, f, g, o)
        self.W_ii = Linear(input_size, hidden_size, bias=bias, dtype=dtype)
        self.W_if = Linear(input_size, hidden_size, bias=bias, dtype=dtype)
        self.W_ig = Linear(input_size, hidden_size, bias=bias, dtype=dtype)
        self.W_io = Linear(input_size, hidden_size, bias=bias, dtype=dtype)
        
        # Hidden to hidden for all gates
        self.W_hi = Linear(hidden_size, hidden_size, bias=bias, dtype=dtype)
        self.W_hf = Linear(hidden_size, hidden_size, bias=bias, dtype=dtype)
        self.W_hg = Linear(hidden_size, hidden_size, bias=bias, dtype=dtype)
        self.W_ho = Linear(hidden_size, hidden_size, bias=bias, dtype=dtype)
    
    def forward(self, x: Tensor, state=None):
        batch_size = x.shape[0]
        np_backend = x._get_backend_module()
        
        # Initialize hidden and cell states if not provided
        if state is None:
            h = Tensor(np_backend.zeros((batch_size, self.hidden_size), dtype=np.float32))
            c = Tensor(np_backend.zeros((batch_size, self.hidden_size), dtype=np.float32))
        else:
            h, c = state
        
        # Input gate
        i = self.W_ii(x) + self.W_hi(h)
        from pysml.nn.autograd import Sigmoid
        i = Sigmoid.apply(Sigmoid, i)
        
        # Forget gate
        f = self.W_if(x) + self.W_hf(h)
        f = Sigmoid.apply(Sigmoid, f)
        
        # Cell gate
        g = self.W_ig(x) + self.W_hg(h)
        from pysml.nn.autograd import Tanh
        g = Tanh.apply(Tanh, g)
        
        # Output gate
        o = self.W_io(x) + self.W_ho(h)
        o = Sigmoid.apply(Sigmoid, o)
        
        # Cell state update
        c_next = f * c + i * g
        
        # Hidden state update
        h_next = o * Tanh.apply(Tanh, c_next)
        
        return h_next, c_next


class LSTM(Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers=1,
                 bias=True, batch_first=False, dropout=0.0, bidirectional=False,
                 dtype=dtype.float32):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Create LSTM cells for each layer
        self.cells_forward = []
        self.cells_backward = [] if bidirectional else None
        
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size * self.num_directions
            
            # Forward direction
            cell = LSTMCell(layer_input_size, hidden_size, bias, dtype)
            self.cells_forward.append(cell)
            
            # Backward direction
            if bidirectional:
                cell_back = LSTMCell(layer_input_size, hidden_size, bias, dtype)
                self.cells_backward.append(cell_back)
        
        # Dropout layer
        if dropout > 0:
            from pysml.nn.conv import Dropout
            self.dropout_layer = Dropout(dropout)
        else:
            self.dropout_layer = None
    
    def forward(self, x: Tensor, state=None):
        # Handle batch_first
        if self.batch_first:
            x = x.transpose(0, 1)  # (batch, seq, feature) -> (seq, batch, feature)
        
        seq_len, batch_size, _ = x.shape
        np_backend = x._get_backend_module()
        
        # Initialize hidden and cell states
        if state is None:
            h0_data = np_backend.zeros(
                (self.num_layers * self.num_directions, batch_size, self.hidden_size),
                dtype=np.float32
            )
            c0_data = np_backend.zeros(
                (self.num_layers * self.num_directions, batch_size, self.hidden_size),
                dtype=np.float32
            )
            h0 = Tensor(h0_data)
            c0 = Tensor(c0_data)
        else:
            h0, c0 = state
        
        # Process each layer
        layer_output = x
        final_hiddens = []
        final_cells = []
        
        for layer in range(self.num_layers):
            # Get initial states for this layer
            if self.bidirectional:
                h_forward = h0[layer * 2]
                c_forward = c0[layer * 2]
                h_backward = h0[layer * 2 + 1]
                c_backward = c0[layer * 2 + 1]
            else:
                h_forward = h0[layer]
                c_forward = c0[layer]
            
            # Forward pass
            outputs_forward = []
            h, c = h_forward, c_forward
            for t in range(seq_len):
                h, c = self.cells_forward[layer](layer_output[t], (h, c))
                outputs_forward.append(h)
            
            final_hiddens.append(h)
            final_cells.append(c)
            
            # Backward pass (if bidirectional)
            if self.bidirectional:
                outputs_backward = []
                h, c = h_backward, c_backward
                for t in range(seq_len - 1, -1, -1):
                    h, c = self.cells_backward[layer](layer_output[t], (h, c))
                    outputs_backward.insert(0, h)
                
                final_hiddens.append(h)
                final_cells.append(c)
                
                # Concatenate forward and backward outputs
                layer_output_data = np_backend.concatenate([
                    np_backend.stack([o.data for o in outputs_forward]),
                    np_backend.stack([o.data for o in outputs_backward])
                ], axis=-1)
                layer_output = Tensor(layer_output_data)
            else:
                layer_output_data = np_backend.stack([o.data for o in outputs_forward])
                layer_output = Tensor(layer_output_data)
            
            # Apply dropout (except for last layer)
            if self.dropout_layer and layer < self.num_layers - 1:
                layer_output = self.dropout_layer(layer_output)
        
        # Stack final states
        h_n_data = np_backend.stack([h.data for h in final_hiddens])
        c_n_data = np_backend.stack([c.data for c in final_cells])
        h_n = Tensor(h_n_data)
        c_n = Tensor(c_n_data)
        
        # Handle batch_first for output
        if self.batch_first:
            layer_output = layer_output.transpose(0, 1)
        
        return layer_output, (h_n, c_n)


class GRUCell(Module):
    def __init__(self, input_size: int, hidden_size: int, bias=True, dtype=dtype.float32):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Input to hidden for all gates (r, z, n)
        self.W_ir = Linear(input_size, hidden_size, bias=bias, dtype=dtype)
        self.W_iz = Linear(input_size, hidden_size, bias=bias, dtype=dtype)
        self.W_in = Linear(input_size, hidden_size, bias=bias, dtype=dtype)
        
        # Hidden to hidden for all gates
        self.W_hr = Linear(hidden_size, hidden_size, bias=bias, dtype=dtype)
        self.W_hz = Linear(hidden_size, hidden_size, bias=bias, dtype=dtype)
        self.W_hn = Linear(hidden_size, hidden_size, bias=bias, dtype=dtype)
    
    def forward(self, x: Tensor, h: Tensor = None) -> Tensor:
        batch_size = x.shape[0]
        
        # Initialize hidden state if not provided
        if h is None:
            np_backend = x._get_backend_module()
            h = Tensor(np_backend.zeros((batch_size, self.hidden_size), dtype=np.float32))
        
        from pysml.nn.autograd import Sigmoid, Tanh
        
        # Reset gate
        r = self.W_ir(x) + self.W_hr(h)
        r = Sigmoid.apply(Sigmoid, r)
        
        # Update gate
        z = self.W_iz(x) + self.W_hz(h)
        z = Sigmoid.apply(Sigmoid, z)
        
        # New gate
        n = self.W_in(x) + r * self.W_hn(h)
        n = Tanh.apply(Tanh, n)
        
        # Hidden state update
        h_next = (Tensor(1.0) - z) * n + z * h
        
        return h_next


class GRU(Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers=1,
                 bias=True, batch_first=False, dropout=0.0, bidirectional=False,
                 dtype=dtype.float32):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Create GRU cells for each layer
        self.cells_forward = []
        self.cells_backward = [] if bidirectional else None
        
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size * self.num_directions
            
            # Forward direction
            cell = GRUCell(layer_input_size, hidden_size, bias, dtype)
            self.cells_forward.append(cell)
            
            # Backward direction
            if bidirectional:
                cell_back = GRUCell(layer_input_size, hidden_size, bias, dtype)
                self.cells_backward.append(cell_back)
        
        # Dropout layer
        if dropout > 0:
            from pysml.nn.conv import Dropout
            self.dropout_layer = Dropout(dropout)
        else:
            self.dropout_layer = None
    
    def forward(self, x: Tensor, h0: Tensor = None):
        # Handle batch_first
        if self.batch_first:
            x = x.transpose(0, 1)  # (batch, seq, feature) -> (seq, batch, feature)
        
        seq_len, batch_size, _ = x.shape
        np_backend = x._get_backend_module()
        
        # Initialize hidden states
        if h0 is None:
            h0_data = np_backend.zeros(
                (self.num_layers * self.num_directions, batch_size, self.hidden_size),
                dtype=np.float32
            )
            h0 = Tensor(h0_data)
        
        # Process each layer
        layer_output = x
        final_hiddens = []
        
        for layer in range(self.num_layers):
            # Get initial hidden states for this layer
            if self.bidirectional:
                h_forward = h0[layer * 2]
                h_backward = h0[layer * 2 + 1]
            else:
                h_forward = h0[layer]
            
            # Forward pass
            outputs_forward = []
            h = h_forward
            for t in range(seq_len):
                h = self.cells_forward[layer](layer_output[t], h)
                outputs_forward.append(h)
            
            final_hiddens.append(h)
            
            # Backward pass (if bidirectional)
            if self.bidirectional:
                outputs_backward = []
                h = h_backward
                for t in range(seq_len - 1, -1, -1):
                    h = self.cells_backward[layer](layer_output[t], h)
                    outputs_backward.insert(0, h)
                
                final_hiddens.append(h)
                
                # Concatenate forward and backward outputs
                layer_output_data = np_backend.concatenate([
                    np_backend.stack([o.data for o in outputs_forward]),
                    np_backend.stack([o.data for o in outputs_backward])
                ], axis=-1)
                layer_output = Tensor(layer_output_data)
            else:
                layer_output_data = np_backend.stack([o.data for o in outputs_forward])
                layer_output = Tensor(layer_output_data)
            
            # Apply dropout (except for last layer)
            if self.dropout_layer and layer < self.num_layers - 1:
                layer_output = self.dropout_layer(layer_output)
        
        # Stack final hidden states
        h_n_data = np_backend.stack([h.data for h in final_hiddens])
        h_n = Tensor(h_n_data)
        
        # Handle batch_first for output
        if self.batch_first:
            layer_output = layer_output.transpose(0, 1)
        
        return layer_output, h_n
    
