from .module import Module, Parameter, ModuleList
from .linear import Linear
from .dropout import Dropout
import math


class RNNCell(Module):
	
	def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh'):
		super().__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.nonlinearity = nonlinearity
		
		# Input to hidden
		self.weight_ih = Parameter(self._initialize_weight((hidden_size, input_size)))
		
		# Hidden to hidden
		self.weight_hh = Parameter(self._initialize_weight((hidden_size, hidden_size)))
		
		if bias:
			self.bias_ih = Parameter(self._initialize_bias(hidden_size))
			self.bias_hh = Parameter(self._initialize_bias(hidden_size))
		else:
			self.bias_ih = None
			self.bias_hh = None
	
	def _initialize_weight(self, shape):
		from .. import Tensor
		import numpy as np
		
		# Xavier initialization
		limit = math.sqrt(1.0 / self.hidden_size)
		data = np.random.uniform(-limit, limit, shape)
		return Tensor(data, requires_grad=True)
	
	def _initialize_bias(self, size):
		from .. import Tensor
		import numpy as np
		
		limit = math.sqrt(1.0 / self.hidden_size)
		data = np.random.uniform(-limit, limit, (size,))
		return Tensor(data, requires_grad=True)
	
	def forward(self, x, h=None):
		# x: (batch, input_size)
		# h: (batch, hidden_size) or None
		
		from .. import engine
		
		batch_size = x.shape[0]
		
		# Initialize hidden state if not provided
		if h is None:
			backend = x._backend
			h_data = backend.zeros((batch_size, self.hidden_size))
			from .. import Tensor
			h = Tensor.__new__(Tensor)
			h._backend = backend
			h._dtype = x._dtype
			h.device = x.device
			h.active_device = x.active_device
			h.data = h_data
			h._requires_grad = False
			h._grad = None
		
		# h_new = activation(W_ih @ x + b_ih + W_hh @ h + b_hh)
		linear_ih = engine.matmul(x, self.weight_ih.data.T())
		if self.bias_ih is not None:
			linear_ih = engine.add(linear_ih, self.bias_ih.data)
		
		linear_hh = engine.matmul(h, self.weight_hh.data.T())
		if self.bias_hh is not None:
			linear_hh = engine.add(linear_hh, self.bias_hh.data)
		
		h_new = engine.add(linear_ih, linear_hh)
		
		# Apply nonlinearity
		if self.nonlinearity == 'tanh':
			h_new = engine.tanh(h_new)
		elif self.nonlinearity == 'relu':
			h_new = engine.relu(h_new)
		else:
			raise ValueError(f"Unknown nonlinearity: {self.nonlinearity}")
		
		return h_new
	
	def extra_repr(self):
		return (f"input_size={self.input_size}, hidden_size={self.hidden_size}, "
				f"nonlinearity='{self.nonlinearity}'")


class LSTMCell(Module):
	
	def __init__(self, input_size, hidden_size, bias=True):
		super().__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		
		# Combined input to hidden (for all 4 gates)
		# i, f, g, o gates
		self.weight_ih = Parameter(self._initialize_weight((4 * hidden_size, input_size)))
		
		# Combined hidden to hidden (for all 4 gates)
		self.weight_hh = Parameter(self._initialize_weight((4 * hidden_size, hidden_size)))
		
		if bias:
			self.bias_ih = Parameter(self._initialize_bias(4 * hidden_size))
			self.bias_hh = Parameter(self._initialize_bias(4 * hidden_size))
		else:
			self.bias_ih = None
			self.bias_hh = None
	
	def _initialize_weight(self, shape):
		from .. import Tensor
		import numpy as np
		
		limit = math.sqrt(1.0 / self.hidden_size)
		data = np.random.uniform(-limit, limit, shape)
		return Tensor(data, requires_grad=True)
	
	def _initialize_bias(self, size):
		from .. import Tensor
		import numpy as np
		
		limit = math.sqrt(1.0 / self.hidden_size)
		data = np.random.uniform(-limit, limit, (size,))
		return Tensor(data, requires_grad=True)
	
	def forward(self, x, state=None):
		# x: (batch, input_size)
		# state: (h, c) where h, c: (batch, hidden_size)
		
		from .. import engine
		
		batch_size = x.shape[0]
		
		# Initialize state if not provided
		if state is None:
			backend = x._backend
			h_data = backend.zeros((batch_size, self.hidden_size))
			c_data = backend.zeros((batch_size, self.hidden_size))
			
			from .. import Tensor
			h = Tensor.__new__(Tensor)
			h._backend = backend
			h._dtype = x._dtype
			h.device = x.device
			h.active_device = x.active_device
			h.data = h_data
			h._requires_grad = False
			h._grad = None
			
			c = Tensor.__new__(Tensor)
			c._backend = backend
			c._dtype = x._dtype
			c.device = x.device
			c.active_device = x.active_device
			c.data = c_data
			c._requires_grad = False
			c._grad = None
			
			state = (h, c)
		
		h, c = state
		
		# Compute all gates at once
		gates = engine.matmul(x, self.weight_ih.data.T())
		if self.bias_ih is not None:
			gates = engine.add(gates, self.bias_ih.data)
		
		gates_h = engine.matmul(h, self.weight_hh.data.T())
		if self.bias_hh is not None:
			gates_h = engine.add(gates_h, self.bias_hh.data)
		
		gates = engine.add(gates, gates_h)
		
		# Split into 4 gates: input, forget, cell, output
		backend = gates._backend
		gate_data = gates.data
		
		# Split along feature dimension
		i_gate_data = gate_data[:, :self.hidden_size]
		f_gate_data = gate_data[:, self.hidden_size:2*self.hidden_size]
		g_gate_data = gate_data[:, 2*self.hidden_size:3*self.hidden_size]
		o_gate_data = gate_data[:, 3*self.hidden_size:]
		
		# Create tensors for each gate
		from .. import Tensor
		
		i_gate = Tensor.__new__(Tensor)
		i_gate._backend = backend
		i_gate._dtype = gates._dtype
		i_gate.device = gates.device
		i_gate.active_device = gates.active_device
		i_gate.data = i_gate_data
		i_gate._requires_grad = gates._requires_grad
		i_gate._grad = None
		
		f_gate = Tensor.__new__(Tensor)
		f_gate._backend = backend
		f_gate._dtype = gates._dtype
		f_gate.device = gates.device
		f_gate.active_device = gates.active_device
		f_gate.data = f_gate_data
		f_gate._requires_grad = gates._requires_grad
		f_gate._grad = None
		
		g_gate = Tensor.__new__(Tensor)
		g_gate._backend = backend
		g_gate._dtype = gates._dtype
		g_gate.device = gates.device
		g_gate.active_device = gates.active_device
		g_gate.data = g_gate_data
		g_gate._requires_grad = gates._requires_grad
		g_gate._grad = None
		
		o_gate = Tensor.__new__(Tensor)
		o_gate._backend = backend
		o_gate._dtype = gates._dtype
		o_gate.device = gates.device
		o_gate.active_device = gates.active_device
		o_gate.data = o_gate_data
		o_gate._requires_grad = gates._requires_grad
		o_gate._grad = None
		
		# Apply activations
		i = engine.sigmoid(i_gate)  # Input gate
		f = engine.sigmoid(f_gate)  # Forget gate
		g = engine.tanh(g_gate)	 # Cell gate
		o = engine.sigmoid(o_gate)  # Output gate
		
		# Update cell state
		# c_new = f * c + i * g
		c_new = engine.add(
			engine.multiply(f, c),
			engine.multiply(i, g)
		)
		
		# Update hidden state
		# h_new = o * tanh(c_new)
		h_new = engine.multiply(o, engine.tanh(c_new))
		
		return h_new, c_new
	
	def extra_repr(self):
		return f"input_size={self.input_size}, hidden_size={self.hidden_size}"


class GRUCell(Module):
	
	def __init__(self, input_size, hidden_size, bias=True):
		super().__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		
		# Input to hidden for reset and update gates
		self.weight_ih = Parameter(self._initialize_weight((3 * hidden_size, input_size)))
		
		# Hidden to hidden for reset and update gates
		self.weight_hh = Parameter(self._initialize_weight((3 * hidden_size, hidden_size)))
		
		if bias:
			self.bias_ih = Parameter(self._initialize_bias(3 * hidden_size))
			self.bias_hh = Parameter(self._initialize_bias(3 * hidden_size))
		else:
			self.bias_ih = None
			self.bias_hh = None
	
	def _initialize_weight(self, shape):
		from .. import Tensor
		import numpy as np
		
		limit = math.sqrt(1.0 / self.hidden_size)
		data = np.random.uniform(-limit, limit, shape)
		return Tensor(data, requires_grad=True)
	
	def _initialize_bias(self, size):
		from .. import Tensor
		import numpy as np
		
		limit = math.sqrt(1.0 / self.hidden_size)
		data = np.random.uniform(-limit, limit, (size,))
		return Tensor(data, requires_grad=True)
	
	def forward(self, x, h=None):
		# x: (batch, input_size)
		# h: (batch, hidden_size) or None
		
		from .. import engine
		
		batch_size = x.shape[0]
		
		# Initialize hidden state if not provided
		if h is None:
			backend = x._backend
			h_data = backend.zeros((batch_size, self.hidden_size))
			from .. import Tensor
			h = Tensor.__new__(Tensor)
			h._backend = backend
			h._dtype = x._dtype
			h.device = x.device
			h.active_device = x.active_device
			h.data = h_data
			h._requires_grad = False
			h._grad = None
		
		# Compute gates
		gi = engine.matmul(x, self.weight_ih.data.T())
		if self.bias_ih is not None:
			gi = engine.add(gi, self.bias_ih.data)
		
		gh = engine.matmul(h, self.weight_hh.data.T())
		if self.bias_hh is not None:
			gh = engine.add(gh, self.bias_hh.data)
		
		# Split into 3 gates: reset, update, new
		backend = gi._backend
		
		i_r = gi.data[:, :self.hidden_size]
		i_z = gi.data[:, self.hidden_size:2*self.hidden_size]
		i_n = gi.data[:, 2*self.hidden_size:]
		
		h_r = gh.data[:, :self.hidden_size]
		h_z = gh.data[:, self.hidden_size:2*self.hidden_size]
		h_n = gh.data[:, 2*self.hidden_size:]
		
		# Create tensors
		from .. import Tensor
		
		def make_tensor(data):
			t = Tensor.__new__(Tensor)
			t._backend = backend
			t._dtype = gi._dtype
			t.device = gi.device
			t.active_device = gi.active_device
			t.data = data
			t._requires_grad = gi._requires_grad
			t._grad = None
			return t
		
		r_gate = engine.sigmoid(engine.add(make_tensor(i_r), make_tensor(h_r)))  # Reset gate
		z_gate = engine.sigmoid(engine.add(make_tensor(i_z), make_tensor(h_z)))  # Update gate
		
		# New gate: n = tanh(i_n + r * h_n)
		n_gate = engine.tanh(
			engine.add(
				make_tensor(i_n),
				engine.multiply(r_gate, make_tensor(h_n))
			)
		)
		
		# Update hidden state
		# h_new = (1 - z) * n + z * h
		one_minus_z = engine.subtract(1.0, z_gate)
		h_new = engine.add(
			engine.multiply(one_minus_z, n_gate),
			engine.multiply(z_gate, h)
		)
		
		return h_new
	
	def extra_repr(self):
		return f"input_size={self.input_size}, hidden_size={self.hidden_size}"


class RNN(Module):
	
	def __init__(self, input_size, hidden_size, num_layers=1, nonlinearity='tanh',
				 bias=True, batch_first=False, dropout=0.0, bidirectional=False):
		super().__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.nonlinearity = nonlinearity
		self.batch_first = batch_first
		self.dropout_p = dropout
		self.bidirectional = bidirectional
		self.num_directions = 2 if bidirectional else 1
		
		# Create RNN cells for each layer
		self.cells_forward = ModuleList()
		self.cells_backward = ModuleList() if bidirectional else None
		
		for layer in range(num_layers):
			layer_input_size = input_size if layer == 0 else hidden_size * self.num_directions
			self.cells_forward.append(
				RNNCell(layer_input_size, hidden_size, bias, nonlinearity)
			)
			if bidirectional:
				self.cells_backward.append(
					RNNCell(layer_input_size, hidden_size, bias, nonlinearity)
				)
		
		# Dropout between layers
		if dropout > 0 and num_layers > 1:
			self.dropout = Dropout(dropout)
		else:
			self.dropout = None
	
	def forward(self, x, h_0=None):
		# x: (seq_len, batch, input_size) or (batch, seq_len, input_size) if batch_first
		# h_0: (num_layers * num_directions, batch, hidden_size)
		
		if self.batch_first:
			x = x.T()  # (batch, seq, features) -> (seq, batch, features)
		
		seq_len, batch_size, _ = x.shape
		
		# Initialize hidden state if not provided
		if h_0 is None:
			backend = x._backend
			h_0_data = backend.zeros((
				self.num_layers * self.num_directions,
				batch_size,
				self.hidden_size
			))
			from .. import Tensor
			h_0 = Tensor.__new__(Tensor)
			h_0._backend = backend
			h_0._dtype = x._dtype
			h_0.device = x.device
			h_0.active_device = x.active_device
			h_0.data = h_0_data
			h_0._requires_grad = False
			h_0._grad = None
		
		# Process through layers
		layer_input = x
		h_n_list = []
		
		for layer in range(self.num_layers):
			# Get initial hidden states for this layer
			if self.bidirectional:
				h_forward = h_0.data[layer * 2]
				h_backward = h_0.data[layer * 2 + 1]
			else:
				h_forward = h_0.data[layer]
			
			# Forward direction
			forward_outputs = []
			h = self._make_tensor_from_data(h_forward, x)
			
			for t in range(seq_len):
				x_t = self._get_timestep(layer_input, t)
				h = self.cells_forward[layer](x_t, h)
				forward_outputs.append(h)
			
			h_n_list.append(h)
			
			# Backward direction
			if self.bidirectional:
				backward_outputs = []
				h = self._make_tensor_from_data(h_backward, x)
				
				for t in range(seq_len - 1, -1, -1):
					x_t = self._get_timestep(layer_input, t)
					h = self.cells_backward[layer](x_t, h)
					backward_outputs.insert(0, h)
				
				h_n_list.append(h)
				
				# Concatenate forward and backward outputs
				from .. import engine
				outputs = []
				for fwd, bwd in zip(forward_outputs, backward_outputs):
					outputs.append(engine.concatenate([fwd, bwd], axis=1))
				layer_output = self._stack_outputs(outputs)
			else:
				layer_output = self._stack_outputs(forward_outputs)
			
			# Apply dropout between layers
			if self.dropout is not None and layer < self.num_layers - 1:
				layer_output = self.dropout(layer_output)
			
			layer_input = layer_output
		
		output = layer_input
		
		# Stack hidden states
		h_n = self._stack_hidden(h_n_list, x)
		
		if self.batch_first:
			output = output.T()
		
		return output, h_n
	
	def _make_tensor_from_data(self, data, reference):
		from .. import Tensor
		backend = reference._backend
		
		t = Tensor.__new__(Tensor)
		t._backend = backend
		t._dtype = reference._dtype
		t.device = reference.device
		t.active_device = reference.active_device
		t.data = data
		t._requires_grad = False
		t._grad = None
		return t
	
	def _get_timestep(self, x, t):
		# Extract timestep t from x
		backend = x._backend
		x_t_data = x.data[t]
		return self._make_tensor_from_data(x_t_data, x)
	
	def _stack_outputs(self, outputs):
		# Stack list of tensors along time dimension
		from .. import engine
		return engine.stack(outputs, axis=0)
	
	def _stack_hidden(self, h_list, reference):
		# Stack hidden states
		from .. import engine
		return engine.stack(h_list, axis=0)
	
	def extra_repr(self):
		s = f"{self.input_size}, {self.hidden_size}"
		if self.num_layers != 1:
			s += f", num_layers={self.num_layers}"
		if self.nonlinearity != 'tanh':
			s += f", nonlinearity='{self.nonlinearity}'"
		if not self.batch_first:
			s += ", batch_first=False"
		if self.dropout_p != 0:
			s += f", dropout={self.dropout_p}"
		if self.bidirectional:
			s += ", bidirectional=True"
		return s


class LSTM(Module):
	
	def __init__(self, input_size, hidden_size, num_layers=1, bias=True,
				 batch_first=False, dropout=0.0, bidirectional=False):
		super().__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.batch_first = batch_first
		self.dropout_p = dropout
		self.bidirectional = bidirectional
		self.num_directions = 2 if bidirectional else 1
		
		# Create LSTM cells for each layer
		self.cells_forward = ModuleList()
		self.cells_backward = ModuleList() if bidirectional else None
		
		for layer in range(num_layers):
			layer_input_size = input_size if layer == 0 else hidden_size * self.num_directions
			self.cells_forward.append(
				LSTMCell(layer_input_size, hidden_size, bias)
			)
			if bidirectional:
				self.cells_backward.append(
					LSTMCell(layer_input_size, hidden_size, bias)
				)
		
		# Dropout between layers
		if dropout > 0 and num_layers > 1:
			self.dropout = Dropout(dropout)
		else:
			self.dropout = None
	
	def forward(self, x, state=None):
		# x: (seq_len, batch, input_size) or (batch, seq_len, input_size) if batch_first
		# state: (h_0, c_0) where each is (num_layers * num_directions, batch, hidden_size)
		
		if self.batch_first:
			x = x.T()
		
		seq_len, batch_size, _ = x.shape
		
		# Initialize state if not provided
		if state is None:
			backend = x._backend
			shape = (self.num_layers * self.num_directions, batch_size, self.hidden_size)
			
			from .. import Tensor
			h_0 = Tensor.__new__(Tensor)
			h_0._backend = backend
			h_0._dtype = x._dtype
			h_0.device = x.device
			h_0.active_device = x.active_device
			h_0.data = backend.zeros(shape)
			h_0._requires_grad = False
			h_0._grad = None
			
			c_0 = Tensor.__new__(Tensor)
			c_0._backend = backend
			c_0._dtype = x._dtype
			c_0.device = x.device
			c_0.active_device = x.active_device
			c_0.data = backend.zeros(shape)
			c_0._requires_grad = False
			c_0._grad = None
			
			state = (h_0, c_0)
		
		h_0, c_0 = state
		
		# Process through layers
		layer_input = x
		h_n_list = []
		c_n_list = []
		
		for layer in range(self.num_layers):
			# Get initial states for this layer
			if self.bidirectional:
				h_fwd = self._make_tensor_from_data(h_0.data[layer * 2], x)
				c_fwd = self._make_tensor_from_data(c_0.data[layer * 2], x)
				h_bwd = self._make_tensor_from_data(h_0.data[layer * 2 + 1], x)
				c_bwd = self._make_tensor_from_data(c_0.data[layer * 2 + 1], x)
			else:
				h_fwd = self._make_tensor_from_data(h_0.data[layer], x)
				c_fwd = self._make_tensor_from_data(c_0.data[layer], x)
			
			# Forward direction
			forward_outputs = []
			
			for t in range(seq_len):
				x_t = self._get_timestep(layer_input, t)
				h_fwd, c_fwd = self.cells_forward[layer](x_t, (h_fwd, c_fwd))
				forward_outputs.append(h_fwd)
			
			h_n_list.append(h_fwd)
			c_n_list.append(c_fwd)
			
			# Backward direction
			if self.bidirectional:
				backward_outputs = []
				
				for t in range(seq_len - 1, -1, -1):
					x_t = self._get_timestep(layer_input, t)
					h_bwd, c_bwd = self.cells_backward[layer](x_t, (h_bwd, c_bwd))
					backward_outputs.insert(0, h_bwd)
				
				h_n_list.append(h_bwd)
				c_n_list.append(c_bwd)
				
				# Concatenate forward and backward
				from .. import engine
				outputs = []
				for fwd, bwd in zip(forward_outputs, backward_outputs):
					outputs.append(engine.concatenate([fwd, bwd], axis=1))
				layer_output = self._stack_outputs(outputs)
			else:
				layer_output = self._stack_outputs(forward_outputs)
			
			# Apply dropout
			if self.dropout is not None and layer < self.num_layers - 1:
				layer_output = self.dropout(layer_output)
			
			layer_input = layer_output
		
		output = layer_input
		
		# Stack final states
		h_n = self._stack_hidden(h_n_list, x)
		c_n = self._stack_hidden(c_n_list, x)
		
		if self.batch_first:
			output = output.T()
		
		return output, (h_n, c_n)
	
	def _make_tensor_from_data(self, data, reference):
		from .. import Tensor
		backend = reference._backend
		
		t = Tensor.__new__(Tensor)
		t._backend = backend
		t._dtype = reference._dtype
		t.device = reference.device
		t.active_device = reference.active_device
		t.data = data
		t._requires_grad = False
		t._grad = None
		return t
	
	def _get_timestep(self, x, t):
		backend = x._backend
		x_t_data = x.data[t]
		return self._make_tensor_from_data(x_t_data, x)
	
	def _stack_outputs(self, outputs):
		from .. import engine
		return engine.stack(outputs, axis=0)
	
	def _stack_hidden(self, h_list, reference):
		from .. import engine
		return engine.stack(h_list, axis=0)
	
	def extra_repr(self):
		s = f"{self.input_size}, {self.hidden_size}"
		if self.num_layers != 1:
			s += f", num_layers={self.num_layers}"
		if not self.batch_first:
			s += ", batch_first=False"
		if self.dropout_p != 0:
			s += f", dropout={self.dropout_p}"
		if self.bidirectional:
			s += ", bidirectional=True"
		return s


class GRU(Module):
	
	def __init__(self, input_size, hidden_size, num_layers=1, bias=True,
				 batch_first=False, dropout=0.0, bidirectional=False):
		super().__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.batch_first = batch_first
		self.dropout_p = dropout
		self.bidirectional = bidirectional
		self.num_directions = 2 if bidirectional else 1
		
		# Create GRU cells
		self.cells_forward = ModuleList()
		self.cells_backward = ModuleList() if bidirectional else None
		
		for layer in range(num_layers):
			layer_input_size = input_size if layer == 0 else hidden_size * self.num_directions
			self.cells_forward.append(
				GRUCell(layer_input_size, hidden_size, bias)
			)
			if bidirectional:
				self.cells_backward.append(
					GRUCell(layer_input_size, hidden_size, bias)
				)
		
		# Dropout
		if dropout > 0 and num_layers > 1:
			self.dropout = Dropout(dropout)
		else:
			self.dropout = None
	
	def forward(self, x, h_0=None):
		# Same structure as RNN.forward but with GRU cells
		if self.batch_first:
			x = x.T()
		
		seq_len, batch_size, _ = x.shape
		
		if h_0 is None:
			backend = x._backend
			from .. import Tensor
			h_0 = Tensor.__new__(Tensor)
			h_0._backend = backend
			h_0._dtype = x._dtype
			h_0.device = x.device
			h_0.active_device = x.active_device
			h_0.data = backend.zeros((
				self.num_layers * self.num_directions,
				batch_size,
				self.hidden_size
			))
			h_0._requires_grad = False
			h_0._grad = None
		
		layer_input = x
		h_n_list = []
		
		for layer in range(self.num_layers):
			if self.bidirectional:
				h_fwd = self._make_tensor_from_data(h_0.data[layer * 2], x)
				h_bwd = self._make_tensor_from_data(h_0.data[layer * 2 + 1], x)
			else:
				h_fwd = self._make_tensor_from_data(h_0.data[layer], x)
			
			# Forward
			forward_outputs = []
			for t in range(seq_len):
				x_t = self._get_timestep(layer_input, t)
				h_fwd = self.cells_forward[layer](x_t, h_fwd)
				forward_outputs.append(h_fwd)
			
			h_n_list.append(h_fwd)
			
			# Backward
			if self.bidirectional:
				backward_outputs = []
				for t in range(seq_len - 1, -1, -1):
					x_t = self._get_timestep(layer_input, t)
					h_bwd = self.cells_backward[layer](x_t, h_bwd)
					backward_outputs.insert(0, h_bwd)
				
				h_n_list.append(h_bwd)
				
				from .. import engine
				outputs = []
				for fwd, bwd in zip(forward_outputs, backward_outputs):
					outputs.append(engine.concatenate([fwd, bwd], axis=1))
				layer_output = self._stack_outputs(outputs)
			else:
				layer_output = self._stack_outputs(forward_outputs)
			
			if self.dropout is not None and layer < self.num_layers - 1:
				layer_output = self.dropout(layer_output)
			
			layer_input = layer_output
		
		output = layer_input
		h_n = self._stack_hidden(h_n_list, x)
		
		if self.batch_first:
			output = output.T()
		
		return output, h_n
	
	def _make_tensor_from_data(self, data, reference):
		from .. import Tensor
		backend = reference._backend
		
		t = Tensor.__new__(Tensor)
		t._backend = backend
		t._dtype = reference._dtype
		t.device = reference.device
		t.active_device = reference.active_device
		t.data = data
		t._requires_grad = False
		t._grad = None
		return t
	
	def _get_timestep(self, x, t):
		backend = x._backend
		x_t_data = x.data[t]
		return self._make_tensor_from_data(x_t_data, x)
	
	def _stack_outputs(self, outputs):
		from .. import engine
		return engine.stack(outputs, axis=0)
	
	def _stack_hidden(self, h_list, reference):
		from .. import engine
		return engine.stack(h_list, axis=0)
	
	def extra_repr(self):
		s = f"{self.input_size}, {self.hidden_size}"
		if self.num_layers != 1:
			s += f", num_layers={self.num_layers}"
		if not self.batch_first:
			s += ", batch_first=False"
		if self.dropout_p != 0:
			s += f", dropout={self.dropout_p}"
		if self.bidirectional:
			s += ", bidirectional=True"
		return s


__all__ = [
	'RNNCell', 'RNN',
	'LSTMCell', 'LSTM',
	'GRUCell', 'GRU',
]"""
PySML Recurrent Neural Networks
================================

RNN, LSTM, and GRU layers for sequence modeling.

Memory-optimized implementations with:
- Efficient state management
- Bidirectional support
- Multi-layer stacking
- Dropout between layers
- Proper gradient flow

Layers:
- RNNCell, RNN: Basic recurrent unit
- LSTMCell, LSTM: Long Short-Term Memory
- GRUCell, GRU: Gated Recurrent Unit

Author: S.H.I.E.L.D. Research Division
Version: 0.4.9b
"""

from .module import Module, Parameter, ModuleList
from .linear import Linear
from .dropout import Dropout
import math


class RNNCell(Module):
	
	def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh'):
		super().__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.nonlinearity = nonlinearity
		
		# Input to hidden
		self.weight_ih = Parameter(self._initialize_weight((hidden_size, input_size)))
		
		# Hidden to hidden
		self.weight_hh = Parameter(self._initialize_weight((hidden_size, hidden_size)))
		
		if bias:
			self.bias_ih = Parameter(self._initialize_bias(hidden_size))
			self.bias_hh = Parameter(self._initialize_bias(hidden_size))
		else:
			self.bias_ih = None
			self.bias_hh = None
	
	def _initialize_weight(self, shape):
		from .. import Tensor
		import numpy as np
		
		# Xavier initialization
		limit = math.sqrt(1.0 / self.hidden_size)
		data = np.random.uniform(-limit, limit, shape)
		return Tensor(data, requires_grad=True)
	
	def _initialize_bias(self, size):
		from .. import Tensor
		import numpy as np
		
		limit = math.sqrt(1.0 / self.hidden_size)
		data = np.random.uniform(-limit, limit, (size,))
		return Tensor(data, requires_grad=True)
	
	def forward(self, x, h=None):
		# x: (batch, input_size)
		# h: (batch, hidden_size) or None
		
		from .. import engine
		
		batch_size = x.shape[0]
		
		# Initialize hidden state if not provided
		if h is None:
			backend = x._backend
			h_data = backend.zeros((batch_size, self.hidden_size))
			from .. import Tensor
			h = Tensor.__new__(Tensor)
			h._backend = backend
			h._dtype = x._dtype
			h.device = x.device
			h.active_device = x.active_device
			h.data = h_data
			h._requires_grad = False
			h._grad = None
		
		# h_new = activation(W_ih @ x + b_ih + W_hh @ h + b_hh)
		linear_ih = engine.matmul(x, self.weight_ih.data.T())
		if self.bias_ih is not None:
			linear_ih = engine.add(linear_ih, self.bias_ih.data)
		
		linear_hh = engine.matmul(h, self.weight_hh.data.T())
		if self.bias_hh is not None:
			linear_hh = engine.add(linear_hh, self.bias_hh.data)
		
		h_new = engine.add(linear_ih, linear_hh)
		
		# Apply nonlinearity
		if self.nonlinearity == 'tanh':
			h_new = engine.tanh(h_new)
		elif self.nonlinearity == 'relu':
			h_new = engine.relu(h_new)
		else:
			raise ValueError(f"Unknown nonlinearity: {self.nonlinearity}")
		
		return h_new
	
	def extra_repr(self):
		return (f"input_size={self.input_size}, hidden_size={self.hidden_size}, "
				f"nonlinearity='{self.nonlinearity}'")


class LSTMCell(Module):
	
	def __init__(self, input_size, hidden_size, bias=True):
		super().__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		
		# Combined input to hidden (for all 4 gates)
		# i, f, g, o gates
		self.weight_ih = Parameter(self._initialize_weight((4 * hidden_size, input_size)))
		
		# Combined hidden to hidden (for all 4 gates)
		self.weight_hh = Parameter(self._initialize_weight((4 * hidden_size, hidden_size)))
		
		if bias:
			self.bias_ih = Parameter(self._initialize_bias(4 * hidden_size))
			self.bias_hh = Parameter(self._initialize_bias(4 * hidden_size))
		else:
			self.bias_ih = None
			self.bias_hh = None
	
	def _initialize_weight(self, shape):
		from .. import Tensor
		import numpy as np
		
		limit = math.sqrt(1.0 / self.hidden_size)
		data = np.random.uniform(-limit, limit, shape)
		return Tensor(data, requires_grad=True)
	
	def _initialize_bias(self, size):
		from .. import Tensor
		import numpy as np
		
		limit = math.sqrt(1.0 / self.hidden_size)
		data = np.random.uniform(-limit, limit, (size,))
		return Tensor(data, requires_grad=True)
	
	def forward(self, x, state=None):
		# x: (batch, input_size)
		# state: (h, c) where h, c: (batch, hidden_size)
		
		from .. import engine
		
		batch_size = x.shape[0]
		
		# Initialize state if not provided
		if state is None:
			backend = x._backend
			h_data = backend.zeros((batch_size, self.hidden_size))
			c_data = backend.zeros((batch_size, self.hidden_size))
			
			from .. import Tensor
			h = Tensor.__new__(Tensor)
			h._backend = backend
			h._dtype = x._dtype
			h.device = x.device
			h.active_device = x.active_device
			h.data = h_data
			h._requires_grad = False
			h._grad = None
			
			c = Tensor.__new__(Tensor)
			c._backend = backend
			c._dtype = x._dtype
			c.device = x.device
			c.active_device = x.active_device
			c.data = c_data
			c._requires_grad = False
			c._grad = None
			
			state = (h, c)
		
		h, c = state
		
		# Compute all gates at once
		gates = engine.matmul(x, self.weight_ih.data.T())
		if self.bias_ih is not None:
			gates = engine.add(gates, self.bias_ih.data)
		
		gates_h = engine.matmul(h, self.weight_hh.data.T())
		if self.bias_hh is not None:
			gates_h = engine.add(gates_h, self.bias_hh.data)
		
		gates = engine.add(gates, gates_h)
		
		# Split into 4 gates: input, forget, cell, output
		backend = gates._backend
		gate_data = gates.data
		
		# Split along feature dimension
		i_gate_data = gate_data[:, :self.hidden_size]
		f_gate_data = gate_data[:, self.hidden_size:2*self.hidden_size]
		g_gate_data = gate_data[:, 2*self.hidden_size:3*self.hidden_size]
		o_gate_data = gate_data[:, 3*self.hidden_size:]
		
		# Create tensors for each gate
		from .. import Tensor
		
		i_gate = Tensor.__new__(Tensor)
		i_gate._backend = backend
		i_gate._dtype = gates._dtype
		i_gate.device = gates.device
		i_gate.active_device = gates.active_device
		i_gate.data = i_gate_data
		i_gate._requires_grad = gates._requires_grad
		i_gate._grad = None
		
		f_gate = Tensor.__new__(Tensor)
		f_gate._backend = backend
		f_gate._dtype = gates._dtype
		f_gate.device = gates.device
		f_gate.active_device = gates.active_device
		f_gate.data = f_gate_data
		f_gate._requires_grad = gates._requires_grad
		f_gate._grad = None
		
		g_gate = Tensor.__new__(Tensor)
		g_gate._backend = backend
		g_gate._dtype = gates._dtype
		g_gate.device = gates.device
		g_gate.active_device = gates.active_device
		g_gate.data = g_gate_data
		g_gate._requires_grad = gates._requires_grad
		g_gate._grad = None
		
		o_gate = Tensor.__new__(Tensor)
		o_gate._backend = backend
		o_gate._dtype = gates._dtype
		o_gate.device = gates.device
		o_gate.active_device = gates.active_device
		o_gate.data = o_gate_data
		o_gate._requires_grad = gates._requires_grad
		o_gate._grad = None
		
		# Apply activations
		i = engine.sigmoid(i_gate)  # Input gate
		f = engine.sigmoid(f_gate)  # Forget gate
		g = engine.tanh(g_gate)	 # Cell gate
		o = engine.sigmoid(o_gate)  # Output gate
		
		# Update cell state
		# c_new = f * c + i * g
		c_new = engine.add(
			engine.multiply(f, c),
			engine.multiply(i, g)
		)
		
		# Update hidden state
		# h_new = o * tanh(c_new)
		h_new = engine.multiply(o, engine.tanh(c_new))
		
		return h_new, c_new
	
	def extra_repr(self):
		return f"input_size={self.input_size}, hidden_size={self.hidden_size}"


class GRUCell(Module):
	
	def __init__(self, input_size, hidden_size, bias=True):
		super().__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		
		# Input to hidden for reset and update gates
		self.weight_ih = Parameter(self._initialize_weight((3 * hidden_size, input_size)))
		
		# Hidden to hidden for reset and update gates
		self.weight_hh = Parameter(self._initialize_weight((3 * hidden_size, hidden_size)))
		
		if bias:
			self.bias_ih = Parameter(self._initialize_bias(3 * hidden_size))
			self.bias_hh = Parameter(self._initialize_bias(3 * hidden_size))
		else:
			self.bias_ih = None
			self.bias_hh = None
	
	def _initialize_weight(self, shape):
		from .. import Tensor
		import numpy as np
		
		limit = math.sqrt(1.0 / self.hidden_size)
		data = np.random.uniform(-limit, limit, shape)
		return Tensor(data, requires_grad=True)
	
	def _initialize_bias(self, size):
		from .. import Tensor
		import numpy as np
		
		limit = math.sqrt(1.0 / self.hidden_size)
		data = np.random.uniform(-limit, limit, (size,))
		return Tensor(data, requires_grad=True)
	
	def forward(self, x, h=None):
		# x: (batch, input_size)
		# h: (batch, hidden_size) or None
		
		from .. import engine
		
		batch_size = x.shape[0]
		
		# Initialize hidden state if not provided
		if h is None:
			backend = x._backend
			h_data = backend.zeros((batch_size, self.hidden_size))
			from .. import Tensor
			h = Tensor.__new__(Tensor)
			h._backend = backend
			h._dtype = x._dtype
			h.device = x.device
			h.active_device = x.active_device
			h.data = h_data
			h._requires_grad = False
			h._grad = None
		
		# Compute gates
		gi = engine.matmul(x, self.weight_ih.data.T())
		if self.bias_ih is not None:
			gi = engine.add(gi, self.bias_ih.data)
		
		gh = engine.matmul(h, self.weight_hh.data.T())
		if self.bias_hh is not None:
			gh = engine.add(gh, self.bias_hh.data)
		
		# Split into 3 gates: reset, update, new
		backend = gi._backend
		
		i_r = gi.data[:, :self.hidden_size]
		i_z = gi.data[:, self.hidden_size:2*self.hidden_size]
		i_n = gi.data[:, 2*self.hidden_size:]
		
		h_r = gh.data[:, :self.hidden_size]
		h_z = gh.data[:, self.hidden_size:2*self.hidden_size]
		h_n = gh.data[:, 2*self.hidden_size:]
		
		# Create tensors
		from .. import Tensor
		
		def make_tensor(data):
			t = Tensor.__new__(Tensor)
			t._backend = backend
			t._dtype = gi._dtype
			t.device = gi.device
			t.active_device = gi.active_device
			t.data = data
			t._requires_grad = gi._requires_grad
			t._grad = None
			return t
		
		r_gate = engine.sigmoid(engine.add(make_tensor(i_r), make_tensor(h_r)))  # Reset gate
		z_gate = engine.sigmoid(engine.add(make_tensor(i_z), make_tensor(h_z)))  # Update gate
		
		# New gate: n = tanh(i_n + r * h_n)
		n_gate = engine.tanh(
			engine.add(
				make_tensor(i_n),
				engine.multiply(r_gate, make_tensor(h_n))
			)
		)
		
		# Update hidden state
		# h_new = (1 - z) * n + z * h
		one_minus_z = engine.subtract(1.0, z_gate)
		h_new = engine.add(
			engine.multiply(one_minus_z, n_gate),
			engine.multiply(z_gate, h)
		)
		
		return h_new
	
	def extra_repr(self):
		return f"input_size={self.input_size}, hidden_size={self.hidden_size}"


class RNN(Module):
	
	def __init__(self, input_size, hidden_size, num_layers=1, nonlinearity='tanh',
				 bias=True, batch_first=False, dropout=0.0, bidirectional=False):
		super().__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.nonlinearity = nonlinearity
		self.batch_first = batch_first
		self.dropout_p = dropout
		self.bidirectional = bidirectional
		self.num_directions = 2 if bidirectional else 1
		
		# Create RNN cells for each layer
		self.cells_forward = ModuleList()
		self.cells_backward = ModuleList() if bidirectional else None
		
		for layer in range(num_layers):
			layer_input_size = input_size if layer == 0 else hidden_size * self.num_directions
			self.cells_forward.append(
				RNNCell(layer_input_size, hidden_size, bias, nonlinearity)
			)
			if bidirectional:
				self.cells_backward.append(
					RNNCell(layer_input_size, hidden_size, bias, nonlinearity)
				)
		
		# Dropout between layers
		if dropout > 0 and num_layers > 1:
			self.dropout = Dropout(dropout)
		else:
			self.dropout = None
	
	def forward(self, x, h_0=None):
		# x: (seq_len, batch, input_size) or (batch, seq_len, input_size) if batch_first
		# h_0: (num_layers * num_directions, batch, hidden_size)
		
		if self.batch_first:
			x = x.T()  # (batch, seq, features) -> (seq, batch, features)
		
		seq_len, batch_size, _ = x.shape
		
		# Initialize hidden state if not provided
		if h_0 is None:
			backend = x._backend
			h_0_data = backend.zeros((
				self.num_layers * self.num_directions,
				batch_size,
				self.hidden_size
			))
			from .. import Tensor
			h_0 = Tensor.__new__(Tensor)
			h_0._backend = backend
			h_0._dtype = x._dtype
			h_0.device = x.device
			h_0.active_device = x.active_device
			h_0.data = h_0_data
			h_0._requires_grad = False
			h_0._grad = None
		
		# Process through layers
		layer_input = x
		h_n_list = []
		
		for layer in range(self.num_layers):
			# Get initial hidden states for this layer
			if self.bidirectional:
				h_forward = h_0.data[layer * 2]
				h_backward = h_0.data[layer * 2 + 1]
			else:
				h_forward = h_0.data[layer]
			
			# Forward direction
			forward_outputs = []
			h = self._make_tensor_from_data(h_forward, x)
			
			for t in range(seq_len):
				x_t = self._get_timestep(layer_input, t)
				h = self.cells_forward[layer](x_t, h)
				forward_outputs.append(h)
			
			h_n_list.append(h)
			
			# Backward direction
			if self.bidirectional:
				backward_outputs = []
				h = self._make_tensor_from_data(h_backward, x)
				
				for t in range(seq_len - 1, -1, -1):
					x_t = self._get_timestep(layer_input, t)
					h = self.cells_backward[layer](x_t, h)
					backward_outputs.insert(0, h)
				
				h_n_list.append(h)
				
				# Concatenate forward and backward outputs
				from .. import engine
				outputs = []
				for fwd, bwd in zip(forward_outputs, backward_outputs):
					outputs.append(engine.concatenate([fwd, bwd], axis=1))
				layer_output = self._stack_outputs(outputs)
			else:
				layer_output = self._stack_outputs(forward_outputs)
			
			# Apply dropout between layers
			if self.dropout is not None and layer < self.num_layers - 1:
				layer_output = self.dropout(layer_output)
			
			layer_input = layer_output
		
		output = layer_input
		
		# Stack hidden states
		h_n = self._stack_hidden(h_n_list, x)
		
		if self.batch_first:
			output = output.T()
		
		return output, h_n
	
	def _make_tensor_from_data(self, data, reference):
		from .. import Tensor
		backend = reference._backend
		
		t = Tensor.__new__(Tensor)
		t._backend = backend
		t._dtype = reference._dtype
		t.device = reference.device
		t.active_device = reference.active_device
		t.data = data
		t._requires_grad = False
		t._grad = None
		return t
	
	def _get_timestep(self, x, t):
		# Extract timestep t from x
		backend = x._backend
		x_t_data = x.data[t]
		return self._make_tensor_from_data(x_t_data, x)
	
	def _stack_outputs(self, outputs):
		# Stack list of tensors along time dimension
		from .. import engine
		return engine.stack(outputs, axis=0)
	
	def _stack_hidden(self, h_list, reference):
		# Stack hidden states
		from .. import engine
		return engine.stack(h_list, axis=0)
	
	def extra_repr(self):
		s = f"{self.input_size}, {self.hidden_size}"
		if self.num_layers != 1:
			s += f", num_layers={self.num_layers}"
		if self.nonlinearity != 'tanh':
			s += f", nonlinearity='{self.nonlinearity}'"
		if not self.batch_first:
			s += ", batch_first=False"
		if self.dropout_p != 0:
			s += f", dropout={self.dropout_p}"
		if self.bidirectional:
			s += ", bidirectional=True"
		return s


class LSTM(Module):
	
	def __init__(self, input_size, hidden_size, num_layers=1, bias=True,
				 batch_first=False, dropout=0.0, bidirectional=False):
		super().__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.batch_first = batch_first
		self.dropout_p = dropout
		self.bidirectional = bidirectional
		self.num_directions = 2 if bidirectional else 1
		
		# Create LSTM cells for each layer
		self.cells_forward = ModuleList()
		self.cells_backward = ModuleList() if bidirectional else None
		
		for layer in range(num_layers):
			layer_input_size = input_size if layer == 0 else hidden_size * self.num_directions
			self.cells_forward.append(
				LSTMCell(layer_input_size, hidden_size, bias)
			)
			if bidirectional:
				self.cells_backward.append(
					LSTMCell(layer_input_size, hidden_size, bias)
				)
		
		# Dropout between layers
		if dropout > 0 and num_layers > 1:
			self.dropout = Dropout(dropout)
		else:
			self.dropout = None
	
	def forward(self, x, state=None):
		# x: (seq_len, batch, input_size) or (batch, seq_len, input_size) if batch_first
		# state: (h_0, c_0) where each is (num_layers * num_directions, batch, hidden_size)
		
		if self.batch_first:
			x = x.T()
		
		seq_len, batch_size, _ = x.shape
		
		# Initialize state if not provided
		if state is None:
			backend = x._backend
			shape = (self.num_layers * self.num_directions, batch_size, self.hidden_size)
			
			from .. import Tensor
			h_0 = Tensor.__new__(Tensor)
			h_0._backend = backend
			h_0._dtype = x._dtype
			h_0.device = x.device
			h_0.active_device = x.active_device
			h_0.data = backend.zeros(shape)
			h_0._requires_grad = False
			h_0._grad = None
			
			c_0 = Tensor.__new__(Tensor)
			c_0._backend = backend
			c_0._dtype = x._dtype
			c_0.device = x.device
			c_0.active_device = x.active_device
			c_0.data = backend.zeros(shape)
			c_0._requires_grad = False
			c_0._grad = None
			
			state = (h_0, c_0)
		
		h_0, c_0 = state
		
		# Process through layers
		layer_input = x
		h_n_list = []
		c_n_list = []
		
		for layer in range(self.num_layers):
			# Get initial states for this layer
			if self.bidirectional:
				h_fwd = self._make_tensor_from_data(h_0.data[layer * 2], x)
				c_fwd = self._make_tensor_from_data(c_0.data[layer * 2], x)
				h_bwd = self._make_tensor_from_data(h_0.data[layer * 2 + 1], x)
				c_bwd = self._make_tensor_from_data(c_0.data[layer * 2 + 1], x)
			else:
				h_fwd = self._make_tensor_from_data(h_0.data[layer], x)
				c_fwd = self._make_tensor_from_data(c_0.data[layer], x)
			
			# Forward direction
			forward_outputs = []
			
			for t in range(seq_len):
				x_t = self._get_timestep(layer_input, t)
				h_fwd, c_fwd = self.cells_forward[layer](x_t, (h_fwd, c_fwd))
				forward_outputs.append(h_fwd)
			
			h_n_list.append(h_fwd)
			c_n_list.append(c_fwd)
			
			# Backward direction
			if self.bidirectional:
				backward_outputs = []
				
				for t in range(seq_len - 1, -1, -1):
					x_t = self._get_timestep(layer_input, t)
					h_bwd, c_bwd = self.cells_backward[layer](x_t, (h_bwd, c_bwd))
					backward_outputs.insert(0, h_bwd)
				
				h_n_list.append(h_bwd)
				c_n_list.append(c_bwd)
				
				# Concatenate forward and backward
				from .. import engine
				outputs = []
				for fwd, bwd in zip(forward_outputs, backward_outputs):
					outputs.append(engine.concatenate([fwd, bwd], axis=1))
				layer_output = self._stack_outputs(outputs)
			else:
				layer_output = self._stack_outputs(forward_outputs)
			
			# Apply dropout
			if self.dropout is not None and layer < self.num_layers - 1:
				layer_output = self.dropout(layer_output)
			
			layer_input = layer_output
		
		output = layer_input
		
		# Stack final states
		h_n = self._stack_hidden(h_n_list, x)
		c_n = self._stack_hidden(c_n_list, x)
		
		if self.batch_first:
			output = output.T()
		
		return output, (h_n, c_n)
	
	def _make_tensor_from_data(self, data, reference):
		from .. import Tensor
		backend = reference._backend
		
		t = Tensor.__new__(Tensor)
		t._backend = backend
		t._dtype = reference._dtype
		t.device = reference.device
		t.active_device = reference.active_device
		t.data = data
		t._requires_grad = False
		t._grad = None
		return t
	
	def _get_timestep(self, x, t):
		backend = x._backend
		x_t_data = x.data[t]
		return self._make_tensor_from_data(x_t_data, x)
	
	def _stack_outputs(self, outputs):
		from .. import engine
		return engine.stack(outputs, axis=0)
	
	def _stack_hidden(self, h_list, reference):
		from .. import engine
		return engine.stack(h_list, axis=0)
	
	def extra_repr(self):
		s = f"{self.input_size}, {self.hidden_size}"
		if self.num_layers != 1:
			s += f", num_layers={self.num_layers}"
		if not self.batch_first:
			s += ", batch_first=False"
		if self.dropout_p != 0:
			s += f", dropout={self.dropout_p}"
		if self.bidirectional:
			s += ", bidirectional=True"
		return s


class GRU(Module):
	
	def __init__(self, input_size, hidden_size, num_layers=1, bias=True,
				 batch_first=False, dropout=0.0, bidirectional=False):
		super().__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.batch_first = batch_first
		self.dropout_p = dropout
		self.bidirectional = bidirectional
		self.num_directions = 2 if bidirectional else 1
		
		# Create GRU cells
		self.cells_forward = ModuleList()
		self.cells_backward = ModuleList() if bidirectional else None
		
		for layer in range(num_layers):
			layer_input_size = input_size if layer == 0 else hidden_size * self.num_directions
			self.cells_forward.append(
				GRUCell(layer_input_size, hidden_size, bias)
			)
			if bidirectional:
				self.cells_backward.append(
					GRUCell(layer_input_size, hidden_size, bias)
				)
		
		# Dropout
		if dropout > 0 and num_layers > 1:
			self.dropout = Dropout(dropout)
		else:
			self.dropout = None
	
	def forward(self, x, h_0=None):
		# Same structure as RNN.forward but with GRU cells
		if self.batch_first:
			x = x.T()
		
		seq_len, batch_size, _ = x.shape
		
		if h_0 is None:
			backend = x._backend
			from .. import Tensor
			h_0 = Tensor.__new__(Tensor)
			h_0._backend = backend
			h_0._dtype = x._dtype
			h_0.device = x.device
			h_0.active_device = x.active_device
			h_0.data = backend.zeros((
				self.num_layers * self.num_directions,
				batch_size,
				self.hidden_size
			))
			h_0._requires_grad = False
			h_0._grad = None
		
		layer_input = x
		h_n_list = []
		
		for layer in range(self.num_layers):
			if self.bidirectional:
				h_fwd = self._make_tensor_from_data(h_0.data[layer * 2], x)
				h_bwd = self._make_tensor_from_data(h_0.data[layer * 2 + 1], x)
			else:
				h_fwd = self._make_tensor_from_data(h_0.data[layer], x)
			
			# Forward
			forward_outputs = []
			for t in range(seq_len):
				x_t = self._get_timestep(layer_input, t)
				h_fwd = self.cells_forward[layer](x_t, h_fwd)
				forward_outputs.append(h_fwd)
			
			h_n_list.append(h_fwd)
			
			# Backward
			if self.bidirectional:
				backward_outputs = []
				for t in range(seq_len - 1, -1, -1):
					x_t = self._get_timestep(layer_input, t)
					h_bwd = self.cells_backward[layer](x_t, h_bwd)
					backward_outputs.insert(0, h_bwd)
				
				h_n_list.append(h_bwd)
				
				from .. import engine
				outputs = []
				for fwd, bwd in zip(forward_outputs, backward_outputs):
					outputs.append(engine.concatenate([fwd, bwd], axis=1))
				layer_output = self._stack_outputs(outputs)
			else:
				layer_output = self._stack_outputs(forward_outputs)
			
			if self.dropout is not None and layer < self.num_layers - 1:
				layer_output = self.dropout(layer_output)
			
			layer_input = layer_output
		
		output = layer_input
		h_n = self._stack_hidden(h_n_list, x)
		
		if self.batch_first:
			output = output.T()
		
		return output, h_n
	
	def _make_tensor_from_data(self, data, reference):
		from .. import Tensor
		backend = reference._backend
		
		t = Tensor.__new__(Tensor)
		t._backend = backend
		t._dtype = reference._dtype
		t.device = reference.device
		t.active_device = reference.active_device
		t.data = data
		t._requires_grad = False
		t._grad = None
		return t
	
	def _get_timestep(self, x, t):
		backend = x._backend
		x_t_data = x.data[t]
		return self._make_tensor_from_data(x_t_data, x)
	
	def _stack_outputs(self, outputs):
		from .. import engine
		return engine.stack(outputs, axis=0)
	
	def _stack_hidden(self, h_list, reference):
		from .. import engine
		return engine.stack(h_list, axis=0)
	
	def extra_repr(self):
		s = f"{self.input_size}, {self.hidden_size}"
		if self.num_layers != 1:
			s += f", num_layers={self.num_layers}"
		if not self.batch_first:
			s += ", batch_first=False"
		if self.dropout_p != 0:
			s += f", dropout={self.dropout_p}"
		if self.bidirectional:
			s += ", bidirectional=True"
		return s


__all__ = [
	'RNNCell', 'RNN',
	'LSTMCell', 'LSTM',
	'GRUCell', 'GRU',
]