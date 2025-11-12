# PySML Neural Network Module
# Complete neural network library for building LLMs, diffusion models, CNNs, RNNs, and more

from .module import (
	Module,
	Parameter,
	Sequential,
	ModuleList,
	ModuleDict,
	Identity,
	get_parameter_count,
	freeze_module,
	unfreeze_module,
)

from .linear import (
	Linear,
	Bilinear,
	LazyLinear,
	init_xavier_uniform_,
	init_xavier_normal_,
	init_kaiming_uniform_,
	init_kaiming_normal_,
)

from .activation import (
	ReLU,
	LeakyReLU,
	PReLU,
	ELU,
	SELU,
	GELU,
	SiLU,
	Swish,
	Mish,
	Tanh,
	Sigmoid,
	Hardsigmoid,
	Hardswish,
	Softmax,
	LogSoftmax,
	Softmin,
	LogSigmoid,
	Softplus,
	Softshrink,
	Hardshrink,
	Tanhshrink,
	GLU,
	SwiGLU,
)

from .conv import (
	Conv1d,
	Conv2d,
	Conv3d,
	ConvTranspose2d,
)

from .pooling import (
	MaxPool1d,
	MaxPool2d,
	AvgPool1d,
	AvgPool2d,
	AdaptiveAvgPool1d,
	AdaptiveAvgPool2d,
	AdaptiveMaxPool1d,
	AdaptiveMaxPool2d,
	GlobalAvgPool2d,
	GlobalMaxPool2d,
)

from .normalization import (
	LayerNorm,
	RMSNorm,
	BatchNorm1d,
	BatchNorm2d,
	BatchNorm3d,
	GroupNorm,
	InstanceNorm1d,
	InstanceNorm2d,
	InstanceNorm3d,
	LocalResponseNorm,
)

from .dropout import (
	Dropout,
	Dropout1d,
	Dropout2d,
	Dropout3d,
	AlphaDropout,
	FeatureAlphaDropout,
)

from .embedding import (
	Embedding,
	EmbeddingBag,
)

from .rnn import (
	RNNCell,
	RNN,
	LSTMCell,
	LSTM,
	GRUCell,
	GRU,
)

from .attention import (
	MultiHeadAttention,
	MultiHeadSelfAttention,
	CrossAttention,
	create_causal_mask,
	create_padding_mask,
)

from .transformer import (
	TransformerEncoderLayer,
	TransformerDecoderLayer,
	TransformerEncoder,
	TransformerDecoder,
	Transformer,
	GPTBlock,
	GPTModel,
	LLaMABlock,
)

from .positional import (
	SinusoidalPositionalEncoding,
	LearnedPositionalEmbedding,
	RotaryPositionalEmbedding,
	ALiBiPositionalBias,
	AbsolutePositionalEmbedding,
)

from .loss import (
	Loss,
	MSELoss,
	MSE,
	L1Loss,
	L1,
	SmoothL1Loss,
	CrossEntropyLoss,
	CrossEntropy,
	NLLLoss,
	NLL,
	BCELoss,
	BCE,
	BCEWithLogitsLoss,
	KLDivLoss,
	HingeLoss,
	CosineEmbeddingLoss,
	TripletMarginLoss,
	CTCLoss,
	FocalLoss,
)

from .upsample import (
	Upsample,
	UpsamplingNearest2d,
	UpsamplingBilinear2d,
	PixelShuffle,
	PixelUnshuffle,
	Interpolate,
	interpolate,
)

from .optim import (
	Optimizer,
	SGD,
	Adam,
	AdamW,
	RMSprop,
)


__all__ = [
	# Module base classes
	'Module',
	'Parameter',
	'Sequential',
	'ModuleList',
	'ModuleDict',
	'Identity',
	'get_parameter_count',
	'freeze_module',
	'unfreeze_module',
	
	# Linear layers
	'Linear',
	'Bilinear',
	'LazyLinear',
	'init_xavier_uniform_',
	'init_xavier_normal_',
	'init_kaiming_uniform_',
	'init_kaiming_normal_',
	
	# Activation functions
	'ReLU',
	'LeakyReLU',
	'PReLU',
	'ELU',
	'SELU',
	'GELU',
	'SiLU',
	'Swish',
	'Mish',
	'Tanh',
	'Sigmoid',
	'Hardsigmoid',
	'Hardswish',
	'Softmax',
	'LogSoftmax',
	'Softmin',
	'LogSigmoid',
	'Softplus',
	'Softshrink',
	'Hardshrink',
	'Tanhshrink',
	'GLU',
	'SwiGLU',
	
	# Convolutional layers
	'Conv1d',
	'Conv2d',
	'Conv3d',
	'ConvTranspose2d',
	
	# Pooling layers
	'MaxPool1d',
	'MaxPool2d',
	'AvgPool1d',
	'AvgPool2d',
	'AdaptiveAvgPool1d',
	'AdaptiveAvgPool2d',
	'AdaptiveMaxPool1d',
	'AdaptiveMaxPool2d',
	'GlobalAvgPool2d',
	'GlobalMaxPool2d',
	
	# Normalization layers
	'LayerNorm',
	'RMSNorm',
	'BatchNorm1d',
	'BatchNorm2d',
	'BatchNorm3d',
	'GroupNorm',
	'InstanceNorm1d',
	'InstanceNorm2d',
	'InstanceNorm3d',
	'LocalResponseNorm',
	
	# Dropout layers
	'Dropout',
	'Dropout1d',
	'Dropout2d',
	'Dropout3d',
	'AlphaDropout',
	'FeatureAlphaDropout',
	
	# Embedding layers
	'Embedding',
	'EmbeddingBag',
	
	# Recurrent layers
	'RNNCell',
	'RNN',
	'LSTMCell',
	'LSTM',
	'GRUCell',
	'GRU',
	
	# Attention mechanisms
	'MultiHeadAttention',
	'MultiHeadSelfAttention',
	'CrossAttention',
	'create_causal_mask',
	'create_padding_mask',
	
	# Transformer models
	'TransformerEncoderLayer',
	'TransformerDecoderLayer',
	'TransformerEncoder',
	'TransformerDecoder',
	'Transformer',
	'GPTBlock',
	'GPTModel',
	'LLaMABlock',
	
	# Positional encodings
	'SinusoidalPositionalEncoding',
	'LearnedPositionalEmbedding',
	'RotaryPositionalEmbedding',
	'ALiBiPositionalBias',
	'AbsolutePositionalEmbedding',
	
	# Loss functions
	'Loss',
	'MSELoss',
	'MSE',
	'L1Loss',
	'L1',
	'SmoothL1Loss',
	'CrossEntropyLoss',
	'CrossEntropy',
	'NLLLoss',
	'NLL',
	'BCELoss',
	'BCE',
	'BCEWithLogitsLoss',
	'KLDivLoss',
	'HingeLoss',
	'CosineEmbeddingLoss',
	'TripletMarginLoss',
	'CTCLoss',
	'FocalLoss',
	
	# Upsampling/Interpolation
	'Upsample',
	'UpsamplingNearest2d',
	'UpsamplingBilinear2d',
	'PixelShuffle',
	'PixelUnshuffle',
	'Interpolate',
	'interpolate',
	
	# Optimizers
	'Optimizer',
	'SGD',
	'Adam',
	'AdamW',
	'RMSprop',
]


# Version info
__version__ = '0.5.1'
__author__ = 'PySML Contributors'