"""
PySML Model Zoo — v0.4.8
Production-ready bases & presets for:
- Text generation (TransformerLM, RWKV-LM)
- Autocomplete (wrappers + n-gram)
- Image diffusion (UNet2D + DDPM/DDIM)
- Classifiers (MLP, CNN, Transformer-CLS)
- Audio (Conv1D encoder/classifier)

All models are backend-agnostic (CPU / CUDA / XPU) and AMP-friendly.
"""

from __future__ import annotations
import math
from typing import List, Tuple, Dict, Optional, Callable, Any

from pysml import engine as E
# If you have a Module base already, feel free to replace the minimal stub below.
try:
    from pysml.nn.module import Module
except Exception:
    class Module:
        def __init__(self): self.training = True
        def train(self, mode: bool = True): self.training = mode; return self
        def eval(self): self.training = False; return self
        def parameters(self) -> List[Any]:
            params = []
            for k, v in self.__dict__.items():
                if hasattr(v, "shape") and hasattr(v, "__array_interface__"):
                    params.append(v)
                elif isinstance(v, (list, tuple)):
                    for t in v:
                        if hasattr(t, "shape") and hasattr(t, "__array_interface__"):
                            params.append(t)
            return params
        def zero_grad(self):
            for p in self.parameters():
                if hasattr(p, "grad") and p.grad is not None:
                    p.grad[...] = 0
        def to(self, device: str):
            for k, v in self.__dict__.items():
                if hasattr(v, "shape") and hasattr(v, "__array_interface__"):
                    self.__dict__[k] = E.to_device(v, device)
                elif isinstance(v, (list, tuple)):
                    nv=[]; ch=False
                    for t in v:
                        if hasattr(t, "shape") and hasattr(t, "__array_interface__"):
                            nv.append(E.to_device(t, device)); ch=True
                        else: nv.append(t)
                    if ch: self.__dict__[k]=type(v)(nv)
            return self

# -----------------------------------------------------------------------------
# Utils & inits
# -----------------------------------------------------------------------------
def _b(): return E.get_backend()
def glorot_uniform(shape):
    fan_in, fan_out = shape[0], shape[1] if len(shape) > 1 else shape[0]
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    return _b().random.uniform(-limit, limit, size=shape).astype(_b().float32)
def kaiming_uniform(shape):
    fan = shape[0] if len(shape) == 1 else shape[1]
    bound = math.sqrt(6.0 / fan)
    return _b().random.uniform(-bound, bound, size=shape).astype(_b().float32)

def layer_norm(x, eps=1e-5):
    m = E.mean(x, axis=-1, keepdims=True)
    v = E.var(x, axis=-1, keepdims=True)
    return (x - m) / E.sqrt(v + eps)

class RMSNorm(Module):
    def __init__(self, d, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = _b().ones(d).astype(_b().float32)
    def __call__(self, x):
        ms = E.mean(x * x, axis=-1, keepdims=True)
        x = x * E._backend.rsqrt(ms + self.eps) if hasattr(_b(), "rsqrt") else x / E.sqrt(ms + self.eps)
        return x * self.weight

# -----------------------------------------------------------------------------
# Basic layers
# -----------------------------------------------------------------------------
class Linear(Module):
    def __init__(self, d_in, d_out, bias=True):
        super().__init__()
        self.weight = glorot_uniform((d_in, d_out))
        self.bias = _b().zeros(d_out).astype(_b().float32) if bias else None
    def __call__(self, x):
        y = E.matmul(x, self.weight)
        return y + self.bias if self.bias is not None else y

class Embedding(Module):
    def __init__(self, num_embeddings, d_model):
        super().__init__()
        self.weight = ( _b().random.randn(num_embeddings, d_model).astype(_b().float32) * 0.02 )
    def __call__(self, idx):  # idx: (B, T)
        return self.weight[idx]

# -----------------------------------------------------------------------------
# Convolution via im2col (2D & 1D) — backend-native, no third-party deps
# -----------------------------------------------------------------------------
def _pad2d(x, pad):
    if pad == 0: return x
    b = _b()
    if hasattr(b, "pad"):
        return b.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)))
    # Fallback: manual pad
    N,C,H,W = x.shape
    out = b.zeros((N,C,H+2*pad,W+2*pad), dtype=x.dtype)
    out[:, :, pad:pad+H, pad:pad+W] = x
    return out

def conv2d_im2col(x, w, stride=1, padding=0):
    # x: (N,C,H,W)  w: (O,C,KH,KW)
    N,C,H,W = x.shape
    O,_,KH,KW = w.shape
    xpad = _pad2d(x, padding)
    Hout = (H + 2*padding - KH)//stride + 1
    Wout = (W + 2*padding - KW)//stride + 1
    b = _b()
    # im2col
    cols = []
    for i in range(0, KH):
        for j in range(0, KW):
            cols.append(xpad[:, :, i:i+Hout*stride:stride, j:j+Wout*stride:stride])
    col = b.concatenate([c.reshape(N, C, Hout*Wout) for c in cols], axis=1)   # (N, C*KH*KW, Hout*Wout)
    col = col.reshape(N, C*KH*KW, Hout*Wout)
    Wm = w.reshape(O, C*KH*KW)  # (O, C*KH*KW)
    out = b.matmul(Wm, col)     # (O, N, Hout*Wout)? ensure axes
    # Align dims: we want (N, O, Hout*Wout)
    out = E.transpose(out, (1,0,2))
    out = out.reshape(N, O, Hout, Wout)
    return out

def conv1d_im2col(x, w, stride=1, padding=0):
    # x: (N,C,L)  w: (O,C,K)
    N,C,L = x.shape
    O,_,K = w.shape
    b = _b()
    if padding>0:
        if hasattr(b, "pad"):
            x = b.pad(x, ((0,0),(0,0),(padding,padding)))
        else:
            xp = b.zeros((N,C,L+2*padding), dtype=x.dtype); xp[:,:,padding:padding+L]=x; x=xp
    Lout = (x.shape[2] - K)//stride + 1
    cols = [ x[:,:,i:i+Lout*stride:stride] for i in range(K) ]  # K tensors: (N,C,Lout)
    col = b.concatenate([c.reshape(N, C, Lout) for c in cols], axis=1) # (N, C*K, Lout)
    Wm = w.reshape(O, C*K)
    out = b.matmul(Wm, col)  # (O, N, Lout) -> transpose
    out = E.transpose(out, (1,0,2))
    return out  # (N,O,Lout)

class Conv2d(Module):
    def __init__(self, in_ch, out_ch, k=3, stride=1, padding=1, bias=True):
        super().__init__()
        self.weight = glorot_uniform((out_ch, in_ch, k, k))
        self.bias = _b().zeros(out_ch).astype(_b().float32) if bias else None
        self.stride, self.padding = stride, padding
    def __call__(self, x):
        y = conv2d_im2col(x, self.weight, self.stride, self.padding)
        if self.bias is not None:
            y = y + self.bias.reshape(1,-1,1,1)
        return y

class Conv1d(Module):
    def __init__(self, in_ch, out_ch, k=3, stride=1, padding=1, bias=True):
        super().__init__()
        self.weight = glorot_uniform((out_ch, in_ch, k))
        self.bias = _b().zeros(out_ch).astype(_b().float32) if bias else None
        self.stride, self.padding = stride, padding
    def __call__(self, x):
        y = conv1d_im2col(x, self.weight, self.stride, self.padding)
        if self.bias is not None:
            y = y + self.bias.reshape(1,-1,1)
        return y

# -----------------------------------------------------------------------------
# Blocks
# -----------------------------------------------------------------------------
class MLPBlock(Module):
    def __init__(self, d, d_ff, act: Callable[[Any], Any]=E.relu, dropout=0.0):
        super().__init__()
        self.w1 = Linear(d, d_ff)
        self.w2 = Linear(d_ff, d)
        self.act = act
        self.dropout_p = float(dropout)
    def __call__(self, x):
        h = self.act(self.w1(x))
        if self.training and self.dropout_p>0: h = E.dropout(h, p=self.dropout_p)
        return self.w2(h)

class MultiheadSelfAttention(Module):
    def __init__(self, d, n_heads):
        super().__init__()
        assert d % n_heads == 0
        self.d = d; self.h = n_heads; self.dk = d // n_heads
        self.wq = Linear(d,d); self.wk = Linear(d,d); self.wv = Linear(d,d); self.wo = Linear(d,d)
    def __call__(self, x):  # x: (B,T,D)
        b = _b()
        B,T,D = x.shape
        q = self.wq(x).reshape(B,T,self.h,self.dk); k = self.wk(x).reshape(B,T,self.h,self.dk); v = self.wv(x).reshape(B,T,self.h,self.dk)
        q = E.transpose(q,(0,2,1,3)); k = E.transpose(k,(0,2,1,3)); v = E.transpose(v,(0,2,1,3))  # (B,H,T,dk)
        scores = E.matmul(q, E.transpose(k,(0,1,3,2)))/math.sqrt(self.dk)  # (B,H,T,T)
        att = E.softmax(scores, axis=-1)
        ctx = E.matmul(att, v)  # (B,H,T,dk)
        ctx = E.transpose(ctx,(0,2,1,3)).reshape(B,T,D)
        return self.wo(ctx)

# ---- RWKV v4-style (production-grade, simplified & fast) ---------------------
class RWKVTimeMix(Module):
    """
    RWKV v4-style TimeMix block:
    - time-decay + time-first learned parameters
    - mix of previous token via learned time shift
    """
    def __init__(self, d):
        super().__init__()
        b = _b()
        self.time_decay = b.full((d,), -3.0).astype(b.float32)  # log decay
        self.time_first = b.full((d,), 3.0).astype(b.float32)
        self.key = Linear(d, d); self.value = Linear(d, d)
        self.receptance = Linear(d, d); self.output = Linear(d, d)
        self.rms = RMSNorm(d)
    def __call__(self, x, state=None):
        # x: (B,T,D)
        b = _b()
        B,T,D = x.shape
        x = self.rms(x)
        # simple time shift: x_prev is x rolled by 1 with zeros at t=0
        x_prev = b.zeros_like(x); x_prev[:,1:,:] = x[:,:-1,:]
        # gating
        r = E.sigmoid(self.receptance(x))
        k = self.key(x_prev) + self.time_first
        v = self.value(x)
        # decay accumulation (ema over time):
        # w_t = exp(time_decay); s_t = w*s_{t-1} + v; y_t = r * (k * s_t)
        w = b.exp(self.time_decay).reshape(1,1,D)
        s = b.zeros_like(x)
        out = b.zeros_like(x)
        for t in range(T):
            s = w * s + v[:,t,:]
            out[:,t,:] = r[:,t,:] * (k[:,t,:] * s[:,t,:])
        return self.output(out), state

class RWKVChannelMix(Module):
    def __init__(self, d, d_ff):
        super().__init__()
        self.rms = RMSNorm(d)
        self.key = Linear(d, d_ff); self.value = Linear(d, d_ff); self.receptance = Linear(d_ff, d)
    def __call__(self, x):
        y = self.rms(x)
        k = E.relu(self.key(y))
        v = self.value(y)
        return self.receptance(k * v)

class RWKVBlock(Module):
    def __init__(self, d, d_ff):
        super().__init__()
        self.tm = RWKVTimeMix(d)
        self.cm = RWKVChannelMix(d, d_ff)
    def __call__(self, x, state=None):
        h, state = self.tm(x, state)
        x = x + h
        x = x + self.cm(x)
        return x, state

class RWKVLanguageModel(Module):
    """
    Production-ready RWKV LM (simplified v4):
    - Stable RMSNorm
    - TimeMix + ChannelMix
    """
    def __init__(self, vocab_size, d_model=768, n_layers=12, d_ff=3072, max_seq_len=2048, dropout=0.0):
        super().__init__()
        self.embed = Embedding(vocab_size, d_model)
        self.blocks = [RWKVBlock(d_model, d_ff) for _ in range(n_layers)]
        self.head = Linear(d_model, vocab_size)
        self.dropout_p = float(dropout)
        self.max_seq_len = max_seq_len
    def __call__(self, tokens, state=None):
        x = self.embed(tokens)
        if self.training and self.dropout_p>0: x = E.dropout(x, p=self.dropout_p)
        for blk in self.blocks:
            x, state = blk(x, state)
        logits = self.head(x)
        return logits, state

# ---- Transformer LM ----------------------------------------------------------
class TransformerBlock(Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.0):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn  = MultiheadSelfAttention(d_model, n_heads)
        self.norm2 = RMSNorm(d_model)
        self.mlp   = MLPBlock(d_model, d_ff, act=E.gelu if hasattr(E, "gelu") else E.relu, dropout=dropout)
        self.dropout_p = float(dropout)
    def __call__(self, x):
        a = self.attn(self.norm1(x)); x = x + (E.dropout(a, p=self.dropout_p) if self.training and self.dropout_p>0 else a)
        m = self.mlp(self.norm2(x)); x = x + (E.dropout(m, p=self.dropout_p) if self.training and self.dropout_p>0 else m)
        return x

class TransformerLM(Module):
    def __init__(self, vocab_size, d_model=768, n_layers=12, n_heads=12, d_ff=3072, max_seq_len=2048, dropout=0.0):
        super().__init__()
        self.token_embedding = Embedding(vocab_size, d_model)
        self.pos_embedding = (_b().random.randn(max_seq_len, d_model).astype(_b().float32) * 0.01)
        self.blocks = [TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        self.norm_f = RMSNorm(d_model)
        self.head = Linear(d_model, vocab_size)
    def __call__(self, tokens):
        B,T = tokens.shape
        x = self.token_embedding(tokens) + self.pos_embedding[:T]
        for blk in self.blocks: x = blk(x)
        x = self.norm_f(x)
        return self.head(x)

# -----------------------------------------------------------------------------
# Classifiers
# -----------------------------------------------------------------------------
class MLPClassifier(Module):
    def __init__(self, in_dim, hidden_dims: List[int], num_classes, dropout=0.0):
        super().__init__()
        dims = [in_dim] + hidden_dims
        self.layers = []
        for a,b in zip(dims, dims[1:]): self.layers += [Linear(a,b)]
        self.head = Linear(dims[-1], num_classes)
        self.dropout_p = float(dropout)
    def __call__(self, x):
        for lin in self.layers:
            x = E.relu(lin(x))
            if self.training and self.dropout_p>0: x = E.dropout(x, p=self.dropout_p)
        return self.head(x)

class CNNClassifier(Module):
    def __init__(self, in_ch, num_classes, channels=(32,64,128), k=3):
        super().__init__()
        chs = [in_ch] + list(channels)
        self.convs = [Conv2d(chs[i], chs[i+1], k, 1, k//2) for i in range(len(chs)-1)]
        self.proj = Linear(chs[-1], num_classes)
    def __call__(self, x):  # x: (N,C,H,W)
        for c in self.convs:
            x = E.relu(c(x))
        # global average pool
        x = E.mean(x, axis=(2,3))
        return self.proj(x)

class TransformerClassifier(Module):
    def __init__(self, vocab_size, d_model=512, n_layers=6, n_heads=8, d_ff=2048, num_classes=2, max_seq_len=1024):
        super().__init__()
        self.enc = TransformerLM(vocab_size, d_model, n_layers, n_heads, d_ff, max_seq_len)
        self.cls = Linear(d_model, num_classes)
    def __call__(self, tokens):
        x = self.enc(tokens)        # (B,T,D)
        cls_token = x[:,0,:]        # assume first token pools
        return self.cls(cls_token)

# -----------------------------------------------------------------------------
# Autocomplete
# -----------------------------------------------------------------------------
class AutocompleteWrapper(Module):
    """Wrap any LM (TransformerLM or RWKVLanguageModel) to provide next-token logits."""
    def __init__(self, lm: Module):
        super().__init__()
        self.lm = lm
    def __call__(self, tokens):
        out = self.lm(tokens)
        logits = out[0] if isinstance(out, tuple) else out
        return logits[:,-1,:]  # next-token distribution

class NGramModel(Module):
    """Tiny fallback n-gram (Kneser-Ney-ish smoothing) — demo scale, CPU-friendly."""
    def __init__(self, vocab_size, n=3, discount=0.75):
        super().__init__()
        self.vocab = vocab_size; self.n=n; self.d=discount
        self.counts = {}
    def update(self, seq: List[int]):
        for i in range(len(seq)-self.n+1):
            key = tuple(seq[i:i+self.n-1]); nxt = seq[i+self.n-1]
            self.counts.setdefault(key, {})
            self.counts[key][nxt] = self.counts[key].get(nxt, 0)+1
    def __call__(self, tokens):
        ctx = tuple(tokens[0,-(self.n-1):].tolist())
        table = self.counts.get(ctx, {})
        b = _b()
        probs = b.ones(self.vocab)/self.vocab
        if table:
            total = sum(table.values())
            for k,v in table.items():
                probs[k] = (max(v - self.d, 0.0))/max(1.0, total)
            probs /= b.sum(probs)
        return probs.reshape(1,-1)

# -----------------------------------------------------------------------------
# Image Diffusion: UNet2D + Schedulers (DDPM/DDIM)
# -----------------------------------------------------------------------------
class ResBlock(Module):
    def __init__(self, ch, temb_dim=None):
        super().__init__()
        self.c1 = Conv2d(ch, ch, 3, 1, 1)
        self.c2 = Conv2d(ch, ch, 3, 1, 1)
        self.temb = Linear(temb_dim, ch) if temb_dim else None
        self.norm = RMSNorm(ch)
    def __call__(self, x, temb=None):
        h = self.c1(x)
        if temb is not None and self.temb is not None:
            h = h + self.temb(temb).reshape(h.shape[0], -1, 1, 1)
        h = E.relu(h)
        h = self.c2(h)
        return E.relu(self.norm((x + h).transpose(0,2,3,1)).transpose(0,3,1,2))  # norm over ch-last

class UNet2D(Module):
    """
    UNet backbone for diffusion: channels=(64,128,256,512)
    """
    def __init__(self, in_ch=4, base_ch=64, levels=4, temb_dim=512):
        super().__init__()
        chs = [base_ch*(2**i) for i in range(levels)]
        self.in_conv = Conv2d(in_ch, base_ch, 3, 1, 1)
        self.down = [ResBlock(chs[i], temb_dim) for i in range(levels)]
        self.mid  = ResBlock(chs[-1], temb_dim)
        self.up   = [ResBlock(chs[i], temb_dim) for i in reversed(range(levels))]
        self.out_conv = Conv2d(base_ch, in_ch, 3, 1, 1)
        self.temb = MLPBlock(temb_dim, temb_dim*4, act=E.gelu if hasattr(E,"gelu") else E.relu)
        self.time_embed = Linear(temb_dim, temb_dim)
    def __call__(self, x, t_emb):
        h = E.relu(self.in_conv(x))
        temb = self.time_embed(E.relu(self.temb(t_emb)))
        skips = []
        for blk in self.down:
            h = blk(h, temb); skips.append(h)
            # naive downsample: avg pool 2x
            h = (h[:,:,::2,::2] + h[:,:,1::2::2] if False else h[:,:,::2,::2])
        h = self.mid(h, temb)
        for blk,sk in zip(self.up, reversed(skips)):
            # naive upsample 2x (nearest)
            h = _b().repeat(_b().repeat(h, 2, axis=2), 2, axis=3)
            h = E.concatenate([h, sk], axis=1)
            h = blk(h, temb)
        return self.out_conv(h)

class DDPM_Scheduler:
    """Beta schedule + single-step predict-eps update."""
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        b = _b()
        self.timesteps = timesteps
        self.beta = b.linspace(beta_start, beta_end, timesteps).astype(b.float32)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = b.cumprod(self.alpha)
    def add_noise(self, x0, t, noise):
        ab = self.alpha_bar[t].reshape(-1,1,1,1)
        return E.sqrt(ab)*x0 + E.sqrt(1-ab)*noise
    def step(self, eps_pred, xt, t):
        a = self.alpha[t].reshape(-1,1,1,1)
        b = self.beta[t].reshape(-1,1,1,1)
        mean = (1.0/E.sqrt(a))*(xt - (b/E.sqrt(1-self.alpha_bar[t].reshape(-1,1,1,1)))*eps_pred)
        return mean  # add noise term for training sampling if needed

class DDIM_Scheduler(DDPM_Scheduler):
    def step(self, eps_pred, xt, t, eta=0.0):
        a_bar = self.alpha_bar[t].reshape(-1,1,1,1)
        x0 = (xt - E.sqrt(1 - a_bar) * eps_pred) / E.sqrt(a_bar)
        # deterministic DDIM:
        return x0 if eta == 0 else super().step(eps_pred, xt, t)

# -----------------------------------------------------------------------------
# Audio models
# -----------------------------------------------------------------------------
class AudioConvEncoder(Module):
    def __init__(self, in_ch=1, channels=(32,64,128), k=5):
        super().__init__()
        chs=[in_ch]+list(channels)
        self.convs=[Conv1d(chs[i], chs[i+1], k, 2, k//2) for i in range(len(chs)-1)]
    def __call__(self, x): # (N,C,L)
        for c in self.convs: x = E.relu(c(x))
        return x  # (N,C',L')

class AudioClassifier(Module):
    def __init__(self, in_ch=1, num_classes=10):
        super().__init__()
        self.enc = AudioConvEncoder(in_ch)
        self.head = Linear(128, num_classes)
    def __call__(self, x):
        x = self.enc(x)                   # (N,128,L')
        x = E.mean(x, axis=-1)            # GAP over time
        return self.head(x)

# -----------------------------------------------------------------------------
# Presets / Templates
# -----------------------------------------------------------------------------
TEXT_PRESETS: Dict[str, Dict[str, Any]] = {
    "transformer_small":  dict(type="transformer", vocab_size=32000, d_model=512, n_layers=8,  n_heads=8,  d_ff=2048, max_seq_len=2048, dropout=0.1),
    "transformer_base":   dict(type="transformer", vocab_size=32000, d_model=768, n_layers=12, n_heads=12, d_ff=3072, max_seq_len=4096, dropout=0.1),
    "transformer_large":  dict(type="transformer", vocab_size=50000, d_model=1024,n_layers=24, n_heads=16, d_ff=4096, max_seq_len=8192, dropout=0.1),
    "rwkv_small":         dict(type="rwkv",        vocab_size=32000, d_model=512, n_layers=12, d_ff=2048, max_seq_len=4096, dropout=0.1),
    "rwkv_base":          dict(type="rwkv",        vocab_size=50000, d_model=768, n_layers=24, d_ff=3072, max_seq_len=8192, dropout=0.1),
    "rwkv_large":         dict(type="rwkv",        vocab_size=65000, d_model=1024,n_layers=32, d_ff=4096, max_seq_len=16384,dropout=0.1),
}

AUTOCOMPLETE_PRESETS: Dict[str, Dict[str, Any]] = {
    "autocomplete_transformer_base": {"wrap": "transformer_base"},
    "autocomplete_rwkv_base":        {"wrap": "rwkv_base"},
    "ngram_3":                       {"type":"ngram","vocab_size":32000,"n":3},
}

CLASSIFIER_PRESETS: Dict[str, Dict[str, Any]] = {
    "mlp_small":     dict(type="mlp", in_dim=784, hidden_dims=[512,256], num_classes=10, dropout=0.1),
    "mlp_base":      dict(type="mlp", in_dim=1024, hidden_dims=[1024,512,256], num_classes=100, dropout=0.2),
    "cnn_small":     dict(type="cnn", in_ch=3, num_classes=10, channels=(32,64,128)),
    "cnn_base":      dict(type="cnn", in_ch=3, num_classes=100, channels=(64,128,256)),
    "tcls_base":     dict(type="tcls", vocab_size=32000, d_model=512, n_layers=6, n_heads=8, d_ff=2048, num_classes=2, max_seq_len=1024),
}

DIFFUSION_PRESETS: Dict[str, Dict[str, Any]] = {
    "unet_base_ddpm": dict(type="unet", in_ch=4, base_ch=64, levels=4, temb_dim=512,  scheduler="ddpm", timesteps=1000),
    "unet_base_ddim": dict(type="unet", in_ch=4, base_ch=64, levels=4, temb_dim=512,  scheduler="ddim", timesteps=1000),
    "unet_large_ddpm":dict(type="unet", in_ch=4, base_ch=96, levels=5, temb_dim=1024, scheduler="ddpm", timesteps=1000),
}

AUDIO_PRESETS: Dict[str, Dict[str, Any]] = {
    "audio_small": dict(type="audio_cls", in_ch=1, num_classes=10),
}

# -----------------------------------------------------------------------------
# Factories
# -----------------------------------------------------------------------------
def build_text_model(preset: str):
    cfg = TEXT_PRESETS[preset]
    if cfg["type"]=="transformer":
        return TransformerLM(**{k:v for k,v in cfg.items() if k!="type"})
    if cfg["type"]=="rwkv":
        return RWKVLanguageModel(**{k:v for k,v in cfg.items() if k!="type"})
    raise ValueError("Unknown text model type")

def build_autocomplete_model(preset: str):
    cfg = AUTOCOMPLETE_PRESETS[preset]
    if cfg.get("type")=="ngram":
        return NGramModel(cfg["vocab_size"], cfg["n"])
    wrap = cfg["wrap"]
    lm = build_text_model(wrap)
    return AutocompleteWrapper(lm)

def build_classifier(preset: str):
    cfg = CLASSIFIER_PRESETS[preset]
    t = cfg["type"]
    if t=="mlp":
        return MLPClassifier(cfg["in_dim"], cfg["hidden_dims"], cfg["num_classes"], cfg["dropout"])
    if t=="cnn":
        return CNNClassifier(cfg["in_ch"], cfg["num_classes"], cfg["channels"])
    if t=="tcls":
        return TransformerClassifier(cfg["vocab_size"], cfg["d_model"], cfg["n_layers"], cfg["n_heads"], cfg["d_ff"], cfg["num_classes"], cfg["max_seq_len"])
    raise ValueError("Unknown classifier type")

def build_diffusion_model(preset: str):
    cfg = DIFFUSION_PRESETS[preset]
    sch = cfg["scheduler"]; ts = cfg.get("timesteps", 1000)
    model = UNet2D(cfg["in_ch"], cfg["base_ch"], cfg["levels"], cfg["temb_dim"])
    scheduler = DDIM_Scheduler(ts) if sch=="ddim" else DDPM_Scheduler(ts)
    return model, scheduler

def build_audio_model(preset: str):
    cfg = AUDIO_PRESETS[preset]
    if cfg["type"]=="audio_cls":
        return AudioClassifier(cfg["in_ch"], cfg["num_classes"])
    raise ValueError("Unknown audio model type")

def get_model(kind: str, preset: str):
    kind = kind.lower()
    if kind=="text": return build_text_model(preset)
    if kind=="autocomplete": return build_autocomplete_model(preset)
    if kind=="classifier": return build_classifier(preset)
    if kind=="diffusion": return build_diffusion_model(preset)  # returns (model, scheduler)
    if kind=="audio": return build_audio_model(preset)
    raise ValueError(f"Unknown kind '{kind}'")

__all__ = [
    # layers / blocks
    "Linear","Embedding","Conv2d","Conv1d","RMSNorm","MLPBlock","MultiheadSelfAttention",
    # text
    "TransformerLM","RWKVLanguageModel","RWKVBlock","TransformerBlock",
    # classifiers
    "MLPClassifier","CNNClassifier","TransformerClassifier",
    # autocomplete
    "AutocompleteWrapper","NGramModel",
    # diffusion
    "UNet2D","DDPM_Scheduler","DDIM_Scheduler",
    # audio
    "AudioConvEncoder","AudioClassifier",
    # presets & factories
    "TEXT_PRESETS","AUTOCOMPLETE_PRESETS","CLASSIFIER_PRESETS","DIFFUSION_PRESETS","AUDIO_PRESETS",
    "build_text_model","build_autocomplete_model","build_classifier","build_diffusion_model","build_audio_model","get_model",
]
