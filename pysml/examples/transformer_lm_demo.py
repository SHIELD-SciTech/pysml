# pysml/examples/transformer_lm_demo.py
import pysml
from pysml.nn.models import TransformerLM

pysml.engine.set_device(next((d for d in pysml.engine.get_available_devices()
                              if d.startswith(("cuda","xpu"))), "cpu"))

model = TransformerLM.from_preset('TINY', vocab_size=8000)
x = pysml.Tensor((_np.random.randint(0, 8000, (2, 16))).astype("int32"))
logits = model(x)
print("logits:", logits.shape)
