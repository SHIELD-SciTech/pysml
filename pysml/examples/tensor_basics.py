# pysml/examples/tensor_basics.py
import pysml

a = pysml.ones(2,3)
b = pysml.randn(2,3)
c = pysml.matmul(pysml.randn(3,4), pysml.randn(4,5))
d = pysml.softmax(b, axis=-1)

print("a:", a.shape, "b:", b.shape, "c:", c.shape, "d:", d.shape)
