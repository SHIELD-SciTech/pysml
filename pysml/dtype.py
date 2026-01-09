

class fp32:
        def __init__(self):
                self.precision = "fp32"

        def get(self, backend):
                mapping = getattr(backend, "precision", None) or getattr(backend, "precission", None)
                if mapping is None:
                        raise AttributeError("Backend does not provide a precision mapping")

                return mapping[self.precision]

        def __str__(self):
                return f"<pysml.dtype.{self.precision}>"

        def __repr__(self):
                return self.__str__()

        def __eq__(self, other):
                if isinstance(other, fp32):
                        return True
                return False

        def __hash__(self):
                return hash(self.precision)

class fp16:
        def __init__(self):
                self.precision = "fp16"

        def get(self, backend):
                mapping = getattr(backend, "precision", None) or getattr(backend, "precission", None)
                if mapping is None:
                        raise AttributeError("Backend does not provide a precision mapping")

                return mapping[self.precision]

        def __str__(self):
                return f"<pysml.dtype.{self.precision}>"

        def __repr__(self):
                return self.__str__()

        def __eq__(self, other):
                if isinstance(other, fp16):
                        return True
                return False

        def __hash__(self):
                return hash(self.precision)

class bf16:
        def __init__(self):
                self.precision = "bf16"

        def get(self, backend):
                mapping = getattr(backend, "precision", None) or getattr(backend, "precission", None)
                if mapping is None:
                        raise AttributeError("Backend does not provide a precision mapping")

                return mapping[self.precision]

        def __str__(self):
                return f"<pysml.dtype.{self.precision}>"

        def __repr__(self):
                return self.__str__()

        def __eq__(self, other):
                if isinstance(other, bf16):
                        return True
                return False

        def __hash__(self):
                return hash(self.precision)

# Convenience instances
float32 = fp32
float16 = fp16
bfloat16 = bf16

__all__ = ['fp32', 'fp16', 'bf16', 'float32', 'float16', 'bfloat16']
