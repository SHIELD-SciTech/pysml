
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