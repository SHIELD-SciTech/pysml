
class fp32:
	def __init__(self):
		self.precission = "fp32"
	
	def get(self, backend):
		return backend.precission[self.precission]
	
	def __str__(self):
		return f"<pysml.dtype.{self.precission}>"
	
	def __repr__(self):
		return self.__str__()


class fp16:
	def __init__(self):
		self.precission = "fp16"
	
	def get(self, backend):
		return backend.precission[self.precission]
	
	def __str__(self):
		return f"<pysml.dtype.{self.precission}>"
	
	def __repr__(self):
		return self.__str__()


class bf16:
	def __init__(self):
		self.precission = "bf16"
	
	def get(self, backend):
		return backend.precission[self.precission]
	
	def __str__(self):
		return f"<pysml.dtype.{self.precission}>"
	
	def __repr__(self):
		return self.__str__()