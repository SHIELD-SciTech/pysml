
class ViewMetadata:
	def __init__(self, is_view=False, base=None, owns_memory=True):
		self.is_view = is_view
		self.base = base
		self.owns_memory = owns_memory
	
	def mark_as_view(self, base_tensor):
		self.is_view = True
		self.base = base_tensor
		self.owns_memory = False
	
	def mark_as_owner(self):
		self.is_view = False
		self.base = None
		self.owns_memory = True
	
	def clone(self):
		return ViewMetadata(
			is_view=self.is_view,
			base=self.base,
			owns_memory=self.owns_memory
		)
	
	def get_base(self):
		if not self.is_view or self.base is None:
			return None
		
		# Follow the chain to the actual owner
		current = self.base
		while hasattr(current, '_view_metadata') and current._view_metadata.is_view:
			if current._view_metadata.base is None:
				break
			current = current._view_metadata.base
		
		return current
	
	def __repr__(self):
		return f"ViewMetadata(is_view={self.is_view}, owns_memory={self.owns_memory})"

