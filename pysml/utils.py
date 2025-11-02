

def build_graph(tensor, nodes=None, edges=None):
	if nodes is None:
		nodes = set()
	if edges is None:
		edges = set()

	# Each tensor is a node
	nodes.add(tensor)

	# If it was created by an operation, add edges to parents
	if tensor._grad_fn:
		for parent in tensor._grad_fn.parents:
			edges.add((parent, tensor))
			build_graph(parent, nodes, edges)

	return nodes, edges


def to_dot(root_tensor) -> str:
	nodes, edges = build_graph(root_tensor)

	def tensor_label(t):
		val = f"{t.data.tolist()}"
		return f"Tensor(value={val}, grad={None if t.grad is None else t.grad.data.tolist()})"

	lines = ["digraph ComputationGraph {", "rankdir=TB;"]

	for n in nodes:
		lines.append(f'"{id(n)}" [label="{tensor_label(n)}"];')

	for p, c in edges:
		lines.append(f'"{id(p)}" -> "{id(c)}";')

	lines.append("}")
	return "\n".join(lines)


