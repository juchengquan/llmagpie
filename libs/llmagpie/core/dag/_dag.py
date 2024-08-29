from networkx import DiGraph, is_directed_acyclic_graph, recursive_simple_cycles
import warnings
# from deprecated import deprecated

class SingleDAG(DiGraph):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def validate(self):
        self._validate_nodes()
        self._validate_edges()
        self._validate_roots()
        return self
        
    def _validate_nodes(self):
        for n in self.nodes:
            self.nodes[n]["_obj"]()._validate()

    def _validate_edges(self):
        try:
            assert is_directed_acyclic_graph(self), "Graph is not directed acyclic."
            assert recursive_simple_cycles(self) == [], "Graph has loop(s)."
        except AssertionError as err:
            warnings.warn(str(err))
            raise AssertionError(err)
    
    def _validate_roots(self):
        assert len(self.root_nodes) == 1, "SingleDAG must have only one root node"
    
    @property
    def root_nodes(self):
        return [n for n, d in self.in_degree() if d == 0]
    
    @property
    def leave_nodes(self):
        return [n for n, d in self.in_degree() if d == 0]
