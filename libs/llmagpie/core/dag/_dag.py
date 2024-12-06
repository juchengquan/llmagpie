from networkx import DiGraph, is_directed_acyclic_graph, recursive_simple_cycles, MultiDiGraph
from deprecated import deprecated

class SingleDAG(DiGraph):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def validate(self): #  -> "Self":
        self._validate_heads_and_tails()
        self._validate_nodes()

        # self._validate_edges_circular()  # FIXME: currently no circles are allowed
        # return self

    def _validate_nodes(self):
        # check node
        for n in self.nodes:
            self.nodes[n]["_obj"]._validate()

    def _validate_heads_and_tails(self):
        assert len(self.head_nodes) >= 1, "The graph has no head node!"
        assert len(self.tail_nodes) >= 1, "The graph has no end node!"
   

    def _validate_edges_circular(self):
        assert is_directed_acyclic_graph(self) is True, "Graph must be directed acyclic."
        assert recursive_simple_cycles(self) == [], "Graph must not include elementary circuits."

    @property
    def head_nodes(self):
        """Get root keys."""
        return [n for n, d in self.in_degree() if d == 0]

    @property
    def tail_nodes(self):
        """Get root keys."""
        return [n for n, d in self.out_degree() if d == 0]
