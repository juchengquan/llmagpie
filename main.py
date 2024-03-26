
from src._enum import NodeType

from src.node_dummy import DummyNode
from src.node_api import NodeAPICallback
from fastapi import APIRouter
# from src.dag import DAG
# TODO: bind DAG for only one time

# instantiate
# a = NodeAPICallback()
a = DummyNode(node_type=NodeType.StartNode)
b = DummyNode(node_type=NodeType.MiddleNode)
c = DummyNode(node_type=NodeType.MiddleNode)
d = DummyNode(node_type=NodeType.EndNode)

# case 1
g = a >> [b, c] >> d

for ele in [a,b,c,d]:
    print(ele.id)

print("====")
print(type(g))
try:
    print(g[0].id)
except:
    print(g.id)
print(d)

print(g.graph)

h1 = APIRouter()
h2 = APIRouter()
# TODO
# g.graph.validate_head(router=h1, router_cb=h2)


exit(0)
