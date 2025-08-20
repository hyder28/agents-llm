from re import S
from dotenv import load_dotenv
load_dotenv()

from typing_extensions import TypedDict
from langgraph.graph.state import StateGraph, START

# --- Subgraph with a different schema -----------------------------------------
class SubgraphState(TypedDict):
    bar: str
    baz: str

def subgraph_node_1(state: SubgraphState):
    return {"baz": "baz"}

def subgraph_node_2(state: SubgraphState):
    return {"bar": state["bar"] + state["baz"]}

subgraph_builder = StateGraph(SubgraphState)
subgraph_builder.add_node(subgraph_node_1)
subgraph_builder.add_node(subgraph_node_2)
subgraph_builder.add_edge(START, "subgraph_node_1")
subgraph_builder.add_edge("subgraph_node_1", "subgraph_node_2")
subgraph = subgraph_builder.compile()

# --- Parent graph -----------------------------------------
class ParentState(TypedDict):
    foo: str

def node_1(state: ParentState):
    return {"foo": "hi!" + state["foo"]}

def node_2(state: ParentState):
    response = subgraph.invoke({"bar": state["foo"]})
    return {"foo": response["bar"]}

builder = StateGraph(ParentState)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
graph = builder.compile()

for chunk in graph.stream({"foo": "foo"}, subgraphs=True):
    print(chunk)
