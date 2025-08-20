from dotenv import load_dotenv
load_dotenv()

from typing_extensions import TypedDict
from langgraph.graph.state import StateGraph, START

class SubgraphState(TypedDict):
    foo: str
    bar: str

def subgraph_node_1(state: SubgraphState):
    return {"bar": "bar"}

def subgraph_node_2(state: SubgraphState):
    return {"foo": state["foo"] + state["bar"]}

subgraph_builder = StateGraph(SubgraphState)
subgraph_builder.add_node(subgraph_node_1)
subgraph_builder.add_node(subgraph_node_2)
subgraph_builder.add_edge(START, "subgraph_node_1")
subgraph_builder.add_edge("subgraph_node_1", "subgraph_node_2")

subgraph = subgraph_builder.compile()

class ParentState(TypedDict):
    foo: str

def node_1(state: ParentState):
    return {"foo": 'hi! ' + state["foo"]}

builder = StateGraph(ParentState)
builder.add_node("node_1", node_1)
builder.add_node("node_2", subgraph)
builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
graph = builder.compile()

# png_graph_data = graph.get_graph().draw_mermaid_png()
# with open("subgraphs.png", "wb") as f:
#     f.write(png_graph_data)

for chunk in graph.stream({"foo": "foo"}):
    print(chunk)
