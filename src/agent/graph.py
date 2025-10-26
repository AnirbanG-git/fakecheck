from __future__ import annotations
from langgraph.graph import StateGraph, END
from .state import AgentState
from .nodes import RetrieveNode, VerifyNode, ExplainNode

def build_graph(retrieve_node: RetrieveNode,
                verify_node: VerifyNode,
                explain_node: ExplainNode):
    g = StateGraph(AgentState)

    g.add_node("retrieve", retrieve_node)
    g.add_node("verify", verify_node)
    g.add_node("explain", explain_node)

    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "verify")
    g.add_edge("verify", "explain")
    g.add_edge("explain", END)

    return g.compile()
