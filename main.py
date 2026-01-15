"""
Entry point principal para langgraph dev.

Este archivo es referenciado por langgraph.json y expone los grafos
para desarrollo y testing con LangGraph Studio.

Uso:
    uv run langgraph dev
"""

# Re-exportar grafos para langgraph.json
from tfm.graphs.conversation_graph import conversation_graph
from tfm.graphs.nlp_graph import nlp_graph
from tfm.graphs.prediction_graph import prediction_graph
from tfm.graphs.evaluation_graph import evaluation_graph

# Alias para compatibilidad con langgraph.json
agent = conversation_graph

__all__ = [
    "conversation_graph",
    "nlp_graph",
    "prediction_graph",
    "evaluation_graph",
    "agent",
]
