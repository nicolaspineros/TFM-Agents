"""
Grafos LangGraph para el sistema TFM.

Los grafos orquestan el flujo de agentes y tools:
- nlp_graph: Pipeline de NLP (bronze → silver → gold)
- prediction_graph: Pipeline de predicción de ventas
- conversation_graph: Flujo conversacional principal
- evaluation_graph: Evaluación offline

Principios de diseño:
- Graphs pueden importar agents y tools
- Cada graph tiene su State (puede heredar de TFMState)
- Graphs usan rutas condicionales para decisiones
"""

from tfm.graphs.nlp_graph import nlp_graph, build_nlp_graph
from tfm.graphs.prediction_graph import prediction_graph, build_prediction_graph
from tfm.graphs.conversation_graph import conversation_graph, build_conversation_graph
from tfm.graphs.evaluation_graph import evaluation_graph, build_evaluation_graph

__all__ = [
    "nlp_graph",
    "build_nlp_graph",
    "prediction_graph",
    "build_prediction_graph",
    "conversation_graph",
    "build_conversation_graph",
    "evaluation_graph",
    "build_evaluation_graph",
]
