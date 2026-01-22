"""
Grafos LangGraph para el sistema TFM.

El grafo principal orquesta el flujo conversacional:
- conversation_graph: Router -> Tools -> Synthesizer -> QA -> Response

Principios de diseno:
- El grafo puede importar agents y tools
- Usa TFMState para compartir estado entre nodos
- Usa rutas condicionales para decisiones dinamicas
"""

from tfm.graphs.conversation_graph import conversation_graph, build_conversation_graph, ask

__all__ = [
    "conversation_graph",
    "build_conversation_graph",
    "ask",
]
