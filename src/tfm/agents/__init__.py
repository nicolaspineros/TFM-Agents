"""
Agentes LLM para el sistema TFM.

Los agentes usan LLMs para tareas de razonamiento:
- Router: planificación y routing de queries
- NLP Worker: coordina ejecución de tools NLP
- Insight Synthesizer: genera reportes estructurados
- QA Evaluator: valida resultados y faithfulness

Principios de diseño:
- Agentes pueden importar tools y schemas
- Agentes NO importan graphs (evitar ciclos)
- Agentes usan LLM solo para razonamiento, no para cálculos masivos
- Cada agente tiene una responsabilidad específica
"""

from tfm.agents.router import create_router_agent, route_query
from tfm.agents.nlp_worker import create_nlp_worker, process_nlp_request
from tfm.agents.insight_synthesizer import create_synthesizer, generate_insights
from tfm.agents.qa_evaluator import create_qa_evaluator, evaluate_result

__all__ = [
    "create_router_agent",
    "route_query",
    "create_nlp_worker",
    "process_nlp_request",
    "create_synthesizer",
    "generate_insights",
    "create_qa_evaluator",
    "evaluate_result",
]
