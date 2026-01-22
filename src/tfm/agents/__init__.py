"""
Agentes LLM para el sistema TFM.

Los agentes usan LLMs para tareas de razonamiento:
- Router: planificación y routing de queries con tool binding
- Insight Synthesizer: genera reportes estructurados
- QA Evaluator: valida resultados y faithfulness

Principios de diseño:
- Agentes pueden importar tools y schemas
- Agentes NO importan graphs (evitar ciclos)
- Agentes usan LLM solo para razonamiento, no para cálculos masivos
- Cada agente tiene una responsabilidad específica
"""

from tfm.agents.router import (
    create_router_agent_with_tools,
    route_query,
    call_model,
    should_continue,
    extract_tool_results,
    get_tool_node,
)
from tfm.agents.insight_synthesizer import create_synthesizer, generate_insights
from tfm.agents.qa_evaluator import create_qa_evaluator, evaluate_result

__all__ = [
    # Router
    "create_router_agent_with_tools",
    "route_query",
    "call_model",
    "should_continue",
    "extract_tool_results",
    "get_tool_node",
    # Synthesizer
    "create_synthesizer",
    "generate_insights",
    # QA
    "create_qa_evaluator",
    "evaluate_result",
]
