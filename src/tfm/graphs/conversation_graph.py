"""
Grafo Conversacional: Orquestacion principal del sistema.

Este es el grafo principal que maneja la conversación usando herramientas
que el LLM puede invocar dinámicamente.

Flujo:
START -> route_query -> call_model -> [should_continue_routing]
                                        ├── "tools" → ToolNode (ejecuta herramienta) → call_model (loop)
                                        └── "synthesize" → extract_results → synthesizer → qa → respond → END

El LLM decide qué herramienta usar basándose en la pregunta del usuario.
Las herramientas se ejecutan y los resultados vuelven al LLM para síntesis.
"""

from typing import Any, Literal
import json

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage

from tfm.schemas.state import TFMState
from tfm.agents.router import (
    route_query, 
    call_model, 
    should_continue, 
    extract_tool_results
)
from tfm.agents.insight_synthesizer import generate_insights
from tfm.agents.qa_evaluator import evaluate_result
from tfm.tools.analysis_tools import get_all_tools


def build_conversation_graph(checkpointer: Any = None):
    """
    Construye el grafo conversacional con tool binding.
    
    Args:
        checkpointer: Checkpointer para persistencia de estado
        
    Returns:
        Grafo compilado
        
    Example:
        >>> graph = build_conversation_graph()
        >>> result = graph.invoke({
        ...     "user_query": "Cual es la distribución de ratings?",
        ...     "current_dataset": "olist"
        ... })
    """
    builder = StateGraph(TFMState)
    
    # Nodo de herramientas
    tools = get_all_tools()
    tool_node = ToolNode(tools)
    
    # Nodos
    builder.add_node("route_query", route_query)
    builder.add_node("call_model", call_model)
    builder.add_node("tools", tool_node)
    builder.add_node("extract_results", extract_tool_results)
    builder.add_node("synthesizer", generate_insights)
    builder.add_node("qa", evaluate_result)
    builder.add_node("respond", generate_response)
    
    # Flujo inicial
    builder.add_edge(START, "route_query")
    builder.add_edge("route_query", "call_model")
    
    # Después del modelo, decidir si usar herramientas o sintetizar
    builder.add_conditional_edges(
        "call_model",
        should_continue_routing,
        {
            "tools": "tools",
            "model": "call_model",
            "synthesize": "extract_results",
        }
    )
    
    # Después de ejecutar herramientas, volver al modelo
    builder.add_edge("tools", "call_model")
    
    # Después de extraer resultados, sintetizar
    builder.add_edge("extract_results", "synthesizer")
    
    # Después de síntesis, QA
    builder.add_edge("synthesizer", "qa")
    
    # QA decide si terminar
    builder.add_conditional_edges(
        "qa",
        route_after_qa,
        {
            "pass": "respond",
            "regenerate": "synthesizer",
            "fail": "respond",
        }
    )
    
    # Respuesta final
    builder.add_edge("respond", END)
    
    return builder.compile(checkpointer=checkpointer)


def should_continue_routing(state: TFMState) -> Literal["tools", "model", "synthesize"]:
    """
    Decide el siguiente paso después de llamar al modelo.
    
    Returns:
        - "tools": Si hay tool_calls pendientes
        - "model": Si hay que volver al modelo (después de ToolMessage)
        - "synthesize": Si el modelo ya respondió
    """
    messages = state.get("messages", [])
    
    if not messages:
        return "synthesize"
    
    last_message = messages[-1]
    
    # Si el último mensaje tiene tool_calls, ejecutarlas
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        print(f"[should_continue_routing] Tool calls detectados -> tools")
        return "tools"
    
    # Si es un ToolMessage, volver al modelo para procesar
    if isinstance(last_message, ToolMessage):
        # Verificar si ya procesamos suficientes herramientas
        tool_count = sum(1 for m in messages if isinstance(m, ToolMessage))
        if tool_count >= 3:  # Límite de herramientas por query
            print(f"[should_continue_routing] Límite de herramientas alcanzado -> synthesize")
            return "synthesize"
        print(f"[should_continue_routing] ToolMessage -> model")
        return "model"
    
    # Si es AIMessage sin tool_calls, ir a síntesis
    if isinstance(last_message, AIMessage):
        print(f"[should_continue_routing] AIMessage sin tools -> synthesize")
        return "synthesize"
    
    return "synthesize"


def route_after_qa(state: TFMState) -> Literal["pass", "regenerate", "fail"]:
    """
    Decide si aceptar resultado o regenerar.
    
    Returns:
        - "pass": QA aprobado, terminar
        - "regenerate": Intentar regenerar (limite de intentos)
        - "fail": Demasiados intentos, terminar con error
    """
    # Acceso por keys - state es dict en runtime
    if state.get("qa_passed", False):
        return "pass"
    return "fail"


def generate_response(state: TFMState) -> dict[str, Any]:
    """
    Nodo final que genera la respuesta para el usuario.
    
    Convierte el insights_report en un mensaje que aparece en el Chat.
    """
    insights_report = state.get("insights_report")
    query_plan = state.get("query_plan", {})
    messages = state.get("messages", [])
    
    # Construir respuesta en texto
    if not insights_report:
        # Verificar si hay error en el plan
        rejection_reason = query_plan.get("rejection_reason") if query_plan else None
        if rejection_reason:
            response_text = f"No pude procesar tu consulta: {rejection_reason}"
        else:
            # Buscar si el modelo ya respondió directamente
            for msg in reversed(messages):
                if isinstance(msg, AIMessage) and msg.content and not hasattr(msg, 'tool_calls'):
                    response_text = msg.content
                    break
            else:
                response_text = "No se pudo generar una respuesta. Por favor intenta reformular tu pregunta."
    else:
        # Construir respuesta estructurada
        summary = insights_report.get("summary", "Análisis completado.")
        bullets = insights_report.get("bullets", [])
        caveats = insights_report.get("caveats", [])
        
        # Formato de respuesta
        parts = [summary]
        
        # Agregar bullets si hay
        if bullets:
            parts.append("\n\n**Hallazgos:**")
            for bullet in bullets:
                if isinstance(bullet, dict):
                    text = bullet.get("text", "")
                    evidence = bullet.get("evidence", "")
                    if evidence:
                        parts.append(f"- {text} ({evidence})")
                    else:
                        parts.append(f"- {text}")
                else:
                    parts.append(f"- {bullet}")
        
        # Agregar caveats si hay
        if caveats:
            parts.append("\n\n**Nota:**")
            for caveat in caveats:
                parts.append(f"- {caveat}")
        
        response_text = "\n".join(parts)
    
    # Crear mensaje de respuesta
    response_message = AIMessage(content=response_text)
    
    return {"messages": messages + [response_message]}


def ask(
    query: str,
    dataset: str = None,
    config: dict = None
) -> dict[str, Any]:
    """
    Helper para hacer preguntas al sistema.
    
    Args:
        query: Pregunta en lenguaje natural
        dataset: Dataset preferido (opcional)
        config: Configuracion adicional
        
    Returns:
        Dict con respuesta estructurada
        
    Example:
        >>> result = ask("Cual es la distribución de ratings?", dataset="olist")
        >>> print(result["summary"])
    """
    initial_state = {
        "user_query": query,
        "current_dataset": dataset,
        "messages": [],
        "artifacts": {},
        "aggregation_results": {},
    }
    
    final_state = conversation_graph.invoke(initial_state, config=config)
    
    # Extraer respuesta
    response = {
        "query": query,
        "success": final_state.get("qa_passed", False),
    }
    
    if final_state.get("insights_report"):
        response["summary"] = final_state["insights_report"].get("summary", "")
        response["bullets"] = final_state["insights_report"].get("bullets", [])
        response["caveats"] = final_state["insights_report"].get("caveats", [])
    
    if final_state.get("error"):
        response["error"] = final_state["error"]
    
    # Extraer respuesta del último mensaje
    messages = final_state.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            response["response"] = msg.content
            break
    
    return response


# Instancia global para langgraph.json
conversation_graph = build_conversation_graph()
