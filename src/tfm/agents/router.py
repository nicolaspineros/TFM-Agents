"""
Router / Query Planner Agent con Tool Binding.

El Router es el agente principal que:
1. Recibe e interpreta la pregunta del usuario
2. Tiene acceso a las herramientas disponibles via bind_tools
3. Decide qué herramienta usar basándose en la pregunta
4. Ejecuta la herramienta y retorna el resultado

Este enfoque permite que el LLM DESCUBRA las herramientas disponibles
y decida dinámicamente cuál usar.
"""

from typing import Optional, Any, List, Dict
import json

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.prebuilt import ToolNode

from tfm.config.settings import get_settings
from tfm.schemas.state import TFMState, QueryPlan
from tfm.tools.analysis_tools import get_all_tools, get_tools_summary
from tfm.tools.preprocess import check_silver_status
from tfm.tools.features import get_gold_status


# System prompt para el Router con Tools
ROUTER_SYSTEM_PROMPT = """Eres un asistente de análisis de reseñas y ventas con acceso a herramientas especializadas.

Tu trabajo es:
1. Entender la pregunta del usuario
2. Seleccionar la herramienta correcta para responderla
3. Si no se disponen de los datos (silver o gold) contruir la capa necesaria.
4. Ejecutar la herramienta con los parámetros apropiados

CONTEXTO DE DATASETS:
- yelp: Reseñas de negocios en inglés. TIENE fechas y datos de usuarios/negocios.
- es: Reseñas de productos en español. NO tiene fechas.
- olist: E-commerce brasileño. TIENE fechas Y datos de ventas/órdenes.

REGLAS IMPORTANTES:
1. Si el usuario especificó un dataset, ÚSALO.
2. Para preguntas sobre ventas/órdenes, usa las herramientas de Olist.
3. Para análisis temporal (por mes), solo yelp y olist tienen fechas.
4. Si no estás seguro qué herramienta usar, usa get_dataset_status primero.

Siempre usa las herramientas disponibles. NO inventes datos."""


def create_router_agent_with_tools():
    """
    Crea el Router agent con herramientas bindeadas.
    
    El LLM puede ver todas las herramientas disponibles y decidir
    cuál usar basándose en la pregunta del usuario.
    
    Returns:
        LLM con tools bindeadas
    """
    settings = get_settings()
    
    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=0,
        api_key=settings.openai_api_key,
    )
    
    # Bindear todas las herramientas
    tools = get_all_tools()
    llm_with_tools = llm.bind_tools(tools)
    
    return llm_with_tools


def get_tool_node():
    """
    Crea el nodo de ejecución de herramientas.
    
    Este nodo ejecuta las herramientas que el LLM decide invocar.
    """
    tools = get_all_tools()
    return ToolNode(tools)


def route_query(state: TFMState) -> dict[str, Any]:
    """
    Nodo de grafo que ejecuta el routing.
    
    Prepara el contexto y genera el primer mensaje para el LLM.
    El LLM decidirá qué herramienta invocar.
    
    Args:
        state: Estado actual del grafo
        
    Returns:
        Dict con messages actualizados
    """
    user_query = state.get("user_query", "")
    current_dataset = state.get("current_dataset")
    messages = state.get("messages", [])
    
    print(f"[route_query] user_query='{user_query}', current_dataset='{current_dataset}'")
    
    # Validación básica
    if not user_query or len(user_query.strip()) < 3:
        return {
            "messages": messages + [
                AIMessage(content="Por favor proporciona una pregunta más específica.")
            ],
            "query_plan": {"is_valid": False, "rejection_reason": "Query muy corta"}
        }
    
    # Construir mensaje del usuario con contexto
    user_message_content = f"Pregunta: {user_query}"
    if current_dataset:
        user_message_content += f"\nDataset seleccionado: {current_dataset}"
    
    # Agregar contexto del estado de los datos
    try:
        silver_status = check_silver_status()
        status_lines = []
        for name, info in silver_status.items():
            exists = "EXISTE" if info.get("exists") else "NO EXISTE"
            status_lines.append(f"  {name}: {exists}")
        user_message_content += f"\n\nEstado de datos:\n" + "\n".join(status_lines)
    except Exception as e:
        user_message_content += f"\n\n(No se pudo verificar estado de datos: {e})"
    
    # Crear mensajes para el LLM
    new_messages = [
        SystemMessage(content=ROUTER_SYSTEM_PROMPT),
        HumanMessage(content=user_message_content)
    ]
    
    # Guardar query_plan básico para compatibilidad
    query_plan = {
        "is_valid": True,
        "datasets_required": [current_dataset] if current_dataset else [],
        "aggregations_needed": [],
        "needs_nlp_features": False,
        "needs_aggregation": True,
    }
    
    return {
        "messages": new_messages,
        "query_plan": query_plan
    }


def call_model(state: TFMState) -> dict[str, Any]:
    """
    Nodo que llama al modelo con las herramientas bindeadas.
    
    El modelo decidirá si invocar una herramienta o responder directamente.
    """
    messages = state.get("messages", [])
    
    print(f"[call_model] Llamando al LLM con {len(messages)} mensajes")
    
    llm = create_router_agent_with_tools()
    response = llm.invoke(messages)
    
    print(f"[call_model] Respuesta: {type(response).__name__}")
    if hasattr(response, 'tool_calls') and response.tool_calls:
        print(f"[call_model] Tool calls: {[tc['name'] for tc in response.tool_calls]}")
    
    return {"messages": messages + [response]}


def should_continue(state: TFMState) -> str:
    """
    Decide si continuar con herramientas o ir a síntesis.
    
    Returns:
        - "tools": Si hay tool_calls pendientes
        - "synthesize": Si el modelo ya respondió o no hay más herramientas
    """
    messages = state.get("messages", [])
    
    if not messages:
        return "synthesize"
    
    last_message = messages[-1]
    
    # Si el último mensaje tiene tool_calls, ejecutarlas
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        print(f"[should_continue] Hay tool_calls -> tools")
        return "tools"
    
    # Si es un ToolMessage, volver al modelo para que procese el resultado
    if isinstance(last_message, ToolMessage):
        print(f"[should_continue] ToolMessage -> model")
        return "model"
    
    print(f"[should_continue] Sin tool_calls -> synthesize")
    return "synthesize"


def extract_tool_results(state: TFMState) -> dict[str, Any]:
    """
    Extrae los resultados de las herramientas ejecutadas para el synthesizer.
    """
    messages = state.get("messages", [])
    
    # Buscar ToolMessages y extraer resultados
    tool_results = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            try:
                content = msg.content
                if isinstance(content, str):
                    result = json.loads(content)
                else:
                    result = content
                tool_results.append(result)
            except json.JSONDecodeError:
                tool_results.append({"raw": msg.content})
    
    if tool_results:
        # Guardar en aggregation_results para el synthesizer
        aggregation_results = {}
        for i, result in enumerate(tool_results):
            tool_name = result.get("tool", f"tool_{i}")
            aggregation_results[tool_name] = result
        
        # Crear last_result para compatibilidad
        last_result = {
            "query_type": "aggregation",
            "data_json": json.dumps(tool_results),
            "row_count": len(tool_results),
            "source_artifact": "tools",
        }
        
        return {
            "aggregation_results": aggregation_results,
            "last_result": last_result
        }
    
    return {}


# =============================================================================
# FUNCIONES DE COMPATIBILIDAD (para otros módulos que las usen)
# =============================================================================

def get_artifacts_status_summary() -> str:
    """
    Obtiene resumen del estado de artefactos para el prompt.
    """
    lines = []
    
    try:
        silver_status = check_silver_status()
        lines.append("SILVER LAYER:")
        for name, info in silver_status.items():
            status = "OK" if info.get("exists") else "NO EXISTE"
            size = f" ({info.get('size_mb', 0):.1f} MB)" if info.get("exists") else ""
            lines.append(f"  - {name}: {status}{size}")
    except Exception as e:
        lines.append(f"SILVER LAYER: Error verificando - {e}")
    
    try:
        gold_status = get_gold_status()
        lines.append("\nGOLD LAYER:")
        for name, info in gold_status.items():
            status = "OK" if info.get("exists") else "NO EXISTE"
            rows = f" ({info.get('rows', 0):,} filas)" if info.get("exists") else ""
            lines.append(f"  - {name}: {status}{rows}")
    except Exception as e:
        lines.append(f"GOLD LAYER: Error verificando - {e}")
    
    return "\n".join(lines)


def _check_missing_artifacts(
    datasets_required: List[str],
    needs_nlp_features: bool
) -> List[str]:
    """
    Verifica qué artefactos faltan.
    """
    missing = []
    
    try:
        # Verificar silver
        silver_status = check_silver_status()
        
        for dataset in datasets_required:
            # Mapear dataset a key de silver_status
            if dataset == "yelp":
                silver_keys = ["yelp_reviews"]
            elif dataset == "es":
                silver_keys = ["es"]
            elif dataset == "olist":
                silver_keys = ["olist_orders", "olist_reviews"]
            else:
                silver_keys = [f"{dataset}_reviews"]
            
            for key in silver_keys:
                if key in silver_status and not silver_status[key].get("exists", False):
                    missing.append(f"{dataset}_silver")
                    break
        
        # Verificar gold si se necesita
        if needs_nlp_features:
            gold_status = get_gold_status()
            for dataset in datasets_required:
                gold_key = f"{dataset}_features"
                if gold_key in gold_status and not gold_status[gold_key].get("exists", False):
                    missing.append(f"{dataset}_gold")
                    
    except Exception:
        # Si hay error verificando, marcar todo como potencialmente faltante
        for dataset in datasets_required:
            missing.append(f"{dataset}_silver")
    
    return list(set(missing))
