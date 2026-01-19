"""
Grafo Conversacional: Orquestacion principal del sistema.

Este es el grafo principal que maneja la conversacion:
1. router: Interpreta query y genera plan
2. nlp_worker: Construye features si necesario
3. aggregator: Ejecuta agregaciones
4. synthesizer: Genera insights
5. qa: Valida resultado

Flujo tipico:
START -> router -> [nlp/prediction si necesario] -> aggregator -> synthesizer -> qa -> END

Rutas condicionales:
- Si query invalida -> END con error
- Si features no existen -> nlp_worker primero
- Si prediccion requerida -> prediction_worker
- Si QA falla -> regenerar (con limite de intentos)
"""

from typing import Any, Literal
import json

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AIMessage

from tfm.schemas.state import TFMState, LastQueryResult
from tfm.agents.router import route_query
from tfm.agents.nlp_worker import process_nlp_request
from tfm.agents.insight_synthesizer import generate_insights
from tfm.agents.qa_evaluator import evaluate_result


def build_conversation_graph(checkpointer: Any = None):
    """
    Construye el grafo conversacional principal.
    
    Args:
        checkpointer: Checkpointer para persistencia de estado
        
    Returns:
        Grafo compilado
        
    Example:
        >>> graph = build_conversation_graph()
        >>> result = graph.invoke({
        ...     "user_query": "Cual es el sentimiento promedio en Yelp?"
        ... })
        >>> print(result["insights_report"]["summary"])
    """
    builder = StateGraph(TFMState)
    
    # Nodos
    builder.add_node("router", route_query)
    builder.add_node("nlp_worker", process_nlp_request)
    builder.add_node("aggregator", run_aggregations)
    builder.add_node("synthesizer", generate_insights)
    builder.add_node("qa", evaluate_result)
    builder.add_node("respond", generate_response)
    
    # Edge inicial
    builder.add_edge(START, "router")
    
    # Router decide siguiente paso
    builder.add_conditional_edges(
        "router",
        route_after_planning,
        {
            "needs_features": "nlp_worker",
            "ready": "aggregator",
            "invalid": END,
        }
    )
    
    # Despues de NLP worker, ir a agregaciones
    builder.add_edge("nlp_worker", "aggregator")
    
    # Despues de agregaciones, sintetizar
    builder.add_edge("aggregator", "synthesizer")
    
    # Despues de sintesis, QA
    builder.add_edge("synthesizer", "qa")
    
    # QA decide si terminar o regenerar
    builder.add_conditional_edges(
        "qa",
        route_after_qa,
        {
            "pass": "respond",
            "regenerate": "synthesizer",
            "fail": "respond",
        }
    )
    
    # Respuesta final al usuario
    builder.add_edge("respond", END)
    
    return builder.compile(checkpointer=checkpointer)


def route_after_planning(state: TFMState) -> Literal["needs_features", "ready", "invalid"]:
    """
    Decide siguiente paso despues del routing.
    
    ORDEN DE VERIFICACION:
    1. Primero verificar si hay datasets requeridos
    2. Luego verificar si silver existe (construir si no)
    3. Finalmente verificar is_valid (solo si todo lo anterior OK)
    
    Returns:
        - "needs_features": Si hay que construir silver/gold primero
        - "ready": Si podemos ir directo a agregaciones
        - "invalid": Si la query no es valida
    """
    # Acceso por keys - state es dict en runtime
    query_plan = state.get("query_plan")
    
    if not query_plan:
        print("[route_after_planning] No hay query_plan -> invalid")
        return "invalid"
    
    datasets_required = query_plan.get("datasets_required", [])
    aggregations_needed = query_plan.get("aggregations_needed", [])
    is_valid = query_plan.get("is_valid", True)
    rejection_reason = query_plan.get("rejection_reason")
    
    print(f"[route_after_planning] datasets={datasets_required}, aggs={aggregations_needed}, is_valid={is_valid}")
    if rejection_reason:
        print(f"[route_after_planning] rejection_reason={rejection_reason}")
    
    # Si no hay datasets requeridos, es invalido
    if not datasets_required:
        print("[route_after_planning] No hay datasets_required -> invalid")
        return "invalid"
    
    # Validar que los datasets son conocidos
    valid_datasets = ["yelp", "es", "olist"]
    for ds in datasets_required:
        if ds not in valid_datasets:
            print(f"[route_after_planning] Dataset desconocido: {ds} -> invalid")
            return "invalid"
    
    # PRIMERO: Verificar si silver existe para los datasets requeridos
    # Si no existe, ir a nlp_worker para construirlo
    try:
        from tfm.tools.preprocess import check_silver_status
        silver_status = check_silver_status()
        
        for dataset in datasets_required:
            if dataset == "yelp":
                key = "yelp_reviews"
            elif dataset == "es":
                key = "es"
            elif dataset == "olist":
                # Verificar orders y reviews
                for olist_key in ["olist_orders", "olist_reviews"]:
                    if olist_key in silver_status and not silver_status[olist_key].get("exists", False):
                        print(f"[route_after_planning] Silver faltante: {olist_key} -> needs_features")
                        return "needs_features"
                continue
            else:
                key = f"{dataset}_reviews"
            
            if key in silver_status and not silver_status[key].get("exists", False):
                print(f"[route_after_planning] Silver faltante: {key} -> needs_features")
                return "needs_features"
                
    except Exception as e:
        print(f"[route_after_planning] Error verificando silver: {e} -> needs_features")
        return "needs_features"
    
    # Verificar gold si necesita features NLP
    if query_plan.get("needs_nlp_features", False):
        try:
            from tfm.tools.features import get_gold_status
            gold_status = get_gold_status()
            
            for dataset in datasets_required:
                gold_key = f"{dataset}_features"
                if gold_key in gold_status and not gold_status[gold_key].get("exists", False):
                    print(f"[route_after_planning] Gold faltante: {gold_key} -> needs_features")
                    return "needs_features"
        except Exception as e:
            print(f"[route_after_planning] Error verificando gold: {e} -> needs_features")
            return "needs_features"
    
    # ULTIMO: Verificar is_valid
    # PERO: Si tenemos datasets validos y agregaciones, ignorar is_valid=False del LLM
    if not is_valid:
        # Verificar si realmente deberia ser invalido
        # Solo rechazar si es un caso claro de guardrail
        if rejection_reason:
            rejection_lower = rejection_reason.lower()
            # Solo rechazar por temporal en ES o ventas en no-Olist
            if "temporal" in rejection_lower and "es" in datasets_required:
                print(f"[route_after_planning] Guardrail valido: temporal en ES -> invalid")
                return "invalid"
            if "ventas" in rejection_lower and "olist" not in datasets_required:
                print(f"[route_after_planning] Guardrail valido: ventas sin Olist -> invalid")
                return "invalid"
        
        # Si el LLM marco is_valid=False pero tenemos datasets y agregaciones validas,
        # continuar de todos modos (el LLM se equivoco)
        if aggregations_needed and datasets_required:
            print(f"[route_after_planning] WARN: LLM marco is_valid=False pero hay datos validos, continuando...")
        else:
            print(f"[route_after_planning] is_valid=False sin datos validos -> invalid")
            return "invalid"
    
    print(f"[route_after_planning] Todo OK -> ready")
    return "ready"


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
    
    # Por ahora, no regenerar - terminar
    return "fail"


def generate_response(state: TFMState) -> dict[str, Any]:
    """
    Nodo final que genera la respuesta para el usuario.
    
    Convierte el insights_report en un mensaje que aparece en el Chat.
    
    Returns:
        Dict con messages para agregar al historial
    """
    insights_report = state.get("insights_report")
    qa_passed = state.get("qa_passed", False)
    qa_feedback = state.get("qa_feedback")
    query_plan = state.get("query_plan", {})
    
    # Construir respuesta en texto
    if not insights_report:
        # Si no hay reporte, generar mensaje de error
        rejection_reason = query_plan.get("rejection_reason") if query_plan else None
        if rejection_reason:
            response_text = f"No pude procesar tu consulta: {rejection_reason}"
        else:
            response_text = "No se pudo generar una respuesta. Por favor intenta reformular tu pregunta."
    else:
        # Construir respuesta estructurada
        summary = insights_report.get("summary", "Analisis completado.")
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
    
    return {"messages": [response_message]}


def run_aggregations(state: TFMState) -> dict[str, Any]:
    """
    Nodo que ejecuta agregaciones segun el plan.
    
    Lee query_plan y ejecuta las agregaciones necesarias.
    Guarda resultado en last_result.
    
    Returns:
        Dict con last_result y aggregation_results
    """
    print("[run_aggregations] Iniciando...")
    
    # Acceso por keys - state es dict en runtime
    query_plan = state.get("query_plan")
    
    if not query_plan:
        print("[run_aggregations] ERROR: No hay query_plan")
        return {"error": "No hay plan de ejecucion"}
    
    datasets = query_plan.get("datasets_required", [])
    aggs = query_plan.get("aggregations_needed", ["reviews_by_stars"])
    print(f"[run_aggregations] datasets={datasets}, aggregations={aggs}")
    
    from tfm.tools.aggregations import (
        aggregate_reviews_by_stars,
        aggregate_reviews_by_month,
        aggregate_yelp_user_stats,
        aggregate_business_stats,
        aggregate_ambiguous_reviews,
        aggregate_olist_sales_by_month,
    )
    
    # Mapeo de tipos de agregacion a funciones
    agg_functions = {
        "reviews_by_stars": aggregate_reviews_by_stars,
        "reviews_by_month": aggregate_reviews_by_month,
        "user_stats": lambda ds: aggregate_yelp_user_stats(),
        "business_stats": lambda ds: aggregate_business_stats(),
        "ambiguous_reviews": aggregate_ambiguous_reviews,
        "olist_sales": lambda ds: aggregate_olist_sales_by_month(),
    }
    
    results = []
    # Acceso seguro a aggregation_results
    existing_agg_results = state.get("aggregation_results")
    aggregation_results = dict(existing_agg_results) if existing_agg_results else {}
    
    # Ejecutar agregaciones solicitadas - query_plan es dict
    aggregations = query_plan.get("aggregations_needed") or ["reviews_by_stars"]
    datasets_required = query_plan.get("datasets_required", [])
    
    for dataset in datasets_required:
        for agg_type in aggregations:
            if agg_type in agg_functions:
                try:
                    result = agg_functions[agg_type](dataset)
                    if "error" not in result:
                        results.append({
                            "type": agg_type,
                            "dataset": dataset,
                            "data": result,
                        })
                        aggregation_results[f"{dataset}_{agg_type}"] = result
                except Exception:
                    pass  # Ignorar errores de agregacion individual
    
    # Construir last_result como dict para serializar correctamente
    if results:
        last_result = {
            "query_type": "aggregation",
            "data_json": json.dumps(results),
            "columns": ["type", "dataset", "data"],
            "row_count": len(results),
            "source_artifact": "aggregations",
        }
    else:
        last_result = {
            "query_type": "aggregation",
            "data_json": json.dumps([]),
            "columns": [],
            "row_count": 0,
            "source_artifact": "no_results",
        }
    
    return {
        "last_result": last_result,
        "aggregation_results": aggregation_results,
    }


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
        >>> result = ask("Cual es el sentimiento promedio de restaurantes?")
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
    
    if final_state.get("qa_feedback"):
        response["qa_feedback"] = final_state["qa_feedback"]
    
    return response


# Instancia global para langgraph.json
conversation_graph = build_conversation_graph()
