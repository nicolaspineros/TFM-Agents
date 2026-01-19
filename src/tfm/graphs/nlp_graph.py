"""
Grafo de NLP: Pipeline de procesamiento de reseñas.

Este grafo construye silver y gold layers:
1. load_bronze: Carga datos crudos
2. build_silver: Limpieza y normalización
3. compute_sentiment: Calcula sentimiento (baseline)
4. compute_aspects: Extrae aspectos
5. save_gold: Persiste features

El grafo puede ser invocado standalone o como subgrafo
del conversation_graph cuando se necesitan features.
"""

from typing import Any, Literal

from langgraph.graph import StateGraph, START, END

from tfm.schemas.state import NLPGraphState
from tfm.agents.nlp_worker import build_silver_node, build_gold_node


def build_nlp_graph(checkpointer: Any = None):
    """
    Construye el grafo de NLP.
    
    Args:
        checkpointer: Checkpointer opcional para persistencia
        
    Returns:
        Grafo compilado
        
    Example:
        >>> graph = build_nlp_graph()
        >>> result = graph.invoke({"dataset": "yelp"})
    """
    builder = StateGraph(NLPGraphState)
    
    # Nodos
    builder.add_node("validate_input", validate_input_node)
    builder.add_node("build_silver", build_silver_node)
    builder.add_node("build_gold", build_gold_node)
    builder.add_node("register_artifacts", register_artifacts_node)
    
    # Edges
    builder.add_edge(START, "validate_input")
    builder.add_conditional_edges(
        "validate_input",
        route_after_validation,
        {
            "build": "build_silver",
            "error": END,
        }
    )
    builder.add_edge("build_silver", "build_gold")
    builder.add_edge("build_gold", "register_artifacts")
    builder.add_edge("register_artifacts", END)
    
    return builder.compile(checkpointer=checkpointer)


def validate_input_node(state: NLPGraphState) -> dict[str, Any]:
    """
    Valida inputs del grafo NLP.
    
    Verifica:
    - Dataset valido
    - Bronze existe
    
    Args:
        state: Estado del grafo NLP (diccionario en runtime)
    
    Returns:
        Dict con bronze_path y/o errors
    """
    valid_datasets = ["yelp", "es", "olist"]
    
    # Acceso por keys - state es dict en runtime
    existing_errors = state.get("errors")
    errors = list(existing_errors) if existing_errors else []
    dataset = state.get("dataset", "")
    
    if dataset not in valid_datasets:
        errors.append(f"Dataset invalido: {dataset}. Validos: {valid_datasets}")
        return {"errors": errors}
    
    # Verificar bronze existe
    from tfm.tools.io_loaders import get_bronze_file_info
    try:
        info = get_bronze_file_info(dataset, "reviews")
        if not info["exists"]:
            errors.append(f"Bronze no encontrado: {info['path']}")
            return {"errors": errors}
        else:
            return {"bronze_path": info["path"]}
    except Exception as e:
        errors.append(f"Error verificando bronze: {e}")
        return {"errors": errors}


def route_after_validation(state: NLPGraphState) -> Literal["build", "error"]:
    """
    Decide si continuar o terminar con error.
    
    Args:
        state: Estado del grafo NLP (diccionario en runtime)
    """
    # Acceso por keys - state es dict en runtime
    errors = state.get("errors")
    if errors:
        return "error"
    return "build"


def register_artifacts_node(state: NLPGraphState) -> dict[str, Any]:
    """
    Registra artefactos generados en DuckDB.
    
    Returns:
        Dict vacio (no hay actualizaciones adicionales)
    """
    # Los artefactos ya fueron creados en build_silver y build_gold
    # Aqui podriamos registrarlos en DuckDB si fuera necesario
    return {}


def run_nlp_pipeline(
    dataset: str,
    overwrite: bool = False
) -> dict[str, Any]:
    """
    Helper para ejecutar pipeline NLP completo.
    
    Args:
        dataset: Dataset a procesar
        overwrite: Si sobrescribir artefactos existentes
        
    Returns:
        Dict con paths de artefactos generados
        
    Example:
        >>> result = run_nlp_pipeline("yelp")
        >>> print(result)
        {'silver_path': 'data/silver/yelp_reviews.parquet', ...}
    """
    initial_state = {
        "dataset": dataset,
        "messages": [],
        "errors": [],
        "features_computed": [],
    }
    
    final_state = nlp_graph.invoke(initial_state)
    
    return {
        "dataset": dataset,
        "silver_path": final_state.get("silver_path"),
        "gold_path": final_state.get("gold_path"),
        "features_computed": final_state.get("features_computed", []),
        "errors": final_state.get("errors", []),
    }


# Instancia global del grafo para langgraph.json
nlp_graph = build_nlp_graph()
