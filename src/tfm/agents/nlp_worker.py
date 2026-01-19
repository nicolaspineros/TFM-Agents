"""
NLP/ML Worker Agent.

El NLP Worker coordina la ejecucion de tools de NLP:
- Construccion de silver layer
- Calculo de features gold (sentimiento, aspectos)
- Procesamiento batch de reviews

Este agente NO usa LLM para calculos; usa LLM solo para:
- Decidir orden de operaciones
- Manejar errores y reintentos
- Reportar progreso
"""

from typing import Optional, Any, Dict, List
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

from tfm.config.settings import get_settings, SILVER_FILES, GOLD_FILES
from tfm.schemas.state import TFMState, NLPGraphState, ArtifactInfo
from tfm.tools.preprocess import (
    build_silver_yelp, build_silver_yelp_users, build_silver_yelp_business,
    build_silver_es, build_silver_olist, check_silver_status
)
from tfm.tools.features import build_gold_features, get_gold_status
from tfm.tools.aggregations import (
    aggregate_reviews_by_stars, aggregate_reviews_by_month,
    aggregate_yelp_user_stats, aggregate_business_stats,
    aggregate_ambiguous_reviews, aggregate_olist_sales_by_month,
    aggregate_olist_reviews_sales
)


def create_nlp_worker():
    """
    Crea instancia del NLP Worker.
    
    Returns:
        LLM configurado con tools
    """
    settings = get_settings()
    
    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=0,
        api_key=settings.openai_api_key,
    )
    
    return llm


# ============================================================================
# TOOLS para el NLP Worker (decorados con @tool para LangGraph)
# ============================================================================

@tool
def ensure_silver_layer(dataset: str) -> Dict[str, Any]:
    """
    Asegura que exista el silver layer para un dataset.
    
    Args:
        dataset: Nombre del dataset (yelp, es, olist)
        
    Returns:
        Dict con status y paths
    """
    return _ensure_silver(dataset)


@tool  
def ensure_gold_features(dataset: str) -> Dict[str, Any]:
    """
    Asegura que existan las features gold para un dataset.
    
    Args:
        dataset: Nombre del dataset (yelp, es, olist)
        
    Returns:
        Dict con status y path
    """
    result = _ensure_gold(dataset)
    if result:
        return {"success": True, "path": str(result)}
    return {"success": False, "error": "No se pudo construir gold"}


@tool
def run_aggregation(aggregation_type: str, dataset: str) -> Dict[str, Any]:
    """
    Ejecuta una agregacion sobre los datos.
    
    Args:
        aggregation_type: Tipo de agregacion
        dataset: Dataset a agregar
        
    Returns:
        Resultado de la agregacion
    """
    return _run_aggregation(aggregation_type, dataset)


def get_nlp_worker_tools() -> List:
    """
    Retorna lista de tools disponibles para el NLP Worker.
    """
    return [ensure_silver_layer, ensure_gold_features, run_aggregation]


def process_nlp_request(state: TFMState) -> dict[str, Any]:
    """
    Nodo que procesa requests de NLP.
    
    Lee el QueryPlan y ejecuta las tools necesarias:
    1. Construir silver si no existe
    2. Construir gold features si no existe
    3. Ejecutar agregaciones solicitadas
    4. Retornar actualizaciones del estado
    
    Args:
        state: Estado con query_plan (diccionario en runtime)
        
    Returns:
        Dict con artifacts y aggregation_results para actualizar estado
    """
    # Acceso por keys - state es dict en runtime
    query_plan = state.get("query_plan")
    
    if not query_plan:
        return {"error": "No hay query_plan para procesar"}
    
    # Inicializar resultados - acceso seguro
    existing_artifacts = state.get("artifacts")
    artifacts = dict(existing_artifacts) if existing_artifacts else {}
    aggregation_results = {}
    
    # query_plan es dict, acceder con get()
    guardrail_warnings = query_plan.get("guardrail_warnings") or []
    warnings = list(guardrail_warnings)
    datasets_required = query_plan.get("datasets_required", [])
    needs_nlp_features = query_plan.get("needs_nlp_features", False)
    needs_aggregation = query_plan.get("needs_aggregation", False)
    aggregations_needed = query_plan.get("aggregations_needed", [])
    
    # Procesar cada dataset requerido
    for dataset in datasets_required:
        try:
            # Construir silver si necesario
            silver_result = _ensure_silver(dataset)
            if silver_result.get("success"):
                for name, path in silver_result.get("paths", {}).items():
                    # Guardar como dict para serializacion
                    artifacts[f"{dataset}_{name}"] = {
                        "path": path,
                        "dataset": dataset,
                        "artifact_type": "silver",
                    }
            
            # Construir gold si necesario
            if needs_nlp_features:
                gold_path = _ensure_gold(dataset)
                if gold_path:
                    artifacts[f"{dataset}_gold"] = {
                        "path": str(gold_path),
                        "dataset": dataset,
                        "artifact_type": "gold",
                    }
        
        except Exception as e:
            warnings.append(f"Error procesando {dataset}: {str(e)}")
    
    # Ejecutar agregaciones si necesario
    if needs_aggregation and aggregations_needed:
        for agg_type in aggregations_needed:
            for dataset in datasets_required:
                try:
                    result = _run_aggregation(agg_type, dataset)
                    if "error" not in result:
                        aggregation_results[f"{dataset}_{agg_type}"] = result
                except Exception as e:
                    warnings.append(f"Error en agregacion {agg_type}: {str(e)}")
    
    # Construir actualizacion del estado
    update = {
        "artifacts": artifacts,
    }
    
    if aggregation_results:
        update["aggregation_results"] = aggregation_results
    
    if warnings:
        # Actualizar query_plan con warnings - crear nuevo dict
        updated_plan = dict(query_plan)
        updated_plan["guardrail_warnings"] = warnings
        update["query_plan"] = updated_plan
    
    return update


def _ensure_silver(dataset: str) -> Dict[str, Any]:
    """
    Asegura que exista silver layer para dataset.
    
    Args:
        dataset: Dataset a procesar
        
    Returns:
        Dict con status y paths
    """
    settings = get_settings()
    result = {"success": False, "paths": {}}
    
    if dataset == "yelp":
        # Reviews
        reviews_path = settings.silver_dir / SILVER_FILES["yelp_reviews"]
        if not reviews_path.exists():
            try:
                reviews_path = build_silver_yelp()
            except Exception:
                pass
        if reviews_path.exists():
            result["paths"]["reviews"] = str(reviews_path)
        
        # Users
        users_path = settings.silver_dir / SILVER_FILES["yelp_users"]
        if not users_path.exists():
            try:
                users_path = build_silver_yelp_users()
            except Exception:
                pass
        if users_path.exists():
            result["paths"]["users"] = str(users_path)
        
        # Business
        business_path = settings.silver_dir / SILVER_FILES["yelp_business"]
        if not business_path.exists():
            try:
                business_path = build_silver_yelp_business()
            except Exception:
                pass
        if business_path.exists():
            result["paths"]["business"] = str(business_path)
        
        result["success"] = len(result["paths"]) > 0
    
    elif dataset == "es":
        silver_path = settings.silver_dir / SILVER_FILES["es"]
        if not silver_path.exists():
            try:
                silver_path = build_silver_es()
            except Exception:
                pass
        if silver_path.exists():
            result["paths"]["reviews"] = str(silver_path)
            result["success"] = True
    
    elif dataset == "olist":
        orders_path = settings.silver_dir / SILVER_FILES["olist_orders"]
        reviews_path = settings.silver_dir / SILVER_FILES["olist_reviews"]
        
        if not orders_path.exists() or not reviews_path.exists():
            try:
                orders_path, reviews_path = build_silver_olist()
            except Exception:
                pass
        
        if orders_path.exists():
            result["paths"]["orders"] = str(orders_path)
        if reviews_path.exists():
            result["paths"]["reviews"] = str(reviews_path)
        result["success"] = len(result["paths"]) > 0
    
    return result


def _ensure_gold(dataset: str) -> Optional[Path]:
    """
    Asegura que existan gold features para dataset.
    
    Args:
        dataset: Dataset a procesar
        
    Returns:
        Path al gold o None si falla
    """
    settings = get_settings()
    
    gold_files = {
        "yelp": GOLD_FILES["yelp_features"],
        "es": GOLD_FILES["es_features"],
        "olist": GOLD_FILES["olist_features"],
    }
    
    if dataset not in gold_files:
        return None
    
    gold_path = settings.gold_dir / gold_files[dataset]
    if gold_path.exists():
        return gold_path
    
    try:
        return build_gold_features(dataset)
    except Exception:
        return None


def _run_aggregation(aggregation_type: str, dataset: str) -> Dict[str, Any]:
    """
    Ejecuta una agregacion especifica.
    
    Args:
        aggregation_type: Tipo de agregacion a ejecutar
        dataset: Dataset sobre el que agregar
        
    Returns:
        Resultado de la agregacion
    """
    aggregation_map = {
        "reviews_by_stars": lambda ds: aggregate_reviews_by_stars(ds),
        "reviews_by_month": lambda ds: aggregate_reviews_by_month(ds),
        "user_stats": lambda ds: aggregate_yelp_user_stats() if ds == "yelp" else {"error": "Solo disponible para yelp"},
        "business_stats": lambda ds: aggregate_business_stats() if ds == "yelp" else {"error": "Solo disponible para yelp"},
        "ambiguous_reviews": lambda ds: aggregate_ambiguous_reviews(ds),
        "olist_sales": lambda ds: aggregate_olist_sales_by_month() if ds == "olist" else {"error": "Solo disponible para olist"},
        "olist_reviews_sales": lambda ds: aggregate_olist_reviews_sales() if ds == "olist" else {"error": "Solo disponible para olist"},
    }
    
    if aggregation_type not in aggregation_map:
        return {"error": f"Agregacion no soportada: {aggregation_type}"}
    
    try:
        return aggregation_map[aggregation_type](dataset)
    except Exception as e:
        return {"error": str(e)}


def build_silver_node(state: NLPGraphState) -> dict[str, Any]:
    """
    Nodo de grafo NLP que construye silver layer.
    
    Args:
        state: Estado del grafo NLP (diccionario en runtime)
        
    Returns:
        Dict con silver_path y/o errors
    """
    # Acceso por keys - state es dict en runtime
    dataset = state.get("dataset", "")
    existing_errors = state.get("errors")
    errors = list(existing_errors) if existing_errors else []
    
    try:
        if dataset == "yelp":
            path = build_silver_yelp()
        elif dataset == "es":
            path = build_silver_es()
        elif dataset == "olist":
            orders_path, reviews_path = build_silver_olist()
            path = reviews_path
        else:
            errors.append(f"Dataset invalido: {dataset}")
            return {"errors": errors}
        
        return {"silver_path": str(path)}
        
    except NotImplementedError as e:
        errors.append(f"Silver no implementado: {e}")
        return {"errors": errors}
    except Exception as e:
        errors.append(f"Error en silver: {e}")
        return {"errors": errors}


def build_gold_node(state: NLPGraphState) -> dict[str, Any]:
    """
    Nodo de grafo NLP que construye gold features.
    
    Args:
        state: Estado del grafo NLP (diccionario en runtime)
        
    Returns:
        Dict con gold_path, features_computed y/o errors
    """
    # Acceso por keys - state es dict en runtime
    dataset = state.get("dataset", "")
    existing_errors = state.get("errors")
    errors = list(existing_errors) if existing_errors else []
    silver_path = state.get("silver_path")
    
    if not silver_path:
        errors.append("Silver path no disponible, no se puede construir gold")
        return {"errors": errors}
    
    try:
        path = build_gold_features(dataset)
        return {
            "gold_path": str(path),
            "features_computed": ["sentiment"]
        }
        
    except NotImplementedError as e:
        errors.append(f"Gold no implementado: {e}")
        return {"errors": errors}
    except Exception as e:
        errors.append(f"Error en gold: {e}")
        return {"errors": errors}
