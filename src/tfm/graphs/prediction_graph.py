"""
Grafo de Prediccion: Pipeline de prediccion de ventas (Olist).

Este grafo construye y ejecuta modelo de prediccion:
1. load_features: Carga features de ventas
2. train_model: Entrena modelo (si no existe)
3. predict: Genera predicciones
4. evaluate: Evalua metricas

Solo aplica para dataset Olist.

Fase 5: Implementar grafo de prediccion
"""

from typing import Any, Literal

from langgraph.graph import StateGraph, START, END

from tfm.schemas.state import PredictionGraphState


def build_prediction_graph(checkpointer: Any = None):
    """
    Construye el grafo de prediccion.
    
    Args:
        checkpointer: Checkpointer opcional
        
    Returns:
        Grafo compilado
    """
    builder = StateGraph(PredictionGraphState)
    
    # Nodos
    builder.add_node("validate_input", validate_prediction_input)
    builder.add_node("load_features", load_features_node)
    builder.add_node("check_model", check_model_exists_node)
    builder.add_node("train_model", train_model_node)
    builder.add_node("predict", predict_node)
    builder.add_node("evaluate", evaluate_predictions_node)
    
    # Edges
    builder.add_edge(START, "validate_input")
    builder.add_conditional_edges(
        "validate_input",
        lambda s: "continue" if not s.error else "end",
        {"continue": "load_features", "end": END}
    )
    builder.add_edge("load_features", "check_model")
    builder.add_conditional_edges(
        "check_model",
        route_model_exists,
        {"train": "train_model", "predict": "predict"}
    )
    builder.add_edge("train_model", "predict")
    builder.add_edge("predict", "evaluate")
    builder.add_edge("evaluate", END)
    
    return builder.compile(checkpointer=checkpointer)


def validate_prediction_input(state: PredictionGraphState) -> dict[str, Any]:
    """
    Valida inputs para prediccion.
    
    Solo Olist tiene datos de ventas.
    """
    # Fase 5: Implementar validacion completa
    return {}


def load_features_node(state: PredictionGraphState) -> dict[str, Any]:
    """
    Carga features de ventas.
    """
    # Fase 5: Implementar carga de features
    return {}


def check_model_exists_node(state: PredictionGraphState) -> dict[str, Any]:
    """
    Verifica si ya existe modelo entrenado.
    """
    from tfm.config.settings import get_settings
    
    settings = get_settings()
    model_path = settings.gold_dir / "models" / "prediction_revenue_xgboost.json"
    
    if model_path.exists():
        return {"model_path": str(model_path)}
    
    return {}


def route_model_exists(state: PredictionGraphState) -> Literal["train", "predict"]:
    """
    Decide si entrenar o usar modelo existente.
    """
    if state.model_path:
        return "predict"
    return "train"


def train_model_node(state: PredictionGraphState) -> dict[str, Any]:
    """
    Entrena modelo de prediccion.
    """
    # Fase 5: Implementar entrenamiento
    return {"model_version": "v1_placeholder"}


def predict_node(state: PredictionGraphState) -> dict[str, Any]:
    """
    Genera predicciones.
    """
    # Fase 5: Implementar prediccion
    return {}


def evaluate_predictions_node(state: PredictionGraphState) -> dict[str, Any]:
    """
    Evalua predicciones contra actuals.
    """
    # Fase 5: Implementar evaluacion
    return {
        "mae": 0.0,
        "rmse": 0.0,
    }


def run_prediction_pipeline(
    date_range: tuple[str, str],
    category: str = None,
    retrain: bool = False
) -> dict[str, Any]:
    """
    Helper para ejecutar pipeline de prediccion.
    
    Args:
        date_range: Rango de fechas a predecir
        category: Categoria especifica (opcional)
        retrain: Si forzar reentrenamiento
        
    Returns:
        Dict con predicciones y metricas
    """
    # Fase 5: Implementar
    return {
        "status": "not_implemented",
        "message": "Implementar en Fase 5"
    }


# Instancia global para langgraph.json
prediction_graph = build_prediction_graph()
