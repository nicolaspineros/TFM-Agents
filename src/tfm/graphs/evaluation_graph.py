"""
Grafo de Evaluacion: Runner de evaluacion offline.

Este grafo ejecuta evaluaciones sistematicas:
1. ML Metrics: F1, MAE, RMSE para modelos
2. QA Checks: Faithfulness, schema validation
3. LangSmith Evals: Evaluadores custom en LangSmith

Util para:
- CI/CD: Verificar que no haya regresiones
- Benchmarking: Comparar versiones de modelos
- Reporting: Generar metricas para TFM

Fase 6: Implementar grafo de evaluacion
"""

from typing import Any, Literal, Optional

from langgraph.graph import StateGraph, START, END

from tfm.schemas.state import EvaluationGraphState


def build_evaluation_graph(checkpointer: Any = None):
    """
    Construye el grafo de evaluacion.
    
    Args:
        checkpointer: Checkpointer opcional
        
    Returns:
        Grafo compilado
    """
    builder = StateGraph(EvaluationGraphState)
    
    # Nodos
    builder.add_node("setup", setup_evaluation)
    builder.add_node("ml_metrics", run_ml_metrics)
    builder.add_node("qa_checks", run_qa_checks)
    builder.add_node("langsmith_eval", run_langsmith_eval)
    builder.add_node("aggregate_results", aggregate_eval_results)
    
    # Edges
    builder.add_edge(START, "setup")
    builder.add_conditional_edges(
        "setup",
        route_eval_type,
        {
            "ml": "ml_metrics",
            "qa": "qa_checks",
            "langsmith": "langsmith_eval",
            "all": "ml_metrics",
        }
    )
    
    # Si es "all", ejecutar en secuencia
    builder.add_edge("ml_metrics", "qa_checks")
    builder.add_edge("qa_checks", "langsmith_eval")
    builder.add_edge("langsmith_eval", "aggregate_results")
    
    builder.add_edge("aggregate_results", END)
    
    return builder.compile(checkpointer=checkpointer)


def setup_evaluation(state: EvaluationGraphState) -> dict[str, Any]:
    """
    Configura evaluacion: carga datos de test, etc.
    """
    # Fase 6: Implementar setup
    return {}


def route_eval_type(state: EvaluationGraphState) -> Literal["ml", "qa", "langsmith", "all"]:
    """
    Decide que tipo de evaluacion ejecutar.
    """
    eval_type = state.eval_type
    
    if eval_type == "ml_metrics":
        return "ml"
    elif eval_type == "qa_faithfulness":
        return "qa"
    elif eval_type == "langsmith_eval":
        return "langsmith"
    else:
        return "all"


def run_ml_metrics(state: EvaluationGraphState) -> dict[str, Any]:
    """
    Ejecuta metricas ML.
    """
    # Fase 6: Implementar metricas ML
    current_metrics = dict(state.metrics) if state.metrics else {}
    current_metrics["ml_status"] = "not_implemented"
    
    return {"metrics": current_metrics}


def run_qa_checks(state: EvaluationGraphState) -> dict[str, Any]:
    """
    Ejecuta checks de QA.
    """
    # Fase 6: Implementar QA checks
    current_metrics = dict(state.metrics) if state.metrics else {}
    current_metrics["qa_status"] = "not_implemented"
    
    return {"metrics": current_metrics}


def run_langsmith_eval(state: EvaluationGraphState) -> dict[str, Any]:
    """
    Ejecuta evaluaciones en LangSmith.
    """
    # Fase 6: Implementar con LangSmith SDK
    current_metrics = dict(state.metrics) if state.metrics else {}
    current_metrics["langsmith_status"] = "not_implemented"
    
    return {"metrics": current_metrics}


def aggregate_eval_results(state: EvaluationGraphState) -> dict[str, Any]:
    """
    Agrega resultados de todas las evaluaciones.
    """
    current_metrics = dict(state.metrics) if state.metrics else {}
    
    # Calcular score global
    total_checks = len(state.passed_checks) + len(state.failed_checks)
    if total_checks > 0:
        current_metrics["pass_rate"] = len(state.passed_checks) / total_checks
    
    return {"metrics": current_metrics}


def run_evaluation(
    eval_type: str = "all",
    dataset: Optional[str] = None,
    output_path: Optional[str] = None
) -> dict[str, Any]:
    """
    Helper para ejecutar evaluacion.
    
    Args:
        eval_type: Tipo de evaluacion (ml_metrics, qa_faithfulness, langsmith_eval, all)
        dataset: Dataset a evaluar (opcional)
        output_path: Path para guardar resultados (opcional)
        
    Returns:
        Dict con resultados de evaluacion
        
    Example:
        >>> results = run_evaluation("all", dataset="yelp")
        >>> print(results["metrics"])
    """
    initial_state = {
        "eval_type": eval_type,
        "dataset": dataset,
        "messages": [],
        "metrics": {},
        "passed_checks": [],
        "failed_checks": [],
    }
    
    final_state = evaluation_graph.invoke(initial_state)
    
    results = {
        "eval_type": eval_type,
        "dataset": dataset,
        "metrics": final_state.get("metrics", {}),
        "passed_checks": final_state.get("passed_checks", []),
        "failed_checks": final_state.get("failed_checks", []),
        "langsmith_run_id": final_state.get("langsmith_run_id"),
    }
    
    # Guardar si se especifico path
    if output_path:
        import json
        from pathlib import Path
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
    
    return results


# Instancia global para langgraph.json
evaluation_graph = build_evaluation_graph()
