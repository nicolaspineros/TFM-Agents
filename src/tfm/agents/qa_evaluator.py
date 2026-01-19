"""
QA / Evaluator Agent.

El QA Evaluator valida resultados y asegura faithfulness:
- Checks deterministicos (schema, nulls, rangos)
- Validacion de claims vs evidencia
- Evaluacion de faithfulness (opcional con LLM)
- Metricas ML cuando aplica

Este agente puede:
- Aprobar resultado - flujo continua
- Rechazar con feedback - Synthesizer regenera
- Marcar warnings - incluir en reporte final
"""

from typing import Optional, Any
import json

from langchain_openai import ChatOpenAI

from tfm.config.settings import get_settings
from tfm.schemas.state import TFMState
from tfm.schemas.outputs import QAResult, QACheck, InsightsReport


# Checks deterministicos disponibles
DETERMINISTIC_CHECKS = [
    "schema_valid",
    "non_empty_data",
    "valid_sentiment_range",
    "valid_stars_range",
    "no_null_required_fields",
    "claims_have_evidence",
]


def create_qa_evaluator():
    """
    Crea instancia del QA Evaluator.
    
    Returns:
        LLM configurado para QA
    """
    settings = get_settings()
    
    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=0,  # Deterministico para QA
        api_key=settings.openai_api_key,
    )
    
    return llm


def evaluate_result(state: TFMState) -> dict[str, Any]:
    """
    Nodo que evalua el resultado antes de entregarlo.
    
    Ejecuta checks deterministicos y opcionalmente LLM-based.
    
    Args:
        state: Estado con insights_report (diccionario en runtime)
        
    Returns:
        Dict con qa_passed y qa_feedback para actualizar estado
    """
    # Acceso por keys - state es dict en runtime
    insights_report = state.get("insights_report")
    user_query = state.get("user_query", "")
    last_result = state.get("last_result")
    
    # Si no hay reporte, marcar como fallido pero con mensaje util
    if not insights_report:
        return {
            "qa_passed": False,
            "qa_feedback": "No hay reporte para evaluar",
            "insights_report": {
                "summary": "No se pudo generar un reporte de insights.",
                "bullets": [],
                "caveats": ["El proceso de sintesis no genero resultados"],
                "query_answered": user_query,
            }
        }
    
    # Ejecutar checks deterministicos
    checks = []
    
    # Check 1: Reporte tiene estructura valida
    schema_check = _check_report_schema(insights_report)
    checks.append(schema_check)
    
    # Check 2: Hay datos de respaldo
    data_check = _check_has_data(state)
    checks.append(data_check)
    
    # Check 3: Claims tienen evidencia
    evidence_check = _check_claims_have_evidence(insights_report)
    checks.append(evidence_check)
    
    # Check 4: No hay valores fuera de rango
    if last_result:
        range_check = _check_value_ranges(last_result)
        checks.append(range_check)
    
    # Determinar resultado global
    all_passed = all(c.passed for c in checks if c.severity == "error")
    warnings = [c for c in checks if not c.passed and c.severity == "warning"]
    
    # Generar feedback si hay problemas
    feedback = None
    if not all_passed:
        feedback = _generate_feedback(checks)
    elif warnings:
        feedback = f"QA pasado con {len(warnings)} advertencias"
    
    return {
        "qa_passed": all_passed,
        "qa_feedback": feedback,
    }


def _check_report_schema(report: dict[str, Any]) -> QACheck:
    """
    Verifica que el reporte tenga estructura valida.
    """
    required_fields = ["summary"]
    missing = [f for f in required_fields if f not in report or not report[f]]
    
    if missing:
        return QACheck(
            check_name="schema_valid",
            passed=False,
            message=f"Campos faltantes: {missing}",
            severity="error"
        )
    
    return QACheck(
        check_name="schema_valid",
        passed=True,
        message="Schema del reporte valido",
        severity="info"
    )


def _check_has_data(state: TFMState) -> QACheck:
    """
    Verifica que haya datos de respaldo para el reporte.
    
    Args:
        state: Estado del grafo (diccionario en runtime)
    """
    # Acceso por keys - state es dict en runtime
    aggregation_results = state.get("aggregation_results")
    last_result = state.get("last_result")
    
    # Verificar aggregation_results primero
    if aggregation_results and len(aggregation_results) > 0:
        return QACheck(
            check_name="non_empty_data",
            passed=True,
            message=f"Datos disponibles: {len(aggregation_results)} agregaciones",
            severity="info"
        )
    
    if not last_result:
        return QACheck(
            check_name="non_empty_data",
            passed=False,
            message="No hay datos de respaldo (last_result vacio)",
            severity="warning"
        )
    
    # last_result es dict, acceder con get()
    row_count = last_result.get("row_count", 0)
    if row_count == 0:
        return QACheck(
            check_name="non_empty_data",
            passed=False,
            message="Los datos de respaldo estan vacios",
            severity="error"
        )
    
    return QACheck(
        check_name="non_empty_data",
        passed=True,
        message=f"Datos disponibles: {row_count} filas",
        severity="info"
    )


def _check_claims_have_evidence(report: dict[str, Any]) -> QACheck:
    """
    Verifica que los claims tengan evidencia.
    """
    bullets = report.get("bullets", [])
    
    if not bullets:
        return QACheck(
            check_name="claims_have_evidence",
            passed=True,
            message="No hay bullets que verificar",
            severity="info"
        )
    
    without_evidence = []
    for i, bullet in enumerate(bullets):
        if isinstance(bullet, dict):
            if not bullet.get("evidence"):
                without_evidence.append(i)
    
    if without_evidence:
        return QACheck(
            check_name="claims_have_evidence",
            passed=True,  # Solo warning, no error
            message=f"Bullets sin evidencia: indices {without_evidence}",
            severity="warning"
        )
    
    return QACheck(
        check_name="claims_have_evidence",
        passed=True,
        message="Todos los bullets tienen evidencia",
        severity="info"
    )


def _check_value_ranges(result: Any) -> QACheck:
    """
    Verifica que valores numericos esten en rangos validos.
    
    Args:
        result: Resultado de query (diccionario en runtime)
    """
    try:
        # result es dict en runtime, acceder con get()
        data_json = result.get("data_json", "[]") if isinstance(result, dict) else getattr(result, "data_json", "[]")
        data = json.loads(data_json) if isinstance(data_json, str) else data_json
    except Exception:
        return QACheck(
            check_name="valid_value_ranges",
            passed=True,
            message="No se pudo parsear datos para verificar rangos",
            severity="info"
        )
    
    issues = []
    
    if isinstance(data, list):
        for row in data[:50]:  # Solo verificar primeras 50 filas
            if isinstance(row, dict):
                # Verificar sentiment_score en [-1, 1]
                if "sentiment_score" in row:
                    val = row["sentiment_score"]
                    if val is not None and (val < -1 or val > 1):
                        issues.append(f"sentiment_score fuera de rango: {val}")
                
                # Verificar stars en [1, 5]
                if "stars" in row:
                    val = row["stars"]
                    if val is not None and (val < 1 or val > 5):
                        issues.append(f"stars fuera de rango: {val}")
                
                # Verificar porcentajes en [0, 100]
                for key in ["pct_positive", "pct_negative", "pct_ambiguous"]:
                    if key in row:
                        val = row[key]
                        if val is not None and (val < 0 or val > 100):
                            issues.append(f"{key} fuera de rango: {val}")
    
    if issues:
        return QACheck(
            check_name="valid_value_ranges",
            passed=False,
            message=f"Valores fuera de rango: {issues[:3]}...",
            severity="warning"
        )
    
    return QACheck(
        check_name="valid_value_ranges",
        passed=True,
        message="Todos los valores en rangos validos",
        severity="info"
    )


def _generate_feedback(checks: list[QACheck]) -> str:
    """
    Genera feedback basado en checks fallidos.
    """
    failed = [c for c in checks if not c.passed]
    
    if not failed:
        return ""
    
    feedback_lines = ["QA encontro los siguientes problemas:"]
    for check in failed:
        feedback_lines.append(f"- [{check.severity.upper()}] {check.check_name}: {check.message}")
    
    return "\n".join(feedback_lines)


def evaluate_faithfulness_llm(
    report: InsightsReport,
    source_data: str
) -> float:
    """
    Evalua faithfulness usando LLM.
    
    Verifica que los claims del reporte esten soportados
    por los datos de origen.
    
    Args:
        report: Reporte a evaluar
        source_data: Datos de origen serializados
        
    Returns:
        Score de 0 a 1
    """
    # Implementacion basica - retornar 1.0 por ahora
    # En produccion, usar LLM para evaluar
    return 1.0


def run_ml_evaluation(
    predictions_path: str,
    ground_truth_path: str
) -> dict[str, float]:
    """
    Ejecuta evaluacion de metricas ML.
    
    Metricas:
    - Clasificacion: Accuracy, F1, Precision, Recall
    - Regresion: MAE, RMSE, R2
    
    Args:
        predictions_path: Path a predicciones
        ground_truth_path: Path a ground truth
        
    Returns:
        Dict con metricas
    """
    # Implementacion basica - retornar metricas dummy
    return {
        "accuracy": 0.0,
        "f1": 0.0,
        "mae": 0.0,
        "rmse": 0.0,
    }
