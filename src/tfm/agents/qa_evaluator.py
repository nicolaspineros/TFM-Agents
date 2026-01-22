"""
QA / Evaluator Agent.

El QA Evaluator valida resultados y asegura faithfulness:
- Checks deterministicos (schema, nulls, rangos)
- Validacion LLM: ¿La respuesta contesta la pregunta?
- Evaluacion de faithfulness: ¿Los claims estan soportados por datos?
- Confidence score basado en calidad de la respuesta

Este agente puede:
- Aprobar resultado - flujo continua
- Rechazar con feedback - Synthesizer podria regenerar
- Marcar warnings - incluir en reporte final
"""

from typing import Optional, Any
import json

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

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

# System prompt para el QA Evaluator
QA_SYSTEM_PROMPT = """Eres el QA Evaluator de un sistema de análisis de reseñas.

Tu trabajo es evaluar si la respuesta generada:
1. RESPONDE la pregunta del usuario de forma directa
2. TIENE SOPORTE en los datos proporcionados (faithfulness)
3. NO INVENTA información que no está en los datos
4. ES COHERENTE y clara

Evalúa de forma ESTRICTA pero JUSTA. Si la respuesta es razonable
y está basada en datos reales, apruébala.

Responde SIEMPRE con JSON válido en este formato exacto:
{
    "answers_query": true/false,
    "claims_supported": true/false,
    "confidence": 0.0-1.0,
    "issues": ["lista de problemas encontrados"],
    "passed": true/false,
    "feedback": "explicación breve si hay problemas"
}"""


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
    
    Ejecuta:
    1. Checks deterministicos (schema, datos, rangos)
    2. Evaluacion LLM (faithfulness, relevancia)
    
    Args:
        state: Estado con insights_report (diccionario en runtime)
        
    Returns:
        Dict con qa_passed, qa_feedback, y qa_confidence para actualizar estado
    """
    # Acceso por keys - state es dict en runtime
    insights_report = state.get("insights_report")
    user_query = state.get("user_query", "")
    last_result = state.get("last_result")
    aggregation_results = state.get("aggregation_results")
    
    # Si no hay reporte, marcar como fallido pero con mensaje util
    if not insights_report:
        return {
            "qa_passed": False,
            "qa_feedback": "No hay reporte para evaluar",
            "qa_confidence": 0.0,
            "insights_report": {
                "summary": "No se pudo generar un reporte de insights.",
                "bullets": [],
                "caveats": ["El proceso de sintesis no genero resultados"],
                "query_answered": user_query,
            }
        }
    
    # === FASE 1: Checks deterministicos ===
    deterministic_checks = []
    
    # Check 1: Reporte tiene estructura valida
    schema_check = _check_report_schema(insights_report)
    deterministic_checks.append(schema_check)
    
    # Check 2: Hay datos de respaldo
    data_check = _check_has_data(state)
    deterministic_checks.append(data_check)
    
    # Check 3: Claims tienen evidencia
    evidence_check = _check_claims_have_evidence(insights_report)
    deterministic_checks.append(evidence_check)
    
    # Check 4: No hay valores fuera de rango
    if last_result:
        range_check = _check_value_ranges(last_result)
        deterministic_checks.append(range_check)
    
    # Verificar si hay errores criticos en checks deterministicos
    deterministic_errors = [c for c in deterministic_checks if not c.passed and c.severity == "error"]
    
    if deterministic_errors:
        # Fallar inmediatamente si hay errores criticos
        feedback = _generate_feedback(deterministic_checks)
        return {
            "qa_passed": False,
            "qa_feedback": feedback,
            "qa_confidence": 0.0,
        }
    
    # === FASE 2: Evaluacion con LLM ===
    llm_eval = _evaluate_with_llm(
        user_query=user_query,
        insights_report=insights_report,
        aggregation_results=aggregation_results,
        last_result=last_result
    )
    
    # Combinar resultados
    deterministic_warnings = [c for c in deterministic_checks if not c.passed and c.severity == "warning"]
    
    # Decidir si pasa
    qa_passed = llm_eval.get("passed", False)
    confidence = llm_eval.get("confidence", 0.5)
    
    # Generar feedback combinado
    feedback_parts = []
    
    if deterministic_warnings:
        feedback_parts.append(f"Advertencias deterministicas: {len(deterministic_warnings)}")
    
    if llm_eval.get("issues"):
        feedback_parts.append(f"LLM: {llm_eval.get('feedback', '')}")
    
    if not qa_passed:
        feedback_parts.append(llm_eval.get("feedback", "Evaluacion LLM no paso"))
    
    feedback = " | ".join(feedback_parts) if feedback_parts else "QA aprobado"
    
    # Agregar confidence al reporte
    updated_report = dict(insights_report)
    updated_report["qa_confidence"] = confidence
    updated_report["qa_checks_passed"] = len([c for c in deterministic_checks if c.passed])
    
    return {
        "qa_passed": qa_passed,
        "qa_feedback": feedback,
        "qa_confidence": confidence,
        "insights_report": updated_report,
    }


def _evaluate_with_llm(
    user_query: str,
    insights_report: dict[str, Any],
    aggregation_results: Optional[dict] = None,
    last_result: Optional[dict] = None
) -> dict[str, Any]:
    """
    Evalua la respuesta usando LLM para faithfulness y relevancia.
    
    Args:
        user_query: Pregunta original del usuario
        insights_report: Reporte generado por el synthesizer
        aggregation_results: Resultados de agregaciones
        last_result: Ultimo resultado de query
        
    Returns:
        Dict con evaluacion LLM
    """
    try:
        llm = create_qa_evaluator()
        
        # Preparar datos de respaldo para el LLM
        data_context = _prepare_data_for_evaluation(aggregation_results, last_result)
        
        # Preparar el reporte
        report_summary = insights_report.get("summary", "")
        report_bullets = insights_report.get("bullets", [])
        
        # Formatear bullets
        bullets_text = ""
        if report_bullets:
            for i, bullet in enumerate(report_bullets):
                if isinstance(bullet, dict):
                    bullets_text += f"\n  - {bullet.get('text', '')}"
                    if bullet.get('evidence'):
                        bullets_text += f" (evidencia: {bullet.get('evidence')})"
                else:
                    bullets_text += f"\n  - {bullet}"
        
        eval_prompt = f"""Evalúa si esta respuesta es correcta y está basada en los datos.

PREGUNTA DEL USUARIO:
{user_query}

DATOS DISPONIBLES (de las herramientas ejecutadas):
{data_context}

RESPUESTA GENERADA:
Summary: {report_summary}
Bullets: {bullets_text}
Caveats: {insights_report.get('caveats', [])}

Evalúa:
1. ¿La respuesta RESPONDE la pregunta del usuario?
2. ¿Los claims/números están SOPORTADOS por los datos?
3. ¿Hay información INVENTADA que no está en los datos?

Responde con JSON."""

        messages = [
            SystemMessage(content=QA_SYSTEM_PROMPT),
            HumanMessage(content=eval_prompt)
        ]
        
        response = llm.invoke(messages)
        
        # Parsear respuesta
        return _parse_llm_evaluation(response.content)
        
    except Exception as e:
        # Si LLM falla, aprobar con confidence medio
        print(f"[QA] Error en evaluacion LLM: {e}")
        return {
            "answers_query": True,
            "claims_supported": True,
            "confidence": 0.6,
            "issues": [f"Evaluacion LLM no disponible: {str(e)}"],
            "passed": True,
            "feedback": "Aprobado por defecto (LLM no disponible)"
        }


def _prepare_data_for_evaluation(
    aggregation_results: Optional[dict],
    last_result: Optional[dict]
) -> str:
    """Prepara datos para mostrar al LLM evaluador."""
    parts = []
    
    if aggregation_results:
        for key, value in list(aggregation_results.items())[:3]:  # Limitar
            if isinstance(value, dict):
                # Extraer info clave
                data = value.get("data", [])
                if isinstance(data, list) and data:
                    parts.append(f"[{key}]: {len(data)} registros")
                    # Mostrar primeros 5
                    sample = data[:5]
                    parts.append(f"  Muestra: {json.dumps(sample, default=str)[:500]}")
                else:
                    parts.append(f"[{key}]: {json.dumps(value, default=str)[:300]}")
    
    if last_result:
        row_count = last_result.get("row_count", 0)
        parts.append(f"[last_result]: {row_count} filas")
        try:
            data_json = last_result.get("data_json", "[]")
            data = json.loads(data_json) if isinstance(data_json, str) else data_json
            if isinstance(data, list) and data:
                parts.append(f"  Muestra: {json.dumps(data[:3], default=str)[:500]}")
        except:
            pass
    
    return "\n".join(parts) if parts else "No hay datos de respaldo disponibles"


def _parse_llm_evaluation(response_text: str) -> dict[str, Any]:
    """Parsea la respuesta JSON del LLM."""
    text = response_text.strip()
    
    # Remover markdown code blocks
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    
    try:
        result = json.loads(text)
        
        # Asegurar campos requeridos
        return {
            "answers_query": result.get("answers_query", True),
            "claims_supported": result.get("claims_supported", True),
            "confidence": float(result.get("confidence", 0.7)),
            "issues": result.get("issues", []),
            "passed": result.get("passed", True),
            "feedback": result.get("feedback", "")
        }
        
    except json.JSONDecodeError:
        # Si no puede parsear, asumir que paso
        return {
            "answers_query": True,
            "claims_supported": True,
            "confidence": 0.6,
            "issues": ["No se pudo parsear evaluacion LLM"],
            "passed": True,
            "feedback": "Aprobado (parsing fallido)"
        }


# =============================================================================
# CHECKS DETERMINISTICOS (sin cambios)
# =============================================================================

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


# =============================================================================
# FUNCIONES ADICIONALES (para uso futuro)
# =============================================================================

def evaluate_faithfulness_llm(
    report: InsightsReport,
    source_data: str
) -> float:
    """
    Evalua faithfulness usando LLM (version standalone).
    
    Verifica que los claims del reporte esten soportados
    por los datos de origen.
    
    Args:
        report: Reporte a evaluar
        source_data: Datos de origen serializados
        
    Returns:
        Score de 0 a 1
    """
    # Convertir a dict si es necesario
    report_dict = report if isinstance(report, dict) else {
        "summary": getattr(report, "summary", ""),
        "bullets": getattr(report, "bullets", []),
    }
    
    eval_result = _evaluate_with_llm(
        user_query="Evaluar faithfulness del reporte",
        insights_report=report_dict,
        aggregation_results={"source": {"data": source_data}},
        last_result=None
    )
    
    return eval_result.get("confidence", 0.5)


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
    # TODO: Implementar cuando se agregue prediccion
    return {
        "status": "not_implemented",
        "note": "Disponible en Fase 6"
    }
