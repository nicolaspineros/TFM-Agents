"""
Insight Synthesizer Agent.

El Synthesizer consume resultados de agregaciones y genera
reportes estructurados (InsightsReport) con:
- Resumen ejecutivo
- Bullets con evidencia
- Caveats y limitaciones
- Trazabilidad de fuentes

Este agente usa el LLM para:
- Interpretar tablas de datos
- Redactar insights en lenguaje natural
- Estructurar la respuesta con Pydantic
"""

from typing import Optional, Any
import json

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from tfm.config.settings import get_settings
from tfm.schemas.state import TFMState, LastQueryResult
from tfm.schemas.outputs import InsightsReport, InsightBullet


# System prompt para el Synthesizer
SYNTHESIZER_SYSTEM_PROMPT = """Eres el Insight Synthesizer de un sistema de análisis de reseñas.

Tu trabajo es:
1. Leer los datos de agregaciones/consultas proporcionados
2. Generar insights accionables y concisos
3. Incluir evidencia numérica para cada consulta
4. Señalar limitaciones o caveats

REGLAS:
- Se CONCISO: resumen de 1-2 oraciones maximo
- Se PRECISO: incluye numeros exactos de los datos
- Se HONESTO: si los datos son limitados, dilo
- NO INVENTES: solo usa informacion de los datos proporcionados
- RESPONDE EN ESPAÑOL

Responde con un JSON valido con esta estructura:
{{
    "summary": "Resumen ejecutivo en 1-2 oraciones",
    "bullets": [
        {{"text": "Insight especifico", "evidence": "dato=valor", "confidence": "high|medium|low"}}
    ],
    "caveats": ["Advertencia o limitacion"],
    "query_answered": "La pregunta que respondiste"
}}
"""


def create_synthesizer():
    """
    Crea instancia del Synthesizer.
    
    Returns:
        LLM configurado para sintesis
    """
    settings = get_settings()
    
    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=0.3,  # Un poco de creatividad para redaccion
        api_key=settings.openai_api_key,
    )
    
    return llm


def generate_insights(state: TFMState) -> dict[str, Any]:
    """
    Nodo que genera insights a partir de resultados.
    
    Lee last_result y aggregation_results del estado y genera InsightsReport.
    
    Args:
        state: Estado con last_result y aggregation_results (diccionario en runtime)
        
    Returns:
        Dict con insights_report para actualizar estado
    """
    # Acceso por keys - state es dict en runtime
    user_query = state.get("user_query", "")
    last_result = state.get("last_result")
    aggregation_results = state.get("aggregation_results")
    query_plan = state.get("query_plan")
    artifacts = state.get("artifacts")
    
    # Si no hay resultados, generar reporte minimo
    if not last_result and not aggregation_results:
        return {
            "insights_report": {
                "summary": "No se encontraron datos para analizar.",
                "bullets": [],
                "caveats": ["No hay resultados de agregacion disponibles"],
                "query_answered": user_query,
                "datasets_used": [],
                "artifacts_used": [],
            },
            "error": "No hay resultados para sintetizar"
        }
    
    # Preparar datos para el LLM
    data_context = _prepare_data_context(state)
    
    try:
        # Llamar al LLM para sintetizar
        synthesizer = create_synthesizer()
        
        messages = [
            SystemMessage(content=SYNTHESIZER_SYSTEM_PROMPT),
            HumanMessage(content=f"""
            Pregunta del usuario: {user_query}

            Datos disponibles:
                {data_context}

            Genera un reporte de insights basado en estos datos.
            """)
        ]
        
        response = synthesizer.invoke(messages)
        
        # Parsear respuesta JSON
        report_dict = _parse_json_response(response.content)
        
        # Asegurar campos requeridos - query_plan y artifacts son dicts
        datasets_used = query_plan.get("datasets_required", []) if query_plan else []
        aggregations_run = query_plan.get("aggregations_needed", []) if query_plan else []
        artifacts_used = list(artifacts.keys()) if artifacts else []
        
        insights_report = {
            "summary": report_dict.get("summary", "Analisis completado."),
            "bullets": report_dict.get("bullets", []),
            "caveats": report_dict.get("caveats", []),
            "query_answered": user_query,
            "datasets_used": datasets_used,
            "artifacts_used": artifacts_used,
            "aggregations_run": aggregations_run,
        }
        
        return {"insights_report": insights_report}
        
    except Exception as e:
        # En caso de error, generar reporte basico con los datos
        return {
            "insights_report": _generate_fallback_report(user_query, state),
            "error": f"Error en synthesizer: {str(e)}"
        }


def _prepare_data_context(state: TFMState) -> str:
    """
    Prepara contexto de datos para el LLM.
    
    Args:
        state: Estado del grafo (diccionario en runtime)
    """
    lines = []
    
    # Acceso por keys - state es dict en runtime
    aggregation_results = state.get("aggregation_results")
    last_result = state.get("last_result")
    
    # Agregaciones
    if aggregation_results:
        lines.append("## Resultados de Agregaciones:")
        for key, value in aggregation_results.items():
            lines.append(f"\n### {key}:")
            if isinstance(value, dict):
                if "data" in value:
                    data_slice = value["data"][:20] if isinstance(value["data"], list) else value["data"]
                    lines.append(json.dumps(data_slice, indent=2))
                else:
                    lines.append(json.dumps(value, indent=2))
            else:
                lines.append(str(value)[:1000])  # Limitar longitud
    
    # Last result - es un dict, no un objeto
    if last_result:
        lines.append("\n## Ultimo Resultado:")
        lines.append(f"- Tipo: {last_result.get('query_type', 'unknown')}")
        lines.append(f"- Columnas: {last_result.get('columns', [])}")
        lines.append(f"- Filas: {last_result.get('row_count', 0)}")
        
        try:
            data_json = last_result.get("data_json", "[]")
            data = json.loads(data_json)
            if isinstance(data, list) and len(data) > 0:
                lines.append(f"- Muestra: {json.dumps(data[:5], indent=2)}")
        except Exception:
            pass
    
    return "\n".join(lines) if lines else "No hay datos disponibles."


def _parse_json_response(response_text: str) -> dict[str, Any]:
    """
    Parsea respuesta JSON del LLM.
    """
    text = response_text.strip()
    
    # Remover markdown code blocks si existen
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    
    if text.endswith("```"):
        text = text[:-3]
    
    text = text.strip()
    
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Intentar extraer JSON con regex
        import re
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # Retornar estructura basica
        return {
            "summary": text[:200] if text else "Analisis completado.",
            "bullets": [],
            "caveats": ["No se pudo parsear respuesta estructurada"]
        }


def _generate_fallback_report(query: str, state: TFMState) -> dict[str, Any]:
    """
    Genera reporte basico cuando el LLM falla.
    
    Args:
        query: Pregunta del usuario
        state: Estado del grafo (diccionario en runtime)
    """
    bullets = []
    
    # Acceso por keys - state es dict en runtime
    aggregation_results = state.get("aggregation_results")
    last_result = state.get("last_result")
    query_plan = state.get("query_plan")
    artifacts = state.get("artifacts")
    
    # Extraer info de agregaciones
    if aggregation_results:
        for key, value in list(aggregation_results.items())[:3]:
            if isinstance(value, dict) and "data" in value:
                data = value["data"]
                if isinstance(data, list) and len(data) > 0:
                    bullets.append({
                        "text": f"Agregacion {key} procesada",
                        "evidence": f"{len(data)} filas de datos",
                        "confidence": "medium"
                    })
    
    # Info de last_result - es un dict
    if last_result:
        row_count = last_result.get("row_count", 0)
        columns = last_result.get("columns", [])
        bullets.append({
            "text": f"Se procesaron {row_count} registros",
            "evidence": f"columnas={columns}",
            "confidence": "high"
        })
    
    # query_plan y artifacts son dicts
    datasets_used = query_plan.get("datasets_required", []) if query_plan else []
    artifacts_used = list(artifacts.keys()) if artifacts else []
    
    return {
        "summary": "Analisis completado. Revisa los datos detallados para mas informacion.",
        "bullets": bullets,
        "caveats": ["Reporte generado automaticamente sin LLM"],
        "query_answered": query,
        "datasets_used": datasets_used,
        "artifacts_used": artifacts_used,
    }


def format_data_for_llm(result: LastQueryResult) -> str:
    """
    Formatea datos para incluir en prompt del LLM.
    
    Optimiza para:
    - No exceder limites de contexto
    - Mantener informacion relevante
    - Facilitar interpretacion del LLM
    """
    # Para tablas pequenas, incluir todo
    if result.row_count <= 50:
        return result.data_json
    
    # Para tablas grandes, incluir sample + stats
    try:
        data = json.loads(result.data_json)
        
        formatted = {
            "total_rows": result.row_count,
            "columns": result.columns,
            "sample_first_10": data[:10] if isinstance(data, list) else data,
            "sample_last_5": data[-5:] if isinstance(data, list) and len(data) > 10 else [],
        }
        
        return json.dumps(formatted, indent=2)
    except:
        return result.data_json


def extract_key_metrics(result: LastQueryResult) -> dict[str, Any]:
    """
    Extrae metricas clave de un resultado.
    
    Util para que el LLM tenga un resumen rapido.
    """
    try:
        data = json.loads(result.data_json)
    except:
        return {}
    
    if not data or not isinstance(data, list):
        return {}
    
    metrics = {
        "row_count": len(data),
        "columns": result.columns,
    }
    
    # Intentar calcular metricas numericas
    for col in result.columns:
        try:
            values = [row.get(col) for row in data if row.get(col) is not None]
            if values and all(isinstance(v, (int, float)) for v in values):
                metrics[f"{col}_min"] = min(values)
                metrics[f"{col}_max"] = max(values)
                metrics[f"{col}_avg"] = sum(values) / len(values)
        except Exception:
            pass
    
    return metrics
