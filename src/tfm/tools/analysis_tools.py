"""
Tools de análisis para el sistema agentic.

Este módulo define todas las herramientas que el LLM puede invocar
para analizar datos de reseñas. Usa el decorador @tool de LangChain
para que el LLM pueda descubrir automáticamente qué herramientas
están disponibles y cuándo usarlas.

El LLM ve las descripciones de cada tool y decide cuál usar
basándose en la pregunta del usuario.
"""

from typing import Literal, Optional, Dict, Any, List
from langchain_core.tools import tool

from tfm.config.settings import get_settings, SILVER_FILES
from tfm.tools.aggregations import (
    aggregate_reviews_by_stars,
    aggregate_reviews_by_month,
    aggregate_olist_sales_by_month,
    aggregate_yelp_user_stats,
    aggregate_business_stats,
    aggregate_ambiguous_reviews,
    aggregate_by_text_length,
    aggregate_olist_by_category,
    aggregate_olist_reviews_sales,
)
from tfm.tools.preprocess import check_silver_status


# =============================================================================
# TOOLS DE AGREGACIÓN - El LLM puede invocar estas directamente
# =============================================================================

@tool
def get_reviews_distribution(dataset: Literal["yelp", "es", "olist"]) -> Dict[str, Any]:
    """
    Obtiene la distribución de reseñas por estrellas/rating.
    
    Usa esta herramienta cuando el usuario pregunte sobre:
    - Distribución de ratings
    - Distribución de estrellas
    - Cuántas reseñas hay por cada puntuación
    - Porcentaje de reseñas positivas/negativas
    - Reviews por score
    
    DISPONIBLE PARA: yelp, es, olist (TODOS los datasets)
    
    Args:
        dataset: El dataset a analizar. Valores válidos: "yelp", "es", "olist"
        
    Returns:
        Diccionario con la distribución de ratings y resumen estadístico
    """
    print(f"[TOOL] get_reviews_distribution(dataset={dataset})")
    
    # Verificar que silver existe
    status = check_silver_status()
    silver_key = _get_silver_key(dataset)
    
    if silver_key and not status.get(silver_key, {}).get("exists", False):
        return {
            "error": f"Los datos silver para {dataset} no existen. Ejecuta build_silver primero.",
            "suggestion": "Usa la herramienta build_dataset_silver para construir los datos."
        }
    
    result = aggregate_reviews_by_stars(dataset)
    result["tool"] = "get_reviews_distribution"
    return result


@tool
def get_reviews_by_month(
    dataset: Literal["yelp", "olist"],
    year: Optional[int] = None
) -> Dict[str, Any]:
    """
    Obtiene la tendencia temporal de reseñas por mes.
    
    Usa esta herramienta cuando el usuario pregunte sobre:
    - Reseñas por mes
    - Tendencia temporal
    - Evolución de reviews
    - Cuántas reseñas hay por mes
    - Análisis mensual
    
    DISPONIBLE PARA: yelp, olist (NO disponible para 'es' porque no tiene fechas)
    
    Args:
        dataset: El dataset a analizar. Solo "yelp" u "olist" (es NO tiene fechas)
        year: Año específico a filtrar (opcional)
        
    Returns:
        Diccionario con conteo de reseñas por año/mes
    """
    print(f"[TOOL] get_reviews_by_month(dataset={dataset}, year={year})")
    
    if dataset == "es":
        return {
            "error": "El dataset 'es' (español) NO tiene campo de fecha.",
            "suggestion": "Usa el dataset 'yelp' u 'olist' para análisis temporal."
        }
    
    result = aggregate_reviews_by_month(dataset, year_filter=year)
    result["tool"] = "get_reviews_by_month"
    return result


@tool
def get_sales_by_month() -> Dict[str, Any]:
    """
    Obtiene las ventas/órdenes de Olist agrupadas por mes.
    
    Usa esta herramienta cuando el usuario pregunte sobre:
    - Ventas por mes
    - Órdenes por mes
    - Revenue mensual
    - Ingresos por periodo
    - Evolución de ventas
    - Cuántas órdenes hay por mes
    
    DISPONIBLE SOLO PARA: olist (los otros datasets NO tienen datos de ventas)
    
    Returns:
        Diccionario con ventas, órdenes y revenue por mes
    """
    print(f"[TOOL] get_sales_by_month()")
    
    # Verificar que silver existe
    status = check_silver_status()
    if not status.get("olist_orders", {}).get("exists", False):
        return {
            "error": "Los datos de órdenes de Olist no existen.",
            "suggestion": "Ejecuta build_silver para Olist primero."
        }
    
    result = aggregate_olist_sales_by_month()
    result["tool"] = "get_sales_by_month"
    return result


@tool
def get_user_stats() -> Dict[str, Any]:
    """
    Obtiene estadísticas de usuarios de Yelp.
    
    Usa esta herramienta cuando el usuario pregunte sobre:
    - Usuarios más influyentes
    - Estadísticas de usuarios
    - Top reviewers
    - Usuarios Elite
    - Análisis de usuarios
    
    DISPONIBLE SOLO PARA: yelp (los otros datasets no tienen datos de usuarios)
    
    Returns:
        Diccionario con estadísticas de usuarios y top reviewers
    """
    print(f"[TOOL] get_user_stats()")
    
    result = aggregate_yelp_user_stats()
    result["tool"] = "get_user_stats"
    return result


@tool
def get_business_stats() -> Dict[str, Any]:
    """
    Obtiene estadísticas de negocios de Yelp.
    
    Usa esta herramienta cuando el usuario pregunte sobre:
    - Estadísticas de negocios
    - Negocios abiertos/cerrados
    - Rating promedio de negocios
    - Análisis de negocios
    - Categorías de negocios
    
    DISPONIBLE SOLO PARA: yelp (los otros datasets no tienen datos de negocios)
    
    Returns:
        Diccionario con estadísticas de negocios
    """
    print(f"[TOOL] get_business_stats()")
    
    result = aggregate_business_stats()
    result["tool"] = "get_business_stats"
    return result


@tool
def get_ambiguous_reviews_analysis(dataset: Literal["yelp", "es", "olist"]) -> Dict[str, Any]:
    """
    Analiza las reseñas ambiguas (de 3 estrellas).
    
    Usa esta herramienta cuando el usuario pregunte sobre:
    - Reseñas de 3 estrellas
    - Reseñas ambiguas
    - Reviews neutrales
    - Análisis de reviews con puntuación media
    
    DISPONIBLE PARA: yelp, es, olist (TODOS los datasets)
    
    Args:
        dataset: El dataset a analizar
        
    Returns:
        Diccionario con análisis de reseñas ambiguas
    """
    print(f"[TOOL] get_ambiguous_reviews_analysis(dataset={dataset})")
    
    result = aggregate_ambiguous_reviews(dataset)
    result["tool"] = "get_ambiguous_reviews_analysis"
    return result


@tool
def get_text_length_analysis(dataset: Literal["yelp", "es", "olist"]) -> Dict[str, Any]:
    """
    Analiza las reseñas según su longitud de texto.
    
    Usa esta herramienta cuando el usuario pregunte sobre:
    - Reseñas largas vs cortas
    - Longitud de texto
    - Comparación por tamaño de review
    - Análisis de extensión de reseñas
    
    DISPONIBLE PARA: yelp, es, olist (TODOS los datasets)
    
    Args:
        dataset: El dataset a analizar
        
    Returns:
        Diccionario con análisis por longitud de texto
    """
    print(f"[TOOL] get_text_length_analysis(dataset={dataset})")
    
    result = aggregate_by_text_length(dataset)
    result["tool"] = "get_text_length_analysis"
    return result


@tool
def get_sales_by_category(top_n: int = 20) -> Dict[str, Any]:
    """
    Obtiene ventas de Olist agrupadas por categoría de producto.
    
    Usa esta herramienta cuando el usuario pregunte sobre:
    - Ventas por categoría
    - Categorías más vendidas
    - Top categorías
    - Productos más vendidos por tipo
    
    DISPONIBLE SOLO PARA: olist
    
    Args:
        top_n: Número de categorías top a retornar (default 20)
        
    Returns:
        Diccionario con ventas por categoría
    """
    print(f"[TOOL] get_sales_by_category(top_n={top_n})")
    
    result = aggregate_olist_by_category(top_n)
    result["tool"] = "get_sales_by_category"
    return result


@tool
def get_reviews_sales_correlation_basic() -> Dict[str, Any]:
    """
    Analiza la correlacion basica entre reviews y ventas en Olist (por orden individual).
    
    Usa esta herramienta para analisis rapido de:
    - Relacion entre puntuacion y valor de orden
    - Rating vs revenue por transaccion
    
    Para correlaciones agregadas por mes, usa get_reviews_sales_monthly_correlation.
    
    DISPONIBLE SOLO PARA: olist
    
    Returns:
        Diccionario con correlaciones basicas
    """
    print(f"[TOOL] get_reviews_sales_correlation_basic()")
    
    result = aggregate_olist_reviews_sales()
    result["tool"] = "get_reviews_sales_correlation_basic"
    return result


@tool
def get_reviews_sales_monthly_correlation() -> Dict[str, Any]:
    """
    Analiza la correlacion SEMANAL entre reviews y ventas en Olist.
    
    Usa esta herramienta cuando el usuario pregunte sobre:
    - Correlacion reviews y ventas
    - Relacion entre rating promedio y revenue
    - Impacto de reviews en ventas
    - Cross-correlation (impacto del sentimiento en ventas futuras)
    
    Incluye ANALISIS DE CROSS-CORRELATION:
    - Lag 0: Reviews(t) vs Ventas(t) - mismo periodo
    - Lag 1: Reviews(t) vs Ventas(t+1) - impacto en proxima semana
    - Lag 2: Reviews(t) vs Ventas(t+2) - impacto en 2 semanas
    
    Calcula correlaciones de Pearson con significancia estadistica.
    Basado en ~110 semanas de datos.
    
    DISPONIBLE SOLO PARA: olist
    
    Returns:
        Diccionario con correlaciones semanales, cross-correlation e interpretacion
    """
    print(f"[TOOL] get_reviews_sales_monthly_correlation()")
    
    from tfm.tools.prediction_models import get_reviews_sales_correlation
    result = get_reviews_sales_correlation()
    result["tool"] = "get_reviews_sales_correlation"
    return result


@tool
def get_sentiment_sales_monthly_correlation() -> Dict[str, Any]:
    """
    Analiza la correlacion entre sentimiento (review_score) y ventas en Olist.
    
    Usa esta herramienta cuando el usuario pregunte sobre:
    - Correlacion sentimiento y ventas
    - Impacto del sentimiento positivo/negativo en revenue
    - Relacion entre opinion de clientes y ventas futuras
    - Si mejorar el sentimiento impacta las ventas
    
    Incluye CROSS-CORRELATION para responder:
    "El sentimiento de HOY afecta las ventas de MANANA?"
    
    DISPONIBLE SOLO PARA: olist
    
    Returns:
        Diccionario con correlaciones cross-temporal e interpretacion
    """
    print(f"[TOOL] get_sentiment_sales_monthly_correlation()")
    
    from tfm.tools.prediction_models import get_sentiment_sales_correlation
    result = get_sentiment_sales_correlation()
    result["tool"] = "get_sentiment_sales_correlation"
    return result


@tool
def predict_monthly_sales(
    week_of_year: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Predice las ventas para la proxima semana usando el modelo entrenado.
    
    Usa esta herramienta cuando el usuario pregunte sobre:
    - Prediccion de ventas
    - Cuanto se vendera la proxima semana
    - Estimacion de revenue futuro
    - Forecast de ventas
    
    El modelo SEMANAL usa features:
    - revenue_lag_1, revenue_lag_2, revenue_lag_4 (ventas pasadas)
    - orders_lag_1 (ordenes semana anterior)
    - revenue_roll_4 (tendencia 4 semanas)
    - avg_review_score (rating promedio)
    - week_of_year (estacionalidad 1-52)
    
    Modelo: Lasso con R2 = 0.92 (EXCELENTE capacidad predictiva)
    
    DISPONIBLE SOLO PARA: olist
    
    Args:
        week_of_year: Semana del año a predecir (1-52). Si no se provee, predice siguiente semana.
    
    Returns:
        Diccionario con la prediccion y metricas del modelo
    """
    print(f"[TOOL] predict_weekly_sales(week_of_year={week_of_year})")
    
    from tfm.tools.prediction_models import predict_weekly_sales
    result = predict_weekly_sales(week_of_year=week_of_year)
    result["tool"] = "predict_weekly_sales"
    return result


@tool
def get_prediction_model_info() -> Dict[str, Any]:
    """
    Obtiene informacion sobre el modelo de prediccion de ventas.
    
    Usa esta herramienta cuando el usuario pregunte sobre:
    - Estado del modelo de prediccion
    - Metricas del modelo
    - Features usadas para predecir
    - Tipo de modelo entrenado
    
    DISPONIBLE SOLO PARA: olist
    
    Returns:
        Diccionario con informacion del modelo y sus metricas
    """
    print(f"[TOOL] get_prediction_model_info()")
    
    from tfm.tools.prediction_models import get_prediction_model_status
    result = get_prediction_model_status()
    result["tool"] = "get_prediction_model_info"
    return result


# =============================================================================
# TOOLS DE UTILIDAD
# =============================================================================

@tool
def get_dataset_status(dataset: Literal["yelp", "es", "olist"]) -> Dict[str, Any]:
    """
    Verifica el estado de los datos para un dataset específico.
    
    Usa esta herramienta para verificar:
    - Si los datos silver existen
    - El tamaño de los archivos
    - Qué datos están disponibles
    
    Args:
        dataset: El dataset a verificar
        
    Returns:
        Estado de los archivos silver para el dataset
    """
    print(f"[TOOL] get_dataset_status(dataset={dataset})")
    
    status = check_silver_status()
    
    if dataset == "yelp":
        keys = ["yelp_reviews", "yelp_users", "yelp_business"]
    elif dataset == "es":
        keys = ["es"]
    elif dataset == "olist":
        keys = ["olist_orders", "olist_reviews"]
    else:
        return {"error": f"Dataset desconocido: {dataset}"}
    
    result = {
        "dataset": dataset,
        "status": {}
    }
    
    for key in keys:
        if key in status:
            result["status"][key] = status[key]
    
    # Determinar si está listo para consultas
    all_exist = all(
        result["status"].get(k, {}).get("exists", False) 
        for k in keys if k in result["status"]
    )
    result["ready_for_queries"] = all_exist
    
    if not all_exist:
        result["suggestion"] = "Algunos archivos silver no existen. Ejecuta build_silver primero."
    
    return result


@tool
def build_dataset_silver(dataset: Literal["yelp", "es", "olist"]) -> Dict[str, Any]:
    """
    Construye la capa silver para un dataset específico.
    
    Usa esta herramienta cuando los datos silver no existen y necesitas
    construirlos antes de poder ejecutar análisis.
    
    NOTA: Para Yelp, usa un límite para evitar procesar millones de registros.
    
    Args:
        dataset: El dataset a construir
        
    Returns:
        Resultado de la construcción
    """
    print(f"[TOOL] build_dataset_silver(dataset={dataset})")
    
    from tfm.tools.preprocess import (
        build_silver_yelp, build_silver_es, build_silver_olist
    )
    
    try:
        if dataset == "yelp":
            # Usar límite para evitar timeout
            path = build_silver_yelp(limit=50000)
            return {
                "success": True,
                "dataset": dataset,
                "path": str(path),
                "note": "Se usó límite de 50,000 reviews para evitar timeout"
            }
        elif dataset == "es":
            path = build_silver_es()
            return {
                "success": True,
                "dataset": dataset,
                "path": str(path)
            }
        elif dataset == "olist":
            orders_path, reviews_path = build_silver_olist()
            return {
                "success": True,
                "dataset": dataset,
                "orders_path": str(orders_path),
                "reviews_path": str(reviews_path)
            }
        else:
            return {"error": f"Dataset desconocido: {dataset}"}
            
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "dataset": dataset
        }


# =============================================================================
# HELPERS
# =============================================================================

def _get_silver_key(dataset: str) -> Optional[str]:
    """Mapea dataset a key de silver_status."""
    mapping = {
        "yelp": "yelp_reviews",
        "es": "es",
        "olist": "olist_reviews"
    }
    return mapping.get(dataset)


def get_all_tools() -> List:
    """
    Retorna lista de todas las tools disponibles para bindear al LLM.
    
    El LLM usara estas herramientas para responder preguntas.
    Incluye tools de agregacion, NLP y prediccion.
    """
    from tfm.tools.nlp_models import get_nlp_tools
    
    aggregation_tools = [
        get_reviews_distribution,
        get_reviews_by_month,
        get_sales_by_month,
        get_user_stats,
        get_business_stats,
        get_ambiguous_reviews_analysis,
        get_text_length_analysis,
        get_sales_by_category,
        get_reviews_sales_correlation_basic,
        get_dataset_status,
        build_dataset_silver,
    ]
    
    prediction_tools = [
        get_reviews_sales_monthly_correlation,
        get_sentiment_sales_monthly_correlation,
        predict_monthly_sales,
        get_prediction_model_info,
    ]
    
    nlp_tools = get_nlp_tools()
    
    return aggregation_tools + prediction_tools + nlp_tools


def get_tools_summary() -> str:
    """
    Genera un resumen de las tools disponibles para incluir en prompts.
    """
    tools = get_all_tools()
    lines = ["HERRAMIENTAS DISPONIBLES:"]
    for t in tools:
        lines.append(f"- {t.name}: {t.description.split(chr(10))[0]}")
    return "\n".join(lines)
