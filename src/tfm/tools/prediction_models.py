"""
Modulo de prediccion de ventas y correlaciones para Olist.

Este modulo contiene:
- Funciones para cargar modelos de prediccion (entrenados con datos SEMANALES)
- Tools para calcular correlaciones reviews-ventas con cross-correlation
- Tool para predecir ventas semanales

Solo disponible para dataset Olist (unico con datos de ventas).

Granularidad: SEMANAL 
"""

import json
from pathlib import Path
from functools import lru_cache
from typing import Dict, Any, Optional, List

import polars as pl
import numpy as np
from scipy import stats

from tfm.config.settings import get_settings

# Cache para modelos
_prediction_model = None
_prediction_scaler = None
_model_metadata = None


@lru_cache(maxsize=1)
def load_prediction_model():
    """Carga el modelo de prediccion de ventas."""
    global _prediction_model, _prediction_scaler, _model_metadata
    
    settings = get_settings()
    prediction_dir = settings.models_dir / "prediction"
    
    model_path = prediction_dir / "sales_predictor.joblib"
    metadata_path = prediction_dir / "model_metadata.json"
    
    if not model_path.exists():
        print(f"[PREDICTION] Modelo no encontrado en {model_path}")
        return None, None, None
    
    try:
        import joblib
        _prediction_model = joblib.load(model_path)
        print(f"[PREDICTION] Modelo cargado: {model_path}")
        
        # Cargar metadata
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                _model_metadata = json.load(f)
            print(f"[PREDICTION] Metadata cargada: {_model_metadata.get('model_type', 'unknown')}")
            print(f"[PREDICTION] Granularidad: {_model_metadata.get('granularity', 'unknown')}")
        
        # Cargar scaler si existe
        scaler_path = prediction_dir / "sales_scaler.joblib"
        if scaler_path.exists():
            _prediction_scaler = joblib.load(scaler_path)
            print(f"[PREDICTION] Scaler cargado")
        
        return _prediction_model, _prediction_scaler, _model_metadata
        
    except Exception as e:
        print(f"[PREDICTION] Error cargando modelo: {e}")
        return None, None, None


def load_correlation_results() -> Optional[Dict[str, Any]]:
    """Carga resultados de correlacion pre-calculados (semanales con cross-correlation)."""
    settings = get_settings()
    corr_path = settings.models_dir / "prediction" / "correlation_results.json"
    
    if not corr_path.exists():
        return None
    
    try:
        with open(corr_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"[PREDICTION] Error cargando correlaciones: {e}")
        return None


def load_latest_sales_data() -> Optional[List[Dict[str, Any]]]:
    """Carga ultimas semanas de datos para prediccion."""
    settings = get_settings()
    data_path = settings.silver_dir / "latest_sales_data.json"
    
    if not data_path.exists():
        return None
    
    try:
        with open(data_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"[PREDICTION] Error cargando datos recientes: {e}")
        return None


def get_reviews_sales_correlation(
    recalculate: bool = False,
) -> Dict[str, Any]:
    """
    Obtiene correlacion entre reviews y ventas para Olist.
    
    Incluye correlaciones basicas y CROSS-CORRELATION (lag analysis):
    - Lag 0: Reviews(t) vs Ventas(t)
    - Lag 1: Reviews(t) vs Ventas(t+1) - impacto en proxima semana
    - Lag 2: Reviews(t) vs Ventas(t+2) - impacto en 2 semanas
    
    Args:
        recalculate: Si True, recalcula en vez de usar cache
        
    Returns:
        Diccionario con correlaciones basicas y cruzadas
    """
    # Usar resultados pre-calculados (semanales)
    if not recalculate:
        cached = load_correlation_results()
        if cached:
            basic = cached.get("basic_correlations", {})
            cross = cached.get("cross_correlations", {})
            n_samples = cached.get("n_samples", 0)
            granularity = cached.get("granularity", "weekly")
            
            # Interpretar cross-correlation
            cross_interpretation = _interpret_cross_correlations(cross)
            
            return {
                "dataset": "olist",
                "type": "reviews_sales",
                "granularity": granularity,
                "samples_analyzed": n_samples,
                "source": "pre-calculated",
                "basic_correlations": {
                    "review_score_vs_revenue": {
                        "r": basic.get("avg_review_score_vs_total_revenue", {}).get("r"),
                        "p": basic.get("avg_review_score_vs_total_revenue", {}).get("p"),
                        "significant": (basic.get("avg_review_score_vs_total_revenue", {}).get("p") or 1) < 0.05
                    },
                    "review_count_vs_revenue": {
                        "r": basic.get("review_count_vs_total_revenue", {}).get("r"),
                        "p": basic.get("review_count_vs_total_revenue", {}).get("p"),
                        "significant": (basic.get("review_count_vs_total_revenue", {}).get("p") or 1) < 0.05
                    },
                },
                "cross_correlations": {
                    "review_score_t_vs_revenue_t0": cross.get("review_score_t_vs_revenue_t0", {}),
                    "review_score_t_vs_revenue_t1": cross.get("review_score_t_vs_revenue_t1", {}),
                    "review_score_t_vs_revenue_t2": cross.get("review_score_t_vs_revenue_t2", {}),
                },
                "interpretation": cross_interpretation,
            }
    
    return {"error": "Ejecuta notebook 07_sales_prediction.ipynb para generar correlaciones semanales"}


def get_sentiment_sales_correlation(
    recalculate: bool = False,
) -> Dict[str, Any]:
    """
    Obtiene correlacion entre sentimiento (review_score como proxy) y ventas para Olist.
    
    Nota: Usa review_score como proxy del sentimiento. Para un analisis mas
    profundo con NLP, ejecutar el notebook 07 con el modelo de sentimiento.
    
    Returns:
        Diccionario con correlaciones y cross-correlation
    """
    # Usar resultados pre-calculados
    if not recalculate:
        cached = load_correlation_results()
        if cached:
            cross = cached.get("cross_correlations", {})
            n_samples = cached.get("n_samples", 0)
            
            # Extraer correlaciones de review_score (proxy de sentimiento)
            return {
                "dataset": "olist",
                "type": "sentiment_sales",
                "granularity": cached.get("granularity", "weekly"),
                "samples_analyzed": n_samples,
                "source": "pre-calculated",
                "note": "Usando review_score como proxy de sentimiento",
                "correlations": {
                    "sentiment_t_vs_revenue_t0": {
                        "r": cross.get("review_score_t_vs_revenue_t0", {}).get("r"),
                        "p": cross.get("review_score_t_vs_revenue_t0", {}).get("p"),
                        "description": "Sentimiento esta semana vs Revenue esta semana"
                    },
                    "sentiment_t_vs_revenue_t1": {
                        "r": cross.get("review_score_t_vs_revenue_t1", {}).get("r"),
                        "p": cross.get("review_score_t_vs_revenue_t1", {}).get("p"),
                        "description": "Sentimiento esta semana vs Revenue proxima semana"
                    },
                    "sentiment_t_vs_revenue_t2": {
                        "r": cross.get("review_score_t_vs_revenue_t2", {}).get("r"),
                        "p": cross.get("review_score_t_vs_revenue_t2", {}).get("p"),
                        "description": "Sentimiento esta semana vs Revenue en 2 semanas"
                    },
                },
                "interpretation": _interpret_cross_correlations(cross),
            }
    
    return {"error": "Ejecuta notebook 07_sales_prediction.ipynb para generar correlaciones"}


def predict_weekly_sales(
    week_of_year: Optional[int] = None,
    revenue_lag_1: Optional[float] = None,
    revenue_lag_2: Optional[float] = None,
    revenue_lag_4: Optional[float] = None,
    orders_lag_1: Optional[int] = None,
    avg_review_score: Optional[float] = None,
    use_latest_data: bool = True,
) -> Dict[str, Any]:
    """
    Predice ventas para la proxima semana usando el modelo entrenado.
    
    El modelo usa granularidad SEMANAL con las siguientes features:
    - revenue_lag_1, revenue_lag_2, revenue_lag_4: Revenue de semanas anteriores
    - orders_lag_1: Ordenes de la semana anterior
    - revenue_roll_4: Media movil de 4 semanas
    - avg_review_score: Rating promedio
    - week_of_year: Semana del ano (1-52) para estacionalidad
    
    Args:
        week_of_year: Semana del ano a predecir (1-52). Si no se provee, usa la siguiente.
        revenue_lag_1: Revenue de la semana anterior
        revenue_lag_2: Revenue de hace 2 semanas
        revenue_lag_4: Revenue de hace 4 semanas
        orders_lag_1: Ordenes de la semana anterior
        avg_review_score: Rating promedio actual
        use_latest_data: Si True, usa los datos mas recientes guardados
        
    Returns:
        Diccionario con prediccion y metricas del modelo
    """
    model, scaler, metadata = load_prediction_model()
    
    if model is None:
        return {
            "error": "Modelo de prediccion no disponible",
            "suggestion": "Ejecutar notebook 07_sales_prediction.ipynb primero"
        }
    
    # Usar datos mas recientes si estan disponibles
    if use_latest_data:
        latest = load_latest_sales_data()
        if latest and len(latest) > 0:
            # Tomar la semana mas reciente con datos
            last_week = latest[-1]
            
            if revenue_lag_1 is None:
                revenue_lag_1 = last_week.get("total_revenue", 0)
            if revenue_lag_2 is None:
                revenue_lag_2 = last_week.get("revenue_lag_1", revenue_lag_1 * 0.95)
            if revenue_lag_4 is None:
                revenue_lag_4 = last_week.get("revenue_lag_4", revenue_lag_1 * 0.9)
            if orders_lag_1 is None:
                orders_lag_1 = int(last_week.get("order_count", 1000))
            if avg_review_score is None:
                avg_review_score = last_week.get("avg_review_score", 4.0)
            if week_of_year is None:
                # Siguiente semana
                week_of_year = (last_week.get("week_of_year", 1) % 52) + 1
    
    # Valores por defecto
    revenue_lag_1 = revenue_lag_1 or 200000
    revenue_lag_2 = revenue_lag_2 or revenue_lag_1 * 0.95
    revenue_lag_4 = revenue_lag_4 or revenue_lag_1 * 0.9
    orders_lag_1 = orders_lag_1 or 1500
    avg_review_score = avg_review_score or 4.1
    week_of_year = week_of_year or 1
    
    # Calcular media movil
    revenue_roll_4 = (revenue_lag_1 + revenue_lag_2 + 
                      (revenue_lag_2 * 0.97) + revenue_lag_4) / 4
    
    # Construir features en el orden correcto
    features = metadata.get("features", [
        "revenue_lag_1", "revenue_lag_2", "revenue_lag_4", 
        "orders_lag_1", "revenue_roll_4", "avg_review_score", "week_of_year"
    ])
    
    feature_values = {
        "revenue_lag_1": revenue_lag_1,
        "revenue_lag_2": revenue_lag_2,
        "revenue_lag_4": revenue_lag_4,
        "orders_lag_1": orders_lag_1,
        "revenue_roll_4": revenue_roll_4,
        "avg_review_score": avg_review_score,
        "week_of_year": week_of_year,
    }
    
    # Crear array de features
    X = np.array([[feature_values.get(f, 0) for f in features]])
    
    # Aplicar scaler si existe
    if scaler is not None:
        X = scaler.transform(X)
    
    # Predecir
    try:
        prediction = model.predict(X)[0]
        
        return {
            "dataset": "olist",
            "granularity": "weekly",
            "prediction_for_week": week_of_year,
            "predicted_revenue": float(prediction),
            "predicted_revenue_formatted": f"R$ {prediction:,.2f}",
            "model_type": metadata.get("model_type", "unknown"),
            "model_metrics": metadata.get("metrics", {}),
            "r2_score": metadata.get("metrics", {}).get("r2", None),
            "mae": metadata.get("metrics", {}).get("mae", None),
            "input_features": feature_values,
            "note": "Prediccion basada en modelo SEMANAL con R2={:.3f}".format(
                metadata.get("metrics", {}).get("r2", 0)
            )
        }
        
    except Exception as e:
        return {"error": f"Error en prediccion: {str(e)}"}


def get_prediction_model_status() -> Dict[str, Any]:
    """
    Verifica el estado del modelo de prediccion.
    
    Returns:
        Diccionario con estado del modelo y metricas
    """
    model, scaler, metadata = load_prediction_model()
    
    if model is None:
        return {
            "model_available": False,
            "error": "Modelo no encontrado",
            "suggestion": "Ejecutar notebook 07_sales_prediction.ipynb"
        }
    
    return {
        "model_available": True,
        "model_type": metadata.get("model_type", "unknown"),
        "model_class": metadata.get("model_class", "unknown"),
        "granularity": metadata.get("granularity", "weekly"),
        "features": metadata.get("features", []),
        "metrics": metadata.get("metrics", {}),
        "training_samples": metadata.get("training_samples", 0),
        "date_range": metadata.get("date_range", {}),
        "scaler_available": scaler is not None,
        "interpretation": _interpret_model_quality(metadata.get("metrics", {}))
    }


def _interpret_cross_correlations(cross_corr: Dict[str, Any]) -> str:
    """Genera interpretacion de correlaciones cruzadas (lag analysis)."""
    lines = ["ANALISIS DE CORRELACION CRUZADA (Cross-Correlation):"]
    
    # Correlacion lag 0
    lag0 = cross_corr.get("review_score_t_vs_revenue_t0", {})
    if lag0.get("r") is not None:
        r = lag0["r"]
        p = lag0.get("p", 1)
        sig = "*" if p < 0.05 else ""
        lines.append(f"- Semana t (mismo periodo): r={r:.3f} {sig}")
    
    # Correlacion lag 1
    lag1 = cross_corr.get("review_score_t_vs_revenue_t1", {})
    if lag1.get("r") is not None:
        r = lag1["r"]
        p = lag1.get("p", 1)
        sig = "*" if p < 0.05 else ""
        lines.append(f"- Semana t+1 (proxima semana): r={r:.3f} {sig}")
    
    # Correlacion lag 2
    lag2 = cross_corr.get("review_score_t_vs_revenue_t2", {})
    if lag2.get("r") is not None:
        r = lag2["r"]
        p = lag2.get("p", 1)
        sig = "*" if p < 0.05 else ""
        lines.append(f"- Semana t+2 (en 2 semanas): r={r:.3f} {sig}")
    
    lines.append("\n* = estadisticamente significativo (p < 0.05)")
    
    # Interpretacion
    if lag1.get("r") and lag1.get("p", 1) < 0.05:
        lines.append("\nCONCLUSION: El sentimiento (review_score) de esta semana")
        lines.append("TIENE correlacion significativa con las ventas de la proxima semana.")
    else:
        lines.append("\nCONCLUSION: El sentimiento no es un predictor fuerte")
        lines.append("de las ventas de la proxima semana.")
    
    return "\n".join(lines)


def _interpret_model_quality(metrics: Dict[str, Any]) -> str:
    """Interpreta la calidad del modelo basado en sus metricas."""
    r2 = metrics.get("r2", 0)
    mae = metrics.get("mae", 0)
    
    if r2 > 0.8:
        quality = "EXCELENTE - El modelo explica mas del 80% de la varianza"
    elif r2 > 0.6:
        quality = "BUENO - El modelo tiene buena capacidad predictiva"
    elif r2 > 0.4:
        quality = "MODERADO - El modelo tiene capacidad predictiva limitada"
    elif r2 > 0:
        quality = "BAJO - Las predicciones deben tomarse con cautela"
    else:
        quality = "MUY BAJO - Se recomienda usar promedios historicos"
    
    return f"{quality} (R2={r2:.3f}, MAE=R${mae:,.0f})"
