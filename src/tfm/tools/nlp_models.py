"""
Tools de NLP que utilizan modelos ML entrenados.

Este modulo proporciona herramientas para:
- Analisis de sentimiento usando TF-IDF + SVM/LogisticRegression
- Extraccion de aspectos de resenias
- Sentimiento por aspecto

Los modelos se cargan desde models/sentiment/ y models/aspects/
"""

import re
import json
from pathlib import Path
from typing import Dict, Any, List, Literal, Optional, Set
from functools import lru_cache

import joblib
from langchain_core.tools import tool

from tfm.config.settings import PROJECT_ROOT


# =============================================================================
# CARGA DE MODELOS Y ARTEFACTOS
# =============================================================================

MODELS_DIR = PROJECT_ROOT / "models"
SENTIMENT_DIR = MODELS_DIR / "sentiment"
ASPECTS_DIR = MODELS_DIR / "aspects"


@lru_cache(maxsize=1)
def load_sentiment_model(model_type: str = "unified_svm"):
    """Carga modelo de sentimiento desde disco."""
    model_path = SENTIMENT_DIR / f"{model_type}_sentiment.joblib"
    if model_path.exists():
        return joblib.load(model_path)
    return None


@lru_cache(maxsize=1)
def load_aspect_model():
    """Carga modelo de aspectos desde disco."""
    models = {}
    
    # Clasificador
    clf_path = ASPECTS_DIR / "aspect_classifier_lr.joblib"
    if clf_path.exists():
        models["classifier"] = joblib.load(clf_path)
    
    # TF-IDF
    tfidf_path = ASPECTS_DIR / "aspect_tfidf.joblib"
    if tfidf_path.exists():
        models["tfidf"] = joblib.load(tfidf_path)
    
    # MultiLabelBinarizer
    mlb_path = ASPECTS_DIR / "aspect_mlb.joblib"
    if mlb_path.exists():
        models["mlb"] = joblib.load(mlb_path)
    
    return models if models else None


@lru_cache(maxsize=1)
def load_aspect_taxonomy() -> Dict[str, Set[str]]:
    """Carga taxonomia de aspectos."""
    taxonomy_path = ASPECTS_DIR / "aspect_taxonomy.json"
    if taxonomy_path.exists():
        with open(taxonomy_path, "r", encoding="utf-8") as f:
            taxonomy = json.load(f)
        # Aplanar keywords por aspecto
        aspect_keywords = {}
        for aspect, langs in taxonomy.items():
            all_keywords = set()
            for lang_keywords in langs.values():
                all_keywords.update([k.lower() for k in lang_keywords])
            aspect_keywords[aspect] = all_keywords
        return aspect_keywords
    return {}


@lru_cache(maxsize=1)
def load_sentiment_lexicon() -> Dict[str, Set[str]]:
    """Carga lexicon de sentimiento."""
    lexicon_path = ASPECTS_DIR / "sentiment_lexicon.json"
    if lexicon_path.exists():
        with open(lexicon_path, "r", encoding="utf-8") as f:
            lexicon = json.load(f)
        # Aplanar por sentimiento
        result = {"positive": set(), "negative": set()}
        for sentiment, langs in lexicon.items():
            for lang_words in langs.values():
                result[sentiment].update(lang_words)
        return result
    return {"positive": set(), "negative": set()}


# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def clean_text_for_ml(text: str) -> str:
    """Limpieza de texto para modelos ML."""
    if not text or not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\b\d+\b', '', text)
    text = re.sub(r'[^a-zA-Z\s\u00C0-\u017F]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_aspects_rules(text: str) -> List[str]:
    """Extrae aspectos usando reglas basadas en keywords."""
    if not text:
        return []
    
    taxonomy = load_aspect_taxonomy()
    if not taxonomy:
        return []
    
    text_lower = text.lower()
    words = set(text_lower.split())
    
    found_aspects = []
    for aspect, keywords in taxonomy.items():
        if words & keywords:
            found_aspects.append(aspect)
            continue
        for keyword in keywords:
            if ' ' in keyword and keyword in text_lower:
                found_aspects.append(aspect)
                break
    
    return found_aspects


def get_aspect_sentiment_local(text: str, aspect: str) -> str:
    """Determina sentimiento de un aspecto usando ventana de contexto."""
    if not text:
        return "neutral"
    
    taxonomy = load_aspect_taxonomy()
    lexicon = load_sentiment_lexicon()
    
    if not taxonomy or not lexicon:
        return "neutral"
    
    words = text.lower().split()
    aspect_keywords = taxonomy.get(aspect, set())
    
    # Encontrar posiciones del aspecto
    aspect_positions = []
    for i, word in enumerate(words):
        if word in aspect_keywords:
            aspect_positions.append(i)
    
    if not aspect_positions:
        return "neutral"
    
    # Buscar sentimiento en ventana
    window_size = 5
    pos_count = 0
    neg_count = 0
    
    for pos in aspect_positions:
        start = max(0, pos - window_size)
        end = min(len(words), pos + window_size + 1)
        window = words[start:end]
        
        for word in window:
            if word in lexicon["positive"]:
                pos_count += 1
            elif word in lexicon["negative"]:
                neg_count += 1
    
    if pos_count > neg_count:
        return "positive"
    elif neg_count > pos_count:
        return "negative"
    return "neutral"


# =============================================================================
# TOOLS DE NLP PARA EL LLM
# =============================================================================

@tool
def analyze_sentiment(
    text: str,
    model: Literal["unified", "yelp", "es", "olist"] = "unified"
) -> Dict[str, Any]:
    """
    Analiza el sentimiento de un texto usando modelos ML entrenados.
    
    Usa esta herramienta cuando el usuario:
    - Quiera saber el sentimiento de una resenia especifica
    - Pregunte si un texto es positivo, negativo o neutral
    - Necesite clasificar el tono de un comentario
    
    Args:
        text: El texto a analizar
        model: Modelo a usar. "unified" funciona para todos los idiomas.
               Usa "yelp" para ingles, "es" para espanol, "olist" para portugues.
    
    Returns:
        Diccionario con sentimiento predicho y confianza
    """
    print(f"[TOOL] analyze_sentiment(text={text[:50]}..., model={model})")
    
    if not text or len(text.strip()) < 5:
        return {
            "error": "Texto muy corto para analizar",
            "text": text
        }
    
    # Determinar modelo a cargar
    if model == "unified":
        model_name = "unified_svm"
    else:
        model_name = f"{model}_svm"
    
    pipeline = load_sentiment_model(model_name)
    
    if pipeline is None:
        # Fallback a unified
        pipeline = load_sentiment_model("unified_svm")
        if pipeline is None:
            return {
                "error": f"Modelo {model_name} no encontrado. Ejecuta el notebook de entrenamiento.",
                "suggestion": "Ejecuta notebooks/05_nlp_sentiment_models.ipynb"
            }
    
    # Limpiar y predecir
    text_clean = clean_text_for_ml(text)
    prediction = pipeline.predict([text_clean])[0]
    
    # Obtener probabilidades si es LogisticRegression
    confidence = None
    if hasattr(pipeline, "predict_proba"):
        try:
            proba = pipeline.predict_proba([text_clean])[0]
            confidence = float(max(proba))
        except:
            pass
    
    return {
        "text": text[:100] + "..." if len(text) > 100 else text,
        "sentiment": prediction,
        "model_used": model_name,
        "confidence": confidence,
        "tool": "analyze_sentiment"
    }


@tool
def analyze_sentiment_batch(
    texts: List[str],
    model: Literal["unified", "yelp", "es", "olist"] = "unified"
) -> Dict[str, Any]:
    """
    Analiza el sentimiento de multiples textos en lote.
    
    Usa esta herramienta cuando necesites analizar varias resenias a la vez.
    Es mas eficiente que llamar analyze_sentiment multiples veces.
    
    Args:
        texts: Lista de textos a analizar
        model: Modelo a usar (unified, yelp, es, olist)
    
    Returns:
        Diccionario con predicciones para cada texto y resumen
    """
    print(f"[TOOL] analyze_sentiment_batch(n_texts={len(texts)}, model={model})")
    
    if not texts:
        return {"error": "Lista de textos vacia"}
    
    if model == "unified":
        model_name = "unified_svm"
    else:
        model_name = f"{model}_svm"
    
    pipeline = load_sentiment_model(model_name)
    if pipeline is None:
        pipeline = load_sentiment_model("unified_svm")
        if pipeline is None:
            return {"error": "Modelo no encontrado"}
    
    # Limpiar y predecir
    texts_clean = [clean_text_for_ml(t) for t in texts]
    predictions = pipeline.predict(texts_clean)
    
    # Conteo de sentimientos
    sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
    results = []
    
    for text, pred in zip(texts, predictions):
        sentiment_counts[pred] = sentiment_counts.get(pred, 0) + 1
        results.append({
            "text": text[:50] + "..." if len(text) > 50 else text,
            "sentiment": pred
        })
    
    return {
        "total_analyzed": len(texts),
        "sentiment_distribution": sentiment_counts,
        "predictions": results,
        "model_used": model_name,
        "tool": "analyze_sentiment_batch"
    }


@tool
def extract_aspects(text: str) -> Dict[str, Any]:
    """
    Extrae los aspectos mencionados en una resenia.
    
    Los aspectos son categorias como: calidad, precio, envio, servicio, etc.
    Usa esta herramienta cuando el usuario pregunte:
    - De que habla esta resenia?
    - Que aspectos menciona el cliente?
    - Cuales son los temas de esta resenia?
    
    Args:
        text: El texto de la resenia a analizar
    
    Returns:
        Lista de aspectos encontrados con sus sentimientos
    """
    print(f"[TOOL] extract_aspects(text={text[:50]}...)")
    
    if not text or len(text.strip()) < 10:
        return {
            "error": "Texto muy corto para extraer aspectos",
            "text": text
        }
    
    text_clean = clean_text_for_ml(text)
    
    # Extraer aspectos con reglas
    aspects = extract_aspects_rules(text_clean)
    
    if not aspects:
        return {
            "text": text[:100] + "..." if len(text) > 100 else text,
            "aspects": [],
            "message": "No se encontraron aspectos especificos en el texto",
            "tool": "extract_aspects"
        }
    
    # Obtener sentimiento por aspecto
    aspect_sentiments = {}
    for aspect in aspects:
        sentiment = get_aspect_sentiment_local(text_clean, aspect)
        aspect_sentiments[aspect] = sentiment
    
    return {
        "text": text[:100] + "..." if len(text) > 100 else text,
        "aspects": aspects,
        "aspect_sentiments": aspect_sentiments,
        "num_aspects": len(aspects),
        "tool": "extract_aspects"
    }


@tool
def analyze_review_complete(
    text: str,
    model: Literal["unified", "yelp", "es", "olist"] = "unified"
) -> Dict[str, Any]:
    """
    Realiza un analisis completo de una resenia: sentimiento global + aspectos.
    
    Esta es la herramienta mas completa para analizar una resenia individual.
    Combina analisis de sentimiento y extraccion de aspectos.
    
    Usa esta herramienta cuando el usuario quiera un analisis detallado
    de una resenia especifica.
    
    Args:
        text: El texto de la resenia
        model: Modelo de sentimiento a usar
    
    Returns:
        Analisis completo con sentimiento global, aspectos y sentimiento por aspecto
    """
    print(f"[TOOL] analyze_review_complete(text={text[:50]}...)")
    
    if not text or len(text.strip()) < 10:
        return {"error": "Texto muy corto para analizar"}
    
    text_clean = clean_text_for_ml(text)
    
    # Sentimiento global
    if model == "unified":
        model_name = "unified_svm"
    else:
        model_name = f"{model}_svm"
    
    pipeline = load_sentiment_model(model_name)
    global_sentiment = "unknown"
    
    if pipeline:
        global_sentiment = pipeline.predict([text_clean])[0]
    
    # Aspectos
    aspects = extract_aspects_rules(text_clean)
    aspect_sentiments = {}
    
    for aspect in aspects:
        sentiment = get_aspect_sentiment_local(text_clean, aspect)
        aspect_sentiments[aspect] = sentiment
    
    return {
        "text": text[:150] + "..." if len(text) > 150 else text,
        "global_sentiment": global_sentiment,
        "model_used": model_name,
        "aspects": aspects,
        "aspect_sentiments": aspect_sentiments,
        "summary": {
            "overall": global_sentiment,
            "num_aspects": len(aspects),
            "positive_aspects": [a for a, s in aspect_sentiments.items() if s == "positive"],
            "negative_aspects": [a for a, s in aspect_sentiments.items() if s == "negative"]
        },
        "tool": "analyze_review_complete"
    }


@tool
def get_nlp_models_status() -> Dict[str, Any]:
    """
    Verifica el estado de los modelos NLP disponibles.
    
    Usa esta herramienta para saber que modelos estan entrenados
    y disponibles para usar.
    
    Returns:
        Estado de cada modelo y artefacto NLP
    """
    print("[TOOL] get_nlp_models_status()")
    
    status = {
        "sentiment_models": {},
        "aspect_models": {},
        "ready": False
    }
    
    # Verificar modelos de sentimiento
    sentiment_models = [
        "unified_svm", "unified_logistic",
        "yelp_svm", "yelp_logistic",
        "es_svm", "es_logistic",
        "olist_svm", "olist_logistic"
    ]
    
    for model_name in sentiment_models:
        path = SENTIMENT_DIR / f"{model_name}_sentiment.joblib"
        status["sentiment_models"][model_name] = path.exists()
    
    # Verificar modelos de aspectos
    aspect_files = [
        "aspect_classifier_lr.joblib",
        "aspect_classifier_svm.joblib",
        "aspect_tfidf.joblib",
        "aspect_mlb.joblib",
        "aspect_taxonomy.json",
        "sentiment_lexicon.json"
    ]
    
    for fname in aspect_files:
        path = ASPECTS_DIR / fname
        status["aspect_models"][fname] = path.exists()
    
    # Verificar si hay al menos un modelo funcional
    status["ready"] = (
        any(status["sentiment_models"].values()) and
        status["aspect_models"].get("aspect_taxonomy.json", False)
    )
    
    return status


# =============================================================================
# TOOLS DE NLP PARA DATASETS COMPLETOS
# =============================================================================

@tool
def get_sentiment_distribution(
    dataset: Literal["yelp", "es", "olist"],
    year: Optional[int] = None,
    stars: Optional[int] = None
) -> Dict[str, Any]:
    """
    Obtiene la distribucion de sentimiento (positivo/neutral/negativo) para un dataset.
    
    Usa esta herramienta cuando el usuario pregunte:
    - Cual es la distribucion de sentimiento?
    - Cuantas resenias positivas/negativas hay?
    - Que porcentaje de resenias son positivas?
    - Cual es el sentimiento promedio?
    
    DISPONIBLE PARA: yelp, es, olist (TODOS los datasets)
    
    Args:
        dataset: Dataset a analizar
        year: Filtrar por anio (solo yelp, olist)
        stars: Filtrar por estrellas (1-5), util para analizar reviews de 3 estrellas
    
    Returns:
        Distribucion de sentimiento con conteos y porcentajes
    """
    print(f"[TOOL] get_sentiment_distribution(dataset={dataset}, year={year}, stars={stars})")
    
    import polars as pl
    from tfm.config.settings import get_settings, SILVER_FILES
    
    settings = get_settings()
    
    # Cargar datos silver
    if dataset == "yelp":
        silver_path = settings.silver_dir / SILVER_FILES.get("yelp_reviews", "yelp_reviews.parquet")
    elif dataset == "es":
        silver_path = settings.silver_dir / SILVER_FILES.get("es", "es_reviews.parquet")
    elif dataset == "olist":
        silver_path = settings.silver_dir / SILVER_FILES.get("olist_reviews", "olist_reviews.parquet")
    else:
        return {"error": f"Dataset no soportado: {dataset}"}
    
    if not silver_path.exists():
        return {"error": f"Datos silver no existen para {dataset}. Ejecuta build_silver primero."}
    
    df = pl.read_parquet(silver_path)
    original_count = df.height
    
    # Filtrar por anio si aplica
    if year and dataset in ["yelp", "olist"]:
        if "year" in df.columns:
            df = df.filter(pl.col("year") == year)
    
    # Filtrar por stars si aplica
    if stars:
        stars_col = "stars" if "stars" in df.columns else "review_score"
        if stars_col in df.columns:
            df = df.filter(pl.col(stars_col) == stars)
    
    if df.height == 0:
        return {"error": "No hay datos con los filtros especificados"}
    
    # Cargar modelo
    pipeline = load_sentiment_model("unified_svm")
    if pipeline is None:
        return {"error": "Modelo de sentimiento no disponible. Ejecuta el notebook de entrenamiento."}
    
    # Obtener textos
    text_col = "text" if "text" in df.columns else "review_body"
    if text_col not in df.columns:
        return {"error": f"Columna de texto no encontrada en {dataset}"}
    
    texts = df[text_col].drop_nulls().to_list()
    texts_clean = [clean_text_for_ml(t) for t in texts if t]
    
    # Filtrar textos vacios
    texts_clean = [t for t in texts_clean if len(t) > 5]
    
    if not texts_clean:
        return {"error": "No hay textos validos para analizar"}
    
    # Predecir
    predictions = pipeline.predict(texts_clean)
    
    # Contar
    from collections import Counter
    counts = Counter(predictions)
    total = len(predictions)
    
    distribution = {
        "positive": counts.get("positive", 0),
        "neutral": counts.get("neutral", 0),
        "negative": counts.get("negative", 0)
    }
    
    percentages = {
        "positive_pct": round(distribution["positive"] / total * 100, 1),
        "neutral_pct": round(distribution["neutral"] / total * 100, 1),
        "negative_pct": round(distribution["negative"] / total * 100, 1)
    }
    
    # Calcular sentimiento promedio (-1 a 1)
    sentiment_scores = {"positive": 1, "neutral": 0, "negative": -1}
    avg_sentiment = sum(sentiment_scores[p] for p in predictions) / total
    
    return {
        "dataset": dataset,
        "filters": {"year": year, "stars": stars},
        "total_analyzed": total,
        "original_count": original_count,
        "distribution": distribution,
        "percentages": percentages,
        "average_sentiment": round(avg_sentiment, 3),
        "sentiment_label": "positive" if avg_sentiment > 0.2 else ("negative" if avg_sentiment < -0.2 else "neutral"),
        "tool": "get_sentiment_distribution"
    }


@tool
def get_aspect_distribution(
    dataset: Literal["yelp", "es", "olist"],
    sentiment_filter: Optional[Literal["positive", "negative", "neutral"]] = None,
    stars: Optional[int] = None
) -> Dict[str, Any]:
    """
    Obtiene la distribucion de aspectos mencionados en las resenias.
    
    Usa esta herramienta cuando el usuario pregunte:
    - Cuales son los aspectos mas mencionados?
    - De que hablan las resenias negativas?
    - Que aspectos tienen las resenias de 3 estrellas?
    - Cuales son los problemas mas comunes?
    
    DISPONIBLE PARA: yelp, es, olist (TODOS los datasets)
    
    Args:
        dataset: Dataset a analizar
        sentiment_filter: Filtrar por sentimiento (positive/negative/neutral)
        stars: Filtrar por estrellas (1-5)
    
    Returns:
        Distribucion de aspectos con frecuencias
    """
    print(f"[TOOL] get_aspect_distribution(dataset={dataset}, sentiment={sentiment_filter}, stars={stars})")
    
    import polars as pl
    from tfm.config.settings import get_settings, SILVER_FILES
    from collections import Counter
    
    settings = get_settings()
    
    # Cargar datos silver
    if dataset == "yelp":
        silver_path = settings.silver_dir / SILVER_FILES.get("yelp_reviews", "yelp_reviews.parquet")
    elif dataset == "es":
        silver_path = settings.silver_dir / SILVER_FILES.get("es", "es_reviews.parquet")
    elif dataset == "olist":
        silver_path = settings.silver_dir / SILVER_FILES.get("olist_reviews", "olist_reviews.parquet")
    else:
        return {"error": f"Dataset no soportado: {dataset}"}
    
    if not silver_path.exists():
        return {"error": f"Datos silver no existen para {dataset}"}
    
    df = pl.read_parquet(silver_path)
    
    # Filtrar por stars
    if stars:
        stars_col = "stars" if "stars" in df.columns else "review_score"
        if stars_col in df.columns:
            df = df.filter(pl.col(stars_col) == stars)
    
    if df.height == 0:
        return {"error": "No hay datos con los filtros especificados"}
    
    # Obtener textos
    text_col = "text" if "text" in df.columns else "review_body"
    if text_col not in df.columns:
        return {"error": f"Columna de texto no encontrada"}
    
    texts = df[text_col].drop_nulls().to_list()
    
    # Cargar modelo de sentimiento si hay filtro
    pipeline = None
    if sentiment_filter:
        pipeline = load_sentiment_model("unified_svm")
    
    # Procesar
    aspect_counts = Counter()
    aspect_sentiment_counts = {aspect: Counter() for aspect in load_aspect_taxonomy().keys()}
    total_with_aspects = 0
    
    for text in texts:
        if not text:
            continue
        
        text_clean = clean_text_for_ml(text)
        if len(text_clean) < 5:
            continue
        
        # Verificar filtro de sentimiento
        if sentiment_filter and pipeline:
            pred = pipeline.predict([text_clean])[0]
            if pred != sentiment_filter:
                continue
        
        # Extraer aspectos
        aspects = extract_aspects_rules(text_clean)
        if aspects:
            total_with_aspects += 1
            aspect_counts.update(aspects)
            
            # Sentimiento por aspecto
            for aspect in aspects:
                sent = get_aspect_sentiment_local(text_clean, aspect)
                aspect_sentiment_counts[aspect][sent] += 1
    
    if not aspect_counts:
        return {"error": "No se encontraron aspectos en las resenias filtradas"}
    
    # Formatear resultados
    total_aspects = sum(aspect_counts.values())
    aspect_distribution = []
    
    for aspect, count in aspect_counts.most_common():
        sentiments = aspect_sentiment_counts[aspect]
        aspect_distribution.append({
            "aspect": aspect,
            "count": count,
            "percentage": round(count / total_with_aspects * 100, 1),
            "sentiment_breakdown": dict(sentiments)
        })
    
    return {
        "dataset": dataset,
        "filters": {"sentiment": sentiment_filter, "stars": stars},
        "total_reviews_analyzed": len(texts),
        "reviews_with_aspects": total_with_aspects,
        "total_aspect_mentions": total_aspects,
        "aspect_distribution": aspect_distribution,
        "top_aspects": [a["aspect"] for a in aspect_distribution[:5]],
        "tool": "get_aspect_distribution"
    }


@tool
def get_ambiguous_reviews_sentiment(
    dataset: Literal["yelp", "es", "olist"]
) -> Dict[str, Any]:
    """
    Analiza el sentimiento de las resenias ambiguas (3 estrellas).
    
    Las resenias de 3 estrellas son dificiles de clasificar solo por el rating.
    Esta herramienta usa NLP para determinar si realmente son neutrales
    o tienen tendencia positiva/negativa.
    
    Usa esta herramienta cuando el usuario pregunte:
    - Cual es el sentimiento real de las resenias de 3 estrellas?
    - Las resenias ambiguas son mas positivas o negativas?
    - Analiza las resenias de 3 estrellas con NLP
    
    DISPONIBLE PARA: yelp, es, olist (TODOS los datasets)
    
    Args:
        dataset: Dataset a analizar
    
    Returns:
        Analisis de sentimiento de resenias de 3 estrellas
    """
    print(f"[TOOL] get_ambiguous_reviews_sentiment(dataset={dataset})")
    
    # Usar la tool de distribucion con filtro de stars=3
    result = get_sentiment_distribution.invoke({
        "dataset": dataset,
        "stars": 3
    })
    
    if "error" in result:
        return result
    
    # Agregar interpretacion
    interpretation = ""
    avg = result.get("average_sentiment", 0)
    
    if avg > 0.15:
        interpretation = "Las resenias de 3 estrellas tienen tendencia POSITIVA segun el analisis NLP"
    elif avg < -0.15:
        interpretation = "Las resenias de 3 estrellas tienen tendencia NEGATIVA segun el analisis NLP"
    else:
        interpretation = "Las resenias de 3 estrellas son genuinamente NEUTRALES"
    
    result["interpretation"] = interpretation
    result["is_ambiguous_analysis"] = True
    result["tool"] = "get_ambiguous_reviews_sentiment"
    
    return result


# =============================================================================
# FUNCIONES DE REGISTRO
# =============================================================================

def get_nlp_tools() -> List:
    """Retorna lista de tools NLP para bind_tools."""
    return [
        # Tools para datasets completos (uso principal)
        get_sentiment_distribution,
        get_aspect_distribution,
        get_ambiguous_reviews_sentiment,
        # Tools para analisis individual (uso secundario)
        analyze_sentiment,
        analyze_review_complete,
        # Diagnostico
        get_nlp_models_status,
    ]
