"""
Analisis de sentimiento.

Este modulo implementa:
- Baseline rule-based usando stars
- Integracion con VADER para ingles
- Manejo especial de reviews ambiguas (stars == 3)

El sentimiento se calcula como:
- sentiment_score: float de -1 (muy negativo) a 1 (muy positivo)
- sentiment_label: "positive", "negative", "neutral"
"""

from typing import Optional, Any, Literal, List, Dict

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import polars as pl

from tfm.schemas.outputs import SentimentResult

# Singleton para VADER analyzer
_vader_analyzer = None


def get_vader_analyzer() -> SentimentIntensityAnalyzer:
    """Retorna instancia singleton de VADER."""
    global _vader_analyzer
    if _vader_analyzer is None:
        _vader_analyzer = SentimentIntensityAnalyzer()
    return _vader_analyzer


def compute_sentiment_baseline(
    text: str,
    stars: int,
    language: str = "en"
) -> SentimentResult:
    """
    Calcula sentimiento usando baseline rule-based.
    
    Este es el modelo inicial que será reemplazado por modelos
    más sofisticados en fases posteriores.
    
    Reglas baseline:
    - stars 1-2: negative (score = -0.5 a -1.0)
    - stars 3: neutral (score = 0.0), marcado como ambiguous
    - stars 4-5: positive (score = 0.5 a 1.0)
    
    Args:
        text: Texto de la review (no usado en baseline, solo para firma)
        stars: Rating 1-5
        language: Idioma (no usado en baseline)
        
    Returns:
        SentimentResult con score, label y flags
        
    Example:
        >>> result = compute_sentiment_baseline("Great food!", stars=5)
        >>> print(result.sentiment_label)
        'positive'
        >>> result = compute_sentiment_baseline("Ok...", stars=3)
        >>> print(result.is_ambiguous)
        True
    """
    # Mapeo stars → sentiment
    if stars <= 2:
        score = -0.5 if stars == 2 else -1.0
        label = "negative"
    elif stars == 3:
        score = 0.0
        label = "neutral"
    else:  # 4-5
        score = 0.5 if stars == 4 else 1.0
        label = "positive"
    
    return SentimentResult(
        review_id="",  # Se asigna externamente
        sentiment_score=score,
        sentiment_label=label,
        confidence=1.0,  # Baseline siempre es "seguro" dado el rating
        is_ambiguous=(stars == 3),
        model_version="baseline_stars_v1"
    )


def compute_sentiment_vader(
    text: str,
    review_id: str = ""
) -> SentimentResult:
    """
    Calcula sentimiento usando VADER (ingles).
    
    VADER es un modelo rule-based optimizado para social media
    y reviews. Funciona bien para ingles.
    
    Args:
        text: Texto de la review
        review_id: ID opcional
        
    Returns:
        SentimentResult
    """
    if not text or not text.strip():
        return SentimentResult(
            review_id=review_id,
            sentiment_score=0.0,
            sentiment_label="neutral",
            confidence=0.0,
            is_ambiguous=True,
            model_version="vader_v1"
        )
    
    analyzer = get_vader_analyzer()
    scores = analyzer.polarity_scores(text)
    
    compound = scores["compound"]
    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"
    
    return SentimentResult(
        review_id=review_id,
        sentiment_score=round(compound, 4),
        sentiment_label=label,
        confidence=round(abs(compound), 4),
        is_ambiguous=(label == "neutral"),
        model_version="vader_v1"
    )


def compute_sentiment_spanish(
    text: str,
    review_id: str = ""
) -> SentimentResult:
    """
    Calcula sentimiento para texto en español.
    
    Opciones de implementación:
    - pysentimiento (HuggingFace)
    - BETO fine-tuned
    - Léxico de sentimiento español
    
    Args:
        text: Texto en español
        review_id: ID opcional
        
    Returns:
        SentimentResult
        
    TODO: Fase 5 - Implementar modelo español
    """
    # TODO: Fase 5 - Implementar con pysentimiento o similar
    # from pysentimiento import create_analyzer
    # analyzer = create_analyzer(task="sentiment", lang="es")
    # result = analyzer.predict(text)
    # ...
    
    raise NotImplementedError(
        "Implementar sentimiento español en Fase 5. "
        "Opciones: pysentimiento, BETO, léxico"
    )


def compute_sentiment_batch(
    texts: list[str],
    stars: list[int],
    review_ids: list[str],
    language: str = "en",
    model: Literal["baseline", "vader", "transformer"] = "baseline"
) -> list[SentimentResult]:
    """
    Calcula sentimiento para un batch de reviews.
    
    Optimizado para procesamiento masivo.
    
    Args:
        texts: Lista de textos
        stars: Lista de ratings
        review_ids: Lista de IDs
        language: Idioma de los textos
        model: Modelo a usar
        
    Returns:
        Lista de SentimentResult
        
    Example:
        >>> results = compute_sentiment_batch(
        ...     texts=["Great!", "Terrible", "Ok"],
        ...     stars=[5, 1, 3],
        ...     review_ids=["r1", "r2", "r3"]
        ... )
        
    TODO: Fase 2 - Implementar batch processing
    """
    if len(texts) != len(stars) != len(review_ids):
        raise ValueError("Todas las listas deben tener el mismo tamaño")
    
    results = []
    
    for text, star, rid in zip(texts, stars, review_ids):
        if model == "baseline":
            result = compute_sentiment_baseline(text, star, language)
            result.review_id = rid
        elif model == "vader":
            result = compute_sentiment_vader(text, rid)
        else:
            raise NotImplementedError(f"Modelo {model} no implementado")
        
        results.append(result)
    
    return results


def classify_ambiguous_reviews(
    texts: List[str],
    review_ids: List[str],
    language: str = "en"
) -> List[Dict[str, Any]]:
    """
    Clasifica reviews ambiguas (stars == 3) en positivas o negativas.
    
    Las reviews con stars == 3 son "neutrales" por rating pero
    el texto puede revelar sentimiento real.
    
    Args:
        texts: Textos de reviews ambiguas
        review_ids: IDs correspondientes
        language: Idioma
        
    Returns:
        Lista de dicts con {review_id, inferred_sentiment, confidence}
    """
    if len(texts) != len(review_ids):
        raise ValueError("texts y review_ids deben tener el mismo tamano")
    
    results = []
    
    for text, rid in zip(texts, review_ids):
        if language == "en":
            # Usar VADER para ingles
            sentiment = compute_sentiment_vader(text, rid)
        else:
            # Para otros idiomas, usar heuristicas basicas
            sentiment = _classify_ambiguous_heuristic(text, language)
            sentiment.review_id = rid
        
        results.append({
            "review_id": rid,
            "original_label": "neutral",
            "inferred_label": sentiment.sentiment_label,
            "inferred_score": sentiment.sentiment_score,
            "confidence": sentiment.confidence,
            "model": sentiment.model_version,
        })
    
    return results


def _classify_ambiguous_heuristic(text: str, language: str) -> SentimentResult:
    """
    Clasificacion heuristica para idiomas sin VADER.
    
    Busca palabras clave positivas/negativas.
    """
    text_lower = text.lower() if text else ""
    
    # Palabras clave por idioma
    if language == "es":
        positive_words = ["bueno", "excelente", "recomiendo", "genial", "perfecto", "encanta"]
        negative_words = ["malo", "terrible", "horrible", "pesimo", "no recomiendo", "decepcion"]
    elif language == "pt":
        positive_words = ["bom", "excelente", "recomendo", "otimo", "perfeito", "adorei"]
        negative_words = ["ruim", "terrivel", "horrivel", "pessimo", "nao recomendo", "decepcionado"]
    else:
        positive_words = ["good", "great", "excellent", "recommend", "love", "amazing"]
        negative_words = ["bad", "terrible", "horrible", "awful", "hate", "worst"]
    
    pos_count = sum(1 for w in positive_words if w in text_lower)
    neg_count = sum(1 for w in negative_words if w in text_lower)
    
    if pos_count > neg_count:
        score = min(0.5 + (pos_count * 0.1), 1.0)
        label = "positive"
    elif neg_count > pos_count:
        score = max(-0.5 - (neg_count * 0.1), -1.0)
        label = "negative"
    else:
        score = 0.0
        label = "neutral"
    
    confidence = abs(pos_count - neg_count) / max(pos_count + neg_count, 1)
    
    return SentimentResult(
        review_id="",
        sentiment_score=round(score, 4),
        sentiment_label=label,
        confidence=round(confidence, 4),
        is_ambiguous=(label == "neutral"),
        model_version=f"heuristic_{language}_v1"
    )


def compute_sentiment_combined(
    text: str,
    stars: int,
    weight_text: float = 0.6,
    language: str = "en"
) -> SentimentResult:
    """
    Combina sentimiento de texto y rating.
    
    Util para detectar discrepancias entre lo que dice el usuario
    en el texto y el rating que asigna.
    
    Args:
        text: Texto de la review
        stars: Rating 1-5
        weight_text: Peso del sentimiento de texto (0-1)
        language: Idioma
        
    Returns:
        SentimentResult combinado
    """
    # Sentimiento baseline (por stars)
    baseline = compute_sentiment_baseline(text, stars, language)
    
    # Sentimiento por texto (solo para ingles por ahora)
    if language == "en" and text and text.strip():
        text_sentiment = compute_sentiment_vader(text)
    else:
        text_sentiment = baseline
    
    # Combinar scores
    weight_baseline = 1 - weight_text
    combined_score = (
        text_sentiment.sentiment_score * weight_text +
        baseline.sentiment_score * weight_baseline
    )
    
    # Determinar label
    if combined_score >= 0.3:
        label = "positive"
    elif combined_score <= -0.3:
        label = "negative"
    else:
        label = "neutral"
    
    # Detectar discrepancia
    is_discrepant = (
        baseline.sentiment_label != text_sentiment.sentiment_label and
        baseline.sentiment_label != "neutral" and
        text_sentiment.sentiment_label != "neutral"
    )
    
    return SentimentResult(
        review_id="",
        sentiment_score=round(combined_score, 4),
        sentiment_label=label,
        confidence=round(min(baseline.confidence, text_sentiment.confidence), 4),
        is_ambiguous=(label == "neutral" or is_discrepant),
        model_version="combined_v1"
    )


# =============================================================================
# EXTRACCION DE CARACTERISTICAS DE TEXTO
# =============================================================================

import re

def extract_text_features(text: str) -> Dict[str, Any]:
    """
    Extrae caracteristicas basicas del texto.
    
    Util para analisis exploratorio y como features para ML.
    
    Args:
        text: Texto a analizar
        
    Returns:
        Dict con caracteristicas: char_count, word_count, sentence_count,
        avg_word_length, exclamation_count, question_count, uppercase_ratio
    """
    if not text:
        return {
            'char_count': 0,
            'word_count': 0,
            'sentence_count': 0,
            'avg_word_length': 0,
            'exclamation_count': 0,
            'question_count': 0,
            'uppercase_ratio': 0,
        }
    
    # Conteos basicos
    char_count = len(text)
    words = text.split()
    word_count = len(words)
    
    # Contar oraciones (aproximado)
    sentences = re.split(r'[.!?]+', text)
    sentence_count = len([s for s in sentences if s.strip()])
    
    # Longitud promedio de palabras
    avg_word_length = sum(len(w) for w in words) / max(word_count, 1)
    
    # Signos de puntuacion expresivos
    exclamation_count = text.count('!')
    question_count = text.count('?')
    
    # Ratio de mayusculas (puede indicar enfasis/emocion)
    uppercase_chars = sum(1 for c in text if c.isupper())
    uppercase_ratio = uppercase_chars / max(char_count, 1)
    
    return {
        'char_count': char_count,
        'word_count': word_count,
        'sentence_count': sentence_count,
        'avg_word_length': round(avg_word_length, 2),
        'exclamation_count': exclamation_count,
        'question_count': question_count,
        'uppercase_ratio': round(uppercase_ratio, 4),
    }


# =============================================================================
# SCORE DE INFLUENCIA DE USUARIO (para Yelp)
# =============================================================================

def compute_user_influence_score(
    fans: int,
    total_compliments: int,
    review_count: int,
    friends_count: int,
    is_elite: bool,
    max_fans: int = 10000,
    max_compliments: int = 10000,
    max_reviews: int = 5000,
    max_friends: int = 5000,
) -> float:
    """
    Calcula score de influencia de usuario (0-1).
    
    Pesos:
    - 30% fans
    - 20% compliments
    - 20% review_count
    - 10% friends
    - 20% elite status
    
    Args:
        fans: Numero de fans
        total_compliments: Total de compliments recibidos
        review_count: Numero de reviews escritas
        friends_count: Numero de amigos
        is_elite: Si es usuario Elite
        max_*: Valores maximos para normalizacion
        
    Returns:
        Score de influencia entre 0 y 1
    """
    # Normalizar cada componente
    norm_fans = min(fans / max(max_fans, 1), 1.0)
    norm_compliments = min(total_compliments / max(max_compliments, 1), 1.0)
    norm_reviews = min(review_count / max(max_reviews, 1), 1.0)
    norm_friends = min(friends_count / max(max_friends, 1), 1.0)
    elite_bonus = 1.0 if is_elite else 0.0
    
    # Calcular score ponderado
    score = (
        norm_fans * 0.30 +
        norm_compliments * 0.20 +
        norm_reviews * 0.20 +
        norm_friends * 0.10 +
        elite_bonus * 0.20
    )
    
    return round(min(score, 1.0), 4)