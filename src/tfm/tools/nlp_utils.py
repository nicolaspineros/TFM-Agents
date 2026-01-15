"""
Utilidades NLP para el TFM.

Este módulo contiene funciones de preprocesamiento de texto y análisis
de sentimiento que serán usadas en los tools y notebooks.

Fase 1: Prototipos desarrollados durante EDA
Fase 2: Integración con tools
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class SentimentResult:
    """Resultado de análisis de sentimiento."""
    score: float  # -1 a 1
    label: str  # positive, negative, neutral
    is_ambiguous: bool
    model: str
    details: Optional[Dict] = None


# =============================================================================
# LIMPIEZA DE TEXTO
# =============================================================================

def clean_text_basic(text: str) -> str:
    """
    Limpieza básica de texto.
    
    Operaciones:
    - Convertir a minúsculas
    - Eliminar URLs
    - Eliminar emails
    - Normalizar espacios múltiples
    - Strip whitespace
    
    Args:
        text: Texto a limpiar
        
    Returns:
        Texto limpio
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Convertir a minúsculas
    text = text.lower()
    
    # Eliminar URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Eliminar emails
    text = re.sub(r'\S+@\S+', '', text)
    
    # Eliminar menciones de usuario (si hubiera)
    text = re.sub(r'@\w+', '', text)
    
    # Normalizar espacios múltiples y newlines
    text = re.sub(r'\s+', ' ', text)
    
    # Eliminar espacios al inicio y final
    text = text.strip()
    
    return text


def clean_text_for_nlp(text: str) -> str:
    """
    Limpieza más agresiva para análisis NLP.
    
    Adicional a clean_text_basic:
    - Remueve puntuación
    - Expande contracciones comunes (inglés)
    
    Args:
        text: Texto a limpiar
        
    Returns:
        Texto limpio para NLP
    """
    text = clean_text_basic(text)
    
    # Remover puntuación (mantener apóstrofes para contracciones)
    text = re.sub(r"[^a-zA-Z0-9'\s]", '', text)
    
    # Normalizar contracciones comunes (inglés)
    contractions = {
        "n't": " not",
        "'re": " are",
        "'s": " is",
        "'d": " would",
        "'ll": " will",
        "'ve": " have",
        "'m": " am",
    }
    for contraction, expansion in contractions.items():
        text = text.replace(contraction, expansion)
    
    return text.strip()


def clean_text_spanish(text: str) -> str:
    """
    Limpieza de texto para español.
    
    Preserva acentos y caracteres especiales del español.
    
    Args:
        text: Texto en español
        
    Returns:
        Texto limpio
    """
    if not text or not isinstance(text, str):
        return ""
    
    text = text.lower()
    
    # Eliminar URLs y emails
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    
    # Mantener caracteres españoles (ñ, acentos)
    text = re.sub(r'[^a-záéíóúüñ\s]', '', text)
    
    # Normalizar espacios
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def clean_text_portuguese(text: str) -> str:
    """
    Limpieza de texto para portugués.
    
    Preserva acentos y caracteres especiales del portugués.
    
    Args:
        text: Texto en portugués
        
    Returns:
        Texto limpio
    """
    if not text or not isinstance(text, str):
        return ""
    
    text = text.lower()
    
    # Eliminar URLs y emails
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    
    # Mantener caracteres portugueses
    text = re.sub(r'[^a-záàâãéêíóôõúç\s]', '', text)
    
    # Normalizar espacios
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


# =============================================================================
# ANÁLISIS DE SENTIMIENTO
# =============================================================================

def compute_sentiment_baseline(stars: int) -> SentimentResult:
    """
    Baseline: inferir sentimiento desde estrellas.
    
    Mapeo:
    - 5 stars -- score=1.0, positive
    - 4 stars -- score=0.5, positive
    - 3 stars -- score=0.0, neutral (AMBIGUOUS)
    - 2 stars -- score=-0.5, negative
    - 1 star  -- score=-1.0, negative
    
    Args:
        stars: Rating 1-5
        
    Returns:
        SentimentResult con score, label, is_ambiguous
    """
    if stars >= 4:
        return SentimentResult(
            score=(stars - 3) / 2.0,
            label='positive',
            is_ambiguous=False,
            model='baseline'
        )
    elif stars <= 2:
        return SentimentResult(
            score=(stars - 3) / 2.0,
            label='negative',
            is_ambiguous=False,
            model='baseline'
        )
    else:  # stars == 3
        return SentimentResult(
            score=0.0,
            label='neutral',
            is_ambiguous=True,
            model='baseline'
        )


def compute_sentiment_vader(text: str) -> SentimentResult:
    """
    Sentimiento usando VADER (para inglés).
    
    VADER es un modelo rule-based optimizado para redes sociales
    y reviews. Devuelve scores de -1 a 1.
    
    Args:
        text: Texto en inglés
        
    Returns:
        SentimentResult con score VADER
        
    Raises:
        ImportError si VADER no está instalado
    """
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except ImportError:
        raise ImportError(
            "VADER no instalado. Ejecutar: uv add vaderSentiment"
        )
    
    if not text or not isinstance(text, str):
        return SentimentResult(
            score=0.0,
            label='neutral',
            is_ambiguous=True,
            model='vader'
        )
    
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    compound = scores['compound']
    
    # Clasificar según umbrales de VADER
    if compound >= 0.05:
        label = 'positive'
        is_ambiguous = False
    elif compound <= -0.05:
        label = 'negative'
        is_ambiguous = False
    else:
        label = 'neutral'
        is_ambiguous = True
    
    return SentimentResult(
        score=compound,
        label=label,
        is_ambiguous=is_ambiguous,
        model='vader',
        details={
            'neg': scores['neg'],
            'neu': scores['neu'],
            'pos': scores['pos'],
        }
    )


def compute_sentiment_combined(
    text: str, 
    stars: int, 
    weight_text: float = 0.6
) -> SentimentResult:
    """
    Combina sentimiento de texto (VADER) con baseline (stars).
    
    Útil para casos ambiguos donde stars=3 pero el texto
    puede indicar sentimiento más claro.
    
    Args:
        text: Texto de la review
        stars: Rating 1-5
        weight_text: Peso del análisis de texto (0-1)
        
    Returns:
        SentimentResult combinado
    """
    baseline = compute_sentiment_baseline(stars)
    
    try:
        vader = compute_sentiment_vader(text)
        
        # Combinar scores
        combined_score = (
            vader.score * weight_text + 
            baseline.score * (1 - weight_text)
        )
        
        # Determinar label
        if combined_score >= 0.05:
            label = 'positive'
        elif combined_score <= -0.05:
            label = 'negative'
        else:
            label = 'neutral'
        
        # Es ambiguo si ambos métodos difieren o score cercano a 0
        is_ambiguous = (
            abs(combined_score) < 0.2 or 
            vader.label != baseline.label
        )
        
        return SentimentResult(
            score=combined_score,
            label=label,
            is_ambiguous=is_ambiguous,
            model='combined',
            details={
                'vader_score': vader.score,
                'baseline_score': baseline.score,
                'weight_text': weight_text,
            }
        )
    except ImportError:
        # Si VADER no disponible, usar solo baseline
        return baseline


# =============================================================================
# EXTRACCIÓN DE CARACTERÍSTICAS
# =============================================================================

def extract_text_features(text: str) -> Dict:
    """
    Extrae características básicas del texto.
    
    Args:
        text: Texto a analizar
        
    Returns:
        Dict con características
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
    
    # Conteos básicos
    char_count = len(text)
    words = text.split()
    word_count = len(words)
    
    # Contar oraciones (aproximado)
    sentences = re.split(r'[.!?]+', text)
    sentence_count = len([s for s in sentences if s.strip()])
    
    # Longitud promedio de palabras
    avg_word_length = sum(len(w) for w in words) / max(word_count, 1)
    
    # Signos de puntuación expresivos
    exclamation_count = text.count('!')
    question_count = text.count('?')
    
    # Ratio de mayúsculas (puede indicar énfasis/emoción)
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
        fans: Número de fans
        total_compliments: Total de compliments recibidos
        review_count: Número de reviews escritas
        friends_count: Número de amigos
        is_elite: Si es usuario Elite
        max_*: Valores máximos para normalización
        
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


# =============================================================================
# FUNCIONES DE ANÁLISIS PARA EDA
# =============================================================================

def analyze_star_distribution(stars_series) -> Dict:
    """
    Analiza distribución de ratings.
    
    Args:
        stars_series: Serie/lista de ratings (1-5)
        
    Returns:
        Dict con estadísticas de distribución
    """
    import numpy as np
    
    stars = np.array(stars_series)
    total = len(stars)
    
    distribution = {}
    for star in range(1, 6):
        count = np.sum(stars == star)
        distribution[star] = {
            'count': int(count),
            'percentage': round(count / total * 100, 2)
        }
    
    # Calcular agregados
    pct_positive = distribution[4]['percentage'] + distribution[5]['percentage']
    pct_negative = distribution[1]['percentage'] + distribution[2]['percentage']
    pct_ambiguous = distribution[3]['percentage']
    
    return {
        'distribution': distribution,
        'total': total,
        'mean': round(float(np.mean(stars)), 2),
        'std': round(float(np.std(stars)), 2),
        'pct_positive': pct_positive,
        'pct_negative': pct_negative,
        'pct_ambiguous': pct_ambiguous,
        'is_imbalanced': pct_positive > 60 or pct_negative > 60,
    }
