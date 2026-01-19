"""
Tests para el modulo sentiment.py
"""

import pytest

from tfm.tools.sentiment import (
    compute_sentiment_baseline,
    compute_sentiment_vader,
    compute_sentiment_combined,
    compute_sentiment_batch,
    classify_ambiguous_reviews,
)


class TestComputeSentimentBaseline:
    """Tests para compute_sentiment_baseline."""
    
    def test_five_stars_positive(self):
        """5 estrellas debe ser positivo."""
        result = compute_sentiment_baseline("Great!", 5)
        assert result.sentiment_label == "positive"
        assert result.sentiment_score == 1.0
    
    def test_one_star_negative(self):
        """1 estrella debe ser negativo."""
        result = compute_sentiment_baseline("Bad", 1)
        assert result.sentiment_label == "negative"
        assert result.sentiment_score == -1.0
    
    def test_three_stars_neutral_ambiguous(self):
        """3 estrellas debe ser neutral y ambiguo."""
        result = compute_sentiment_baseline("Ok", 3)
        assert result.sentiment_label == "neutral"
        assert result.is_ambiguous == True
    
    def test_four_stars_positive(self):
        """4 estrellas debe ser positivo."""
        result = compute_sentiment_baseline("Good", 4)
        assert result.sentiment_label == "positive"
        assert result.sentiment_score == 0.5
    
    def test_two_stars_negative(self):
        """2 estrellas debe ser negativo."""
        result = compute_sentiment_baseline("Bad", 2)
        assert result.sentiment_label == "negative"
        assert result.sentiment_score == -0.5


class TestComputeSentimentVader:
    """Tests para compute_sentiment_vader."""
    
    def test_positive_text(self):
        """Texto positivo debe retornar sentimiento positivo."""
        result = compute_sentiment_vader("This is amazing and wonderful!")
        assert result.sentiment_label == "positive"
        assert result.sentiment_score > 0
    
    def test_negative_text(self):
        """Texto negativo debe retornar sentimiento negativo."""
        result = compute_sentiment_vader("This is terrible and awful!")
        assert result.sentiment_label == "negative"
        assert result.sentiment_score < 0
    
    def test_neutral_text(self):
        """Texto neutral debe retornar neutral."""
        result = compute_sentiment_vader("The product is a product.")
        assert result.sentiment_label == "neutral"
    
    def test_empty_text(self):
        """Texto vacio debe retornar neutral ambiguo."""
        result = compute_sentiment_vader("")
        assert result.sentiment_label == "neutral"
        assert result.is_ambiguous == True
    
    def test_model_version(self):
        """Debe incluir version del modelo."""
        result = compute_sentiment_vader("Hello")
        assert result.model_version == "vader_v1"


class TestComputeSentimentCombined:
    """Tests para compute_sentiment_combined."""
    
    def test_consistent_positive(self):
        """Texto y rating consistentemente positivos."""
        result = compute_sentiment_combined("Amazing experience!", 5)
        assert result.sentiment_label == "positive"
    
    def test_consistent_negative(self):
        """Texto y rating consistentemente negativos."""
        result = compute_sentiment_combined("Horrible, terrible!", 1)
        assert result.sentiment_label == "negative"
    
    def test_discrepant_text_positive_rating_low(self):
        """Discrepancia: texto positivo pero rating bajo."""
        result = compute_sentiment_combined("Great food!", 2)
        # El resultado depende de los pesos, pero debe detectar ambiguedad
        assert result.is_ambiguous or result.sentiment_label != "negative"
    
    def test_model_version(self):
        """Debe usar modelo combinado."""
        result = compute_sentiment_combined("Test", 3)
        assert result.model_version == "combined_v1"


class TestComputeSentimentBatch:
    """Tests para compute_sentiment_batch."""
    
    def test_batch_baseline(self):
        """Batch baseline debe procesar multiples reviews."""
        texts = ["Great!", "Bad", "Ok"]
        stars = [5, 1, 3]
        ids = ["r1", "r2", "r3"]
        
        results = compute_sentiment_batch(texts, stars, ids, model="baseline")
        assert len(results) == 3
        assert results[0].sentiment_label == "positive"
        assert results[1].sentiment_label == "negative"
        assert results[2].sentiment_label == "neutral"
    
    def test_batch_preserves_ids(self):
        """Debe preservar los review_ids."""
        texts = ["Test1", "Test2"]
        stars = [5, 1]
        ids = ["id_a", "id_b"]
        
        results = compute_sentiment_batch(texts, stars, ids)
        assert results[0].review_id == "id_a"
        assert results[1].review_id == "id_b"
    
    def test_batch_mismatched_lengths_raises(self):
        """Longitudes diferentes deben lanzar error."""
        with pytest.raises(ValueError):
            compute_sentiment_batch(
                texts=["a", "b"],
                stars=[1],
                review_ids=["r1", "r2"]
            )


class TestClassifyAmbiguousReviews:
    """Tests para classify_ambiguous_reviews."""
    
    def test_classifies_english(self):
        """Debe clasificar reviews en ingles."""
        texts = ["This was actually pretty good!", "Horrible experience"]
        ids = ["r1", "r2"]
        
        results = classify_ambiguous_reviews(texts, ids, language="en")
        assert len(results) == 2
        assert results[0]["inferred_label"] == "positive"
        assert results[1]["inferred_label"] == "negative"
    
    def test_returns_expected_keys(self):
        """Debe retornar las claves esperadas."""
        results = classify_ambiguous_reviews(["Test"], ["r1"])
        assert "review_id" in results[0]
        assert "inferred_label" in results[0]
        assert "inferred_score" in results[0]
        assert "confidence" in results[0]
    
    def test_mismatched_lengths_raises(self):
        """Longitudes diferentes deben lanzar error."""
        with pytest.raises(ValueError):
            classify_ambiguous_reviews(
                texts=["a", "b"],
                review_ids=["r1"]
            )
