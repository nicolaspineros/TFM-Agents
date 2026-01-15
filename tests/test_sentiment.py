"""
Tests para sentiment analysis.

Verifica:
- Baseline sentiment (stars-based)
- Manejo de reviews ambiguas (stars == 3)
- Batch processing
"""

import pytest


class TestSentimentBaseline:
    """Tests para compute_sentiment_baseline."""
    
    def test_baseline_positive_5_stars(self):
        """5 estrellas debe ser positivo."""
        from tfm.tools.sentiment import compute_sentiment_baseline
        
        result = compute_sentiment_baseline("Great!", stars=5)
        
        assert result.sentiment_label == "positive"
        assert result.sentiment_score == 1.0
        assert not result.is_ambiguous
    
    def test_baseline_positive_4_stars(self):
        """4 estrellas debe ser positivo."""
        from tfm.tools.sentiment import compute_sentiment_baseline
        
        result = compute_sentiment_baseline("Good", stars=4)
        
        assert result.sentiment_label == "positive"
        assert result.sentiment_score == 0.5
        assert not result.is_ambiguous
    
    def test_baseline_neutral_3_stars_is_ambiguous(self):
        """3 estrellas debe ser neutral Y marcado como ambiguo."""
        from tfm.tools.sentiment import compute_sentiment_baseline
        
        result = compute_sentiment_baseline("Ok", stars=3)
        
        assert result.sentiment_label == "neutral"
        assert result.sentiment_score == 0.0
        assert result.is_ambiguous  # IMPORTANTE: caso especial
    
    def test_baseline_negative_2_stars(self):
        """2 estrellas debe ser negativo."""
        from tfm.tools.sentiment import compute_sentiment_baseline
        
        result = compute_sentiment_baseline("Not great", stars=2)
        
        assert result.sentiment_label == "negative"
        assert result.sentiment_score == -0.5
        assert not result.is_ambiguous
    
    def test_baseline_negative_1_star(self):
        """1 estrella debe ser muy negativo."""
        from tfm.tools.sentiment import compute_sentiment_baseline
        
        result = compute_sentiment_baseline("Terrible!", stars=1)
        
        assert result.sentiment_label == "negative"
        assert result.sentiment_score == -1.0
        assert not result.is_ambiguous
    
    def test_baseline_model_version(self):
        """Verifica que model_version sea baseline."""
        from tfm.tools.sentiment import compute_sentiment_baseline
        
        result = compute_sentiment_baseline("Test", stars=3)
        
        assert "baseline" in result.model_version.lower()


class TestSentimentBatch:
    """Tests para compute_sentiment_batch."""
    
    def test_batch_basic(self):
        """Verifica procesamiento batch básico."""
        from tfm.tools.sentiment import compute_sentiment_batch
        
        texts = ["Great!", "Ok", "Terrible"]
        stars = [5, 3, 1]
        ids = ["r1", "r2", "r3"]
        
        results = compute_sentiment_batch(texts, stars, ids)
        
        assert len(results) == 3
        assert results[0].review_id == "r1"
        assert results[0].sentiment_label == "positive"
        assert results[1].is_ambiguous  # stars == 3
        assert results[2].sentiment_label == "negative"
    
    def test_batch_mismatched_lengths_raises(self):
        """Listas de diferente tamaño deben lanzar error."""
        from tfm.tools.sentiment import compute_sentiment_batch
        
        with pytest.raises(ValueError):
            compute_sentiment_batch(
                texts=["a", "b"],
                stars=[1, 2, 3],  # diferente longitud
                review_ids=["r1", "r2"]
            )
    
    def test_batch_all_ambiguous(self):
        """Batch con todas las reviews ambiguas."""
        from tfm.tools.sentiment import compute_sentiment_batch
        
        texts = ["Ok", "Meh", "Average"]
        stars = [3, 3, 3]
        ids = ["r1", "r2", "r3"]
        
        results = compute_sentiment_batch(texts, stars, ids)
        
        assert all(r.is_ambiguous for r in results)
        assert all(r.sentiment_label == "neutral" for r in results)


class TestVaderSentiment:
    """Tests para compute_sentiment_vader."""
    
    def test_vader_not_implemented(self):
        """VADER no implementado en fase temprana."""
        from tfm.tools.sentiment import compute_sentiment_vader
        
        with pytest.raises(NotImplementedError):
            compute_sentiment_vader("Great service!")


class TestSpanishSentiment:
    """Tests para compute_sentiment_spanish."""
    
    def test_spanish_not_implemented(self):
        """Sentimiento español no implementado en fase temprana."""
        from tfm.tools.sentiment import compute_sentiment_spanish
        
        with pytest.raises(NotImplementedError):
            compute_sentiment_spanish("Excelente servicio!")


class TestAmbiguousClassification:
    """Tests para classify_ambiguous_reviews."""
    
    def test_classify_ambiguous_not_implemented(self):
        """Clasificación de ambiguos no implementada en fase temprana."""
        from tfm.tools.sentiment import classify_ambiguous_reviews
        
        with pytest.raises(NotImplementedError):
            classify_ambiguous_reviews(
                texts=["It was ok but had issues"],
                review_ids=["r1"]
            )
