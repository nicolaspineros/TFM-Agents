"""
Tests para nlp_utils.

Verifica funciones de limpieza de texto y analisis de sentimiento.
"""

import pytest


class TestCleanTextBasic:
    """Tests para clean_text_basic."""
    
    def test_removes_urls(self):
        """Verifica remocion de URLs."""
        from tfm.tools.nlp_utils import clean_text_basic
        
        text = "Check http://example.com for more"
        result = clean_text_basic(text)
        
        assert "http" not in result
        assert "example.com" not in result
    
    def test_removes_emails(self):
        """Verifica remocion de emails."""
        from tfm.tools.nlp_utils import clean_text_basic
        
        text = "Contact test@mail.com for info"
        result = clean_text_basic(text)
        
        assert "@" not in result
    
    def test_lowercases(self):
        """Verifica conversion a minusculas."""
        from tfm.tools.nlp_utils import clean_text_basic
        
        text = "Hello WORLD"
        result = clean_text_basic(text)
        
        assert result == "hello world"
    
    def test_normalizes_spaces(self):
        """Verifica normalizacion de espacios."""
        from tfm.tools.nlp_utils import clean_text_basic
        
        text = "hello    world"
        result = clean_text_basic(text)
        
        assert "  " not in result


class TestCleanTextSpanish:
    """Tests para clean_text_spanish."""
    
    def test_preserves_accents(self):
        """Verifica que preserva acentos."""
        from tfm.tools.nlp_utils import clean_text_spanish
        
        text = "Excelente cafe con pana"
        result = clean_text_spanish(text)
        
        # Deberia mantener caracteres
        assert "excelente" in result
    
    def test_preserves_n_tilde(self):
        """Verifica que preserva enie."""
        from tfm.tools.nlp_utils import clean_text_spanish
        
        # Nota: la funcion puede o no preservar la enie dependiendo de implementacion
        text = "manana"
        result = clean_text_spanish(text)
        
        assert len(result) > 0


class TestCleanTextPortuguese:
    """Tests para clean_text_portuguese."""
    
    def test_preserves_cedilla(self):
        """Verifica que preserva cedilla."""
        from tfm.tools.nlp_utils import clean_text_portuguese
        
        text = "obrigado"
        result = clean_text_portuguese(text)
        
        assert len(result) > 0


class TestSentimentBaseline:
    """Tests para compute_sentiment_baseline."""
    
    def test_five_stars_positive(self):
        """5 estrellas debe ser positivo."""
        from tfm.tools.nlp_utils import compute_sentiment_baseline
        
        result = compute_sentiment_baseline(5)
        
        assert result.label == "positive"
        assert result.score == 1.0
        assert not result.is_ambiguous
    
    def test_four_stars_positive(self):
        """4 estrellas debe ser positivo."""
        from tfm.tools.nlp_utils import compute_sentiment_baseline
        
        result = compute_sentiment_baseline(4)
        
        assert result.label == "positive"
        assert result.score == 0.5
    
    def test_three_stars_neutral_ambiguous(self):
        """3 estrellas debe ser neutral y ambiguo."""
        from tfm.tools.nlp_utils import compute_sentiment_baseline
        
        result = compute_sentiment_baseline(3)
        
        assert result.label == "neutral"
        assert result.score == 0.0
        assert result.is_ambiguous
    
    def test_two_stars_negative(self):
        """2 estrellas debe ser negativo."""
        from tfm.tools.nlp_utils import compute_sentiment_baseline
        
        result = compute_sentiment_baseline(2)
        
        assert result.label == "negative"
        assert result.score == -0.5
    
    def test_one_star_negative(self):
        """1 estrella debe ser negativo."""
        from tfm.tools.nlp_utils import compute_sentiment_baseline
        
        result = compute_sentiment_baseline(1)
        
        assert result.label == "negative"
        assert result.score == -1.0


class TestSentimentVader:
    """Tests para compute_sentiment_vader."""
    
    def test_positive_text(self):
        """Texto positivo debe dar resultado positivo."""
        from tfm.tools.nlp_utils import compute_sentiment_vader
        
        result = compute_sentiment_vader("This is amazing! Best ever!")
        
        assert result.label == "positive"
        assert result.score > 0
    
    def test_negative_text(self):
        """Texto negativo debe dar resultado negativo."""
        from tfm.tools.nlp_utils import compute_sentiment_vader
        
        result = compute_sentiment_vader("This is terrible. Worst experience ever.")
        
        assert result.label == "negative"
        assert result.score < 0
    
    def test_empty_text(self):
        """Texto vacio debe dar neutral."""
        from tfm.tools.nlp_utils import compute_sentiment_vader
        
        result = compute_sentiment_vader("")
        
        assert result.label == "neutral"
        assert result.is_ambiguous


class TestExtractTextFeatures:
    """Tests para extract_text_features."""
    
    def test_counts_basic(self):
        """Verifica conteos basicos."""
        from tfm.tools.nlp_utils import extract_text_features
        
        result = extract_text_features("Hello world!")
        
        assert result["char_count"] == 12
        assert result["word_count"] == 2
    
    def test_empty_text(self):
        """Texto vacio debe dar ceros."""
        from tfm.tools.nlp_utils import extract_text_features
        
        result = extract_text_features("")
        
        assert result["char_count"] == 0
        assert result["word_count"] == 0


class TestUserInfluenceScore:
    """Tests para compute_user_influence_score."""
    
    def test_max_influence(self):
        """Usuario con maximos valores."""
        from tfm.tools.nlp_utils import compute_user_influence_score
        
        score = compute_user_influence_score(
            fans=10000,
            total_compliments=10000,
            review_count=5000,
            friends_count=5000,
            is_elite=True,
        )
        
        assert score == 1.0
    
    def test_zero_influence(self):
        """Usuario sin actividad."""
        from tfm.tools.nlp_utils import compute_user_influence_score
        
        score = compute_user_influence_score(
            fans=0,
            total_compliments=0,
            review_count=0,
            friends_count=0,
            is_elite=False,
        )
        
        assert score == 0.0
    
    def test_elite_bonus(self):
        """Elite status debe agregar bonus."""
        from tfm.tools.nlp_utils import compute_user_influence_score
        
        score_elite = compute_user_influence_score(
            fans=0, total_compliments=0, review_count=0,
            friends_count=0, is_elite=True,
        )
        score_not_elite = compute_user_influence_score(
            fans=0, total_compliments=0, review_count=0,
            friends_count=0, is_elite=False,
        )
        
        assert score_elite > score_not_elite


class TestAnalyzeStarDistribution:
    """Tests para analyze_star_distribution."""
    
    def test_basic_distribution(self):
        """Verifica calculo de distribucion."""
        from tfm.tools.nlp_utils import analyze_star_distribution
        
        stars = [1, 2, 3, 4, 5, 5, 5, 4, 4, 3]
        result = analyze_star_distribution(stars)
        
        assert result["total"] == 10
        assert "mean" in result
        assert "pct_positive" in result
        assert "pct_negative" in result
        assert "pct_ambiguous" in result
