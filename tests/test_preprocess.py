"""
Tests para preprocess.

Verifica funciones de limpieza y normalización.
"""

import pytest


class TestCleanText:
    """Tests para clean_text."""
    
    def test_clean_text_strips_whitespace(self):
        """Verifica que strip whitespace."""
        from tfm.tools.preprocess import clean_text
        
        result = clean_text("  hello world  ")
        assert result == "hello world"
    
    def test_clean_text_normalizes_spaces(self):
        """Verifica que normaliza espacios múltiples."""
        from tfm.tools.preprocess import clean_text
        
        result = clean_text("hello    world")
        assert result == "hello world"
    
    def test_clean_text_handles_empty(self):
        """Verifica manejo de string vacío."""
        from tfm.tools.preprocess import clean_text
        
        result = clean_text("")
        assert result == ""
    
    def test_clean_text_handles_none(self):
        """Verifica manejo de None."""
        from tfm.tools.preprocess import clean_text
        
        result = clean_text(None)
        assert result == ""
    
    def test_clean_text_removes_control_chars(self):
        """Verifica remoción de caracteres de control."""
        from tfm.tools.preprocess import clean_text
        
        result = clean_text("hello\x00world")
        assert "\x00" not in result


class TestComputeTextStats:
    """Tests para compute_text_stats."""
    
    def test_text_stats_basic(self):
        """Verifica cálculo básico de stats."""
        from tfm.tools.preprocess import compute_text_stats
        
        result = compute_text_stats("hello world")
        
        assert result["text_length"] == 11
        assert result["word_count"] == 2
    
    def test_text_stats_empty(self):
        """Verifica stats de string vacío."""
        from tfm.tools.preprocess import compute_text_stats
        
        result = compute_text_stats("")
        
        assert result["text_length"] == 0
        assert result["word_count"] == 0
    
    def test_text_stats_multiword(self):
        """Verifica conteo de múltiples palabras."""
        from tfm.tools.preprocess import compute_text_stats
        
        result = compute_text_stats("The quick brown fox jumps")
        
        assert result["word_count"] == 5


class TestBuildSilver:
    """Tests para build_silver_* functions."""
    
    def test_build_silver_yelp_creates_file(self):
        """Verifica que build_silver_yelp crea archivo."""
        from tfm.tools.preprocess import build_silver_yelp
        from tfm.tools.io_loaders import get_bronze_file_info
        from tfm.config.settings import get_settings, SILVER_FILES
        
        # Verificar que bronze existe
        info = get_bronze_file_info("yelp", "reviews")
        if not info["exists"]:
            pytest.skip("Archivo Yelp bronze no existe")
        
        # Construir con limite pequeno
        path = build_silver_yelp(limit=100, overwrite=True)
        
        assert path.exists()
        assert path.suffix == ".parquet"
    
    def test_build_silver_es_creates_file(self):
        """Verifica que build_silver_es crea archivo."""
        from tfm.tools.preprocess import build_silver_es
        from tfm.tools.io_loaders import get_bronze_file_info
        
        info = get_bronze_file_info("es", "reviews")
        if not info["exists"]:
            pytest.skip("Archivo ES bronze no existe")
        
        path = build_silver_es(limit=100, overwrite=True)
        
        assert path.exists()
        assert path.suffix == ".parquet"
    
    def test_build_silver_olist_creates_files(self):
        """Verifica que build_silver_olist crea archivos."""
        from tfm.tools.preprocess import build_silver_olist
        from tfm.tools.io_loaders import get_bronze_file_info
        
        info = get_bronze_file_info("olist", "reviews")
        if not info["exists"]:
            pytest.skip("Archivo Olist bronze no existe")
        
        orders_path, reviews_path = build_silver_olist(overwrite=True)
        
        assert orders_path.exists()
        assert reviews_path.exists()


class TestCheckSilverStatus:
    """Tests para check_silver_status."""
    
    def test_check_silver_status_returns_dict(self):
        """Verifica que retorna diccionario."""
        from tfm.tools.preprocess import check_silver_status
        
        status = check_silver_status()
        
        assert isinstance(status, dict)
        for key, value in status.items():
            assert "exists" in value
            assert "path" in value
