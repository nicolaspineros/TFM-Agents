"""
Tests para io_loaders.

Verifica que los loaders puedan:
- Detectar archivos existentes
- Reportar archivos faltantes correctamente
- Cargar datos correctamente
"""

import pytest
from pathlib import Path


class TestBronzeFileInfo:
    """Tests para get_bronze_file_info."""
    
    def test_yelp_reviews_info(self):
        """Verifica info de archivo Yelp reviews."""
        from tfm.tools.io_loaders import get_bronze_file_info
        
        info = get_bronze_file_info("yelp", "reviews")
        
        assert "path" in info
        assert "exists" in info
        assert "yelp" in info["path"]
        assert "review" in info["path"].lower()
    
    def test_es_reviews_info(self):
        """Verifica info de archivo ES reviews."""
        from tfm.tools.io_loaders import get_bronze_file_info
        
        info = get_bronze_file_info("es", "reviews")
        
        assert "path" in info
        assert "exists" in info
    
    def test_olist_orders_info(self):
        """Verifica info de archivo Olist orders."""
        from tfm.tools.io_loaders import get_bronze_file_info
        
        info = get_bronze_file_info("olist", "orders")
        
        assert "path" in info
        assert "exists" in info
    
    def test_invalid_dataset_raises(self):
        """Dataset invalido debe lanzar error."""
        from tfm.tools.io_loaders import get_bronze_file_info
        
        with pytest.raises(ValueError, match="invalido"):
            get_bronze_file_info("invalid_dataset", "reviews")
    
    def test_invalid_table_raises(self):
        """Tabla invalida debe lanzar error."""
        from tfm.tools.io_loaders import get_bronze_file_info
        
        with pytest.raises(ValueError, match="no existe"):
            get_bronze_file_info("yelp", "invalid_table")


class TestCheckBronzeFiles:
    """Tests para check_bronze_files."""
    
    def test_check_bronze_files_returns_dict(self):
        """Verifica que check_bronze_files retorne dict."""
        from tfm.tools.io_loaders import check_bronze_files
        
        status = check_bronze_files()
        
        assert isinstance(status, dict)
        assert "yelp" in status
        assert "es" in status
        assert "olist" in status


class TestYelpLoader:
    """Tests para load_yelp_reviews."""
    
    def test_load_yelp_reviews_with_limit(self):
        """Verifica carga de reviews con limite."""
        from tfm.tools.io_loaders import load_yelp_reviews, get_bronze_file_info
        
        # Verificar que el archivo exista primero
        info = get_bronze_file_info("yelp", "reviews")
        if not info["exists"]:
            pytest.skip("Archivo Yelp reviews no existe")
        
        df = load_yelp_reviews(limit=100)
        
        assert df.height == 100
        assert "stars" in df.columns
        assert "text" in df.columns
        assert "review_id" in df.columns
    
    def test_load_yelp_users_with_limit(self):
        """Verifica carga de usuarios con limite."""
        from tfm.tools.io_loaders import load_yelp_users, get_bronze_file_info
        
        info = get_bronze_file_info("yelp", "users")
        if not info["exists"]:
            pytest.skip("Archivo Yelp users no existe")
        
        df = load_yelp_users(limit=100)
        
        assert df.height == 100
        assert "user_id" in df.columns


class TestESLoader:
    """Tests para load_es_reviews."""
    
    def test_load_es_reviews(self):
        """Verifica carga de reviews ES."""
        from tfm.tools.io_loaders import load_es_reviews, get_bronze_file_info
        
        info = get_bronze_file_info("es", "reviews")
        if not info["exists"]:
            pytest.skip("Archivo ES reviews no existe")
        
        df = load_es_reviews(limit=100)
        
        assert df.height == 100
        assert "stars" in df.columns


class TestOlistLoader:
    """Tests para load_olist_data."""
    
    def test_load_olist_orders(self):
        """Verifica carga de ordenes Olist."""
        from tfm.tools.io_loaders import load_olist_data, get_bronze_file_info
        
        info = get_bronze_file_info("olist", "orders")
        if not info["exists"]:
            pytest.skip("Archivo Olist orders no existe")
        
        df = load_olist_data("orders", limit=100)
        
        assert df.height == 100
        assert "order_id" in df.columns
    
    def test_load_olist_reviews(self):
        """Verifica carga de reviews Olist."""
        from tfm.tools.io_loaders import load_olist_data, get_bronze_file_info
        
        info = get_bronze_file_info("olist", "reviews")
        if not info["exists"]:
            pytest.skip("Archivo Olist reviews no existe")
        
        df = load_olist_data("reviews", limit=100)
        
        assert df.height == 100
        assert "review_score" in df.columns
    
    def test_load_olist_invalid_table(self):
        """Tabla invalida debe lanzar ValueError."""
        from tfm.tools.io_loaders import load_olist_data
        
        with pytest.raises(ValueError, match="invalida"):
            load_olist_data("invalid_table")
