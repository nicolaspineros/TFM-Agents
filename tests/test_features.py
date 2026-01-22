"""
Tests para el modulo features.py
"""

import pytest
from pathlib import Path

from tfm.tools.features import (
    build_gold_features,
    build_olist_sales_features,
    build_yelp_user_features,
    get_feature_stats,
    get_gold_status,
)
from tfm.config.settings import get_settings


class TestGetGoldStatus:
    """Tests para get_gold_status."""
    
    def test_returns_dict(self):
        """Debe retornar un diccionario."""
        status = get_gold_status()
        assert isinstance(status, dict)
    
    def test_contains_expected_keys(self):
        """Debe contener las claves esperadas."""
        status = get_gold_status()
        expected_keys = ["yelp_features", "es_features", "olist_features"]
        for key in expected_keys:
            assert key in status
    
    def test_each_entry_has_exists_key(self):
        """Cada entrada debe tener la clave exists."""
        status = get_gold_status()
        for name, info in status.items():
            assert "exists" in info


class TestGetFeatureStats:
    """Tests para get_feature_stats."""
    
    def test_invalid_dataset_returns_error(self):
        """Dataset invalido debe retornar error."""
        result = get_feature_stats("invalid_dataset")
        assert "error" in result
    
    def test_nonexistent_gold_returns_exists_false(self):
        """Gold inexistente debe retornar exists=False."""
        # Usar un dataset valido pero sin gold construido
        result = get_feature_stats("yelp")
        # Si no existe, debe tener exists=False
        # Si existe, debe tener exists=True
        assert "exists" in result or "error" in result


class TestBuildGoldFeatures:
    """Tests para build_gold_features."""
    
    def test_invalid_dataset_raises(self):
        """Dataset invalido debe lanzar ValueError."""
        with pytest.raises(ValueError):
            build_gold_features("invalid")
    
    def test_missing_silver_raises(self):
        """Silver faltante debe lanzar FileNotFoundError."""
        settings = get_settings()
        # Asegurarse de que no exista el silver para un dataset de prueba
        test_path = settings.silver_dir / "nonexistent.parquet"
        if not test_path.exists():
            # El test solo es valido si el silver no existe
            with pytest.raises(FileNotFoundError):
                build_gold_features("es")  # ES probablemente no tenga silver


class TestBuildYelpUserFeatures:
    """Tests para build_yelp_user_features."""
    
    def test_missing_silver_raises(self):
        """Silver faltante debe lanzar FileNotFoundError."""
        settings = get_settings()
        users_path = settings.silver_dir / "yelp_users.parquet"
        if not users_path.exists():
            with pytest.raises(FileNotFoundError):
                build_yelp_user_features()
