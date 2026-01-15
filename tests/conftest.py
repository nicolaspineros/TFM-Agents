"""
Configuración de pytest para tests TFM.

Define fixtures compartidos y configuración global.
"""

import pytest
import sys
from pathlib import Path


# Añadir src al path para imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


@pytest.fixture
def sample_review():
    """Review de ejemplo para tests."""
    return {
        "review_id": "test_r1",
        "business_id": "test_b1",
        "stars": 4,
        "text": "Great food and excellent service!",
        "date": "2022-06-15",
    }


@pytest.fixture
def sample_reviews_batch():
    """Batch de reviews para tests."""
    return [
        {"review_id": "r1", "stars": 5, "text": "Amazing!"},
        {"review_id": "r2", "stars": 3, "text": "It was okay"},
        {"review_id": "r3", "stars": 1, "text": "Terrible experience"},
        {"review_id": "r4", "stars": 4, "text": "Pretty good"},
        {"review_id": "r5", "stars": 2, "text": "Not recommended"},
    ]


@pytest.fixture
def ambiguous_reviews():
    """Reviews ambiguas (stars == 3) para tests."""
    return [
        {"review_id": "a1", "stars": 3, "text": "It was fine, nothing special"},
        {"review_id": "a2", "stars": 3, "text": "Average experience overall"},
        {"review_id": "a3", "stars": 3, "text": "Could be better, could be worse"},
    ]


@pytest.fixture
def temp_data_dir(tmp_path):
    """
    Directorio temporal para tests que necesitan escribir archivos.
    
    Crea estructura bronze/silver/gold.
    """
    (tmp_path / "bronze").mkdir()
    (tmp_path / "silver").mkdir()
    (tmp_path / "gold").mkdir()
    return tmp_path


@pytest.fixture
def mock_settings(monkeypatch, temp_data_dir):
    """
    Settings mockeados para tests.
    
    Apunta data_dir a directorio temporal.
    """
    from tfm.config.settings import Settings
    
    # Crear settings con paths temporales
    test_settings = Settings(
        data_dir=temp_data_dir,
        warehouse_path=temp_data_dir / "test.duckdb",
        openai_api_key="test_key",
    )
    
    # Mockear get_settings
    def mock_get_settings():
        return test_settings
    
    monkeypatch.setattr("tfm.config.settings.get_settings", mock_get_settings)
    
    return test_settings
