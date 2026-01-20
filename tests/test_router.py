"""
Tests para el Router Agent.
"""

import pytest

from tfm.agents.router import (
    get_artifacts_status_summary,
    apply_guardrails,
    suggest_clarifications,
    _parse_json_response,
)
from tfm.schemas.request import UserQuery


class TestArtifactsStatus:
    """Tests para get_artifacts_status_summary."""
    
    def test_returns_string(self):
        """Debe retornar un string."""
        result = get_artifacts_status_summary()
        assert isinstance(result, str)
    
    def test_contains_silver_section(self):
        """Debe contener seccion SILVER LAYER."""
        result = get_artifacts_status_summary()
        assert "SILVER LAYER" in result
    
    def test_contains_gold_section(self):
        """Debe contener seccion GOLD LAYER."""
        result = get_artifacts_status_summary()
        assert "GOLD LAYER" in result


class TestParseJsonResponse:
    """Tests para _parse_json_response."""
    
    def test_parse_valid_json(self):
        """Debe parsear JSON valido."""
        response = '{"is_valid": true, "datasets_required": ["yelp"]}'
        result = _parse_json_response(response)
        assert result["is_valid"] == True
        assert result["datasets_required"] == ["yelp"]
    
    def test_parse_json_with_markdown(self):
        """Debe parsear JSON envuelto en markdown."""
        response = '```json\n{"is_valid": true}\n```'
        result = _parse_json_response(response)
        assert result["is_valid"] == True
    
    def test_parse_invalid_returns_defaults(self):
        """JSON invalido debe retornar defaults."""
        response = "esto no es json valido"
        result = _parse_json_response(response)
        assert "is_valid" in result
        assert "datasets_required" in result


class TestApplyGuardrails:
    """Tests para apply_guardrails."""
    
    def test_empty_query_invalid(self):
        """Query vacia debe ser invalida."""
        query = UserQuery(text="")
        is_valid, errors = apply_guardrails(query, {"datasets": {}})
        assert is_valid == False
    
    def test_short_query_invalid(self):
        """Query muy corta debe ser invalida."""
        query = UserQuery(text="ab")
        is_valid, errors = apply_guardrails(query, {"datasets": {}})
        assert is_valid == False
    
    def test_temporal_query_on_es_invalid(self):
        """Query temporal en ES debe ser invalida."""
        query = UserQuery(
            text="Cual es la tendencia por mes?",
            preferred_dataset="es"
        )
        is_valid, errors = apply_guardrails(query, {"datasets": {"es": {"exists": True}}})
        assert is_valid == False
        assert "fecha" in errors[0].lower() or "temporal" in errors[0].lower()
    
    def test_sales_query_on_yelp_invalid(self):
        """Query de ventas en Yelp debe ser invalida."""
        query = UserQuery(
            text="Cuales son las ventas por mes?",
            preferred_dataset="yelp"
        )
        is_valid, errors = apply_guardrails(query, {"datasets": {"yelp": {"exists": True}}})
        assert is_valid == False
        assert "olist" in errors[0].lower()
    
    def test_valid_query_passes(self):
        """Query valida debe pasar."""
        query = UserQuery(
            text="Cual es la distribucion de ratings en Yelp?"
        )
        is_valid, warnings = apply_guardrails(query, {"datasets": {"yelp": {"exists": True}}})
        assert is_valid == True


class TestSuggestClarifications:
    """Tests para suggest_clarifications."""
    
    def test_no_dataset_suggests_options(self):
        """Sin dataset debe sugerir opciones."""
        query = UserQuery(text="Cual es el sentimiento?")
        suggestions = suggest_clarifications(query, {"datasets": {}})
        assert len(suggestions) > 0
        assert "yelp" in suggestions[0].lower() or "olist" in suggestions[0].lower()
    
    def test_short_query_suggests_examples(self):
        """Query corta debe sugerir ejemplos."""
        query = UserQuery(text="Hola")
        suggestions = suggest_clarifications(query, {"datasets": {}})
        assert len(suggestions) > 0
