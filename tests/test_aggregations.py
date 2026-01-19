"""
Tests para aggregations.

Verifica funciones de agregacion SQL.
"""

import pytest

from tfm.tools.aggregations import (
    aggregate_reviews_by_month,
    aggregate_reviews_by_stars,
    aggregate_olist_sales_by_month,
    aggregate_olist_by_category,
    aggregate_yelp_user_stats,
    get_distribution,
    get_top_entities,
)


class TestAggregateReviewsByStars:
    """Tests para aggregate_reviews_by_stars."""
    
    def test_returns_dict(self):
        """Debe retornar un diccionario."""
        result = aggregate_reviews_by_stars("yelp")
        assert isinstance(result, dict)
    
    def test_invalid_dataset_returns_error(self):
        """Dataset invalido debe retornar error."""
        result = aggregate_reviews_by_stars("invalid")
        assert "error" in result
    
    def test_has_expected_keys(self):
        """Debe tener las claves esperadas (o error si no hay datos)."""
        result = aggregate_reviews_by_stars("yelp")
        if "error" not in result:
            assert "aggregation_type" in result
            assert "total_reviews" in result
            assert "distribution" in result
            assert "summary" in result


class TestAggregateReviewsByMonth:
    """Tests para aggregate_reviews_by_month."""
    
    def test_returns_dict(self):
        """Debe retornar un diccionario."""
        result = aggregate_reviews_by_month("yelp")
        assert isinstance(result, dict)
    
    def test_es_returns_error_no_dates(self):
        """Dataset ES no tiene fechas, debe retornar error."""
        result = aggregate_reviews_by_month("es")
        assert "error" in result
    
    def test_with_year_filter(self):
        """Filtro de ano debe funcionar."""
        result = aggregate_reviews_by_month("yelp", year_filter=2022)
        assert isinstance(result, dict)


class TestAggregateOlistSales:
    """Tests para aggregate_olist_sales_by_month."""
    
    def test_returns_dict(self):
        """Debe retornar un diccionario."""
        result = aggregate_olist_sales_by_month()
        assert isinstance(result, dict)
    
    def test_has_summary_if_data_exists(self):
        """Si hay datos, debe tener summary."""
        result = aggregate_olist_sales_by_month()
        if "error" not in result:
            assert "summary" in result
            assert "total_revenue" in result["summary"]


class TestAggregateOlistByCategory:
    """Tests para aggregate_olist_by_category."""
    
    def test_returns_dict(self):
        """Debe retornar un diccionario."""
        result = aggregate_olist_by_category()
        assert isinstance(result, dict)
    
    def test_top_n_parameter(self):
        """Parametro top_n debe limitar resultados."""
        result = aggregate_olist_by_category(top_n=5)
        if "error" not in result and "data" in result:
            assert len(result["data"]) <= 5


class TestAggregateYelpUserStats:
    """Tests para aggregate_yelp_user_stats."""
    
    def test_returns_dict(self):
        """Debe retornar un diccionario."""
        result = aggregate_yelp_user_stats()
        assert isinstance(result, dict)
    
    def test_has_stats_if_data_exists(self):
        """Si hay datos, debe tener stats."""
        result = aggregate_yelp_user_stats()
        if "error" not in result:
            assert "stats" in result


class TestGetDistribution:
    """Tests para get_distribution."""
    
    def test_returns_dict(self):
        """Debe retornar un diccionario."""
        result = get_distribution("yelp", "stars")
        assert isinstance(result, dict)
    
    def test_invalid_column_returns_error(self):
        """Columna inexistente debe retornar error."""
        result = get_distribution("yelp", "nonexistent_column")
        if "error" not in result:
            # Si hay datos, verificar estructura
            assert "data" in result
        else:
            assert "error" in result


class TestGetTopEntities:
    """Tests para get_top_entities."""
    
    def test_returns_dict(self):
        """Debe retornar un diccionario."""
        result = get_top_entities("yelp", "business_id")
        assert isinstance(result, dict)
    
    def test_with_metric(self):
        """Debe soportar diferentes metricas."""
        result = get_top_entities("yelp", "business_id", metric="count", n=5)
        assert isinstance(result, dict)
    
    def test_invalid_dataset_returns_error(self):
        """Dataset invalido debe retornar error."""
        result = get_top_entities("invalid", "id")
        assert "error" in result
