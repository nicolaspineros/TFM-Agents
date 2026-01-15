"""
Tests para aggregations.

Verifica funciones de agregación SQL.
"""

import pytest


class TestAggregationsByMonth:
    """Tests para aggregate_sentiment_by_month."""
    
    def test_aggregate_by_month_not_implemented(self):
        """Agregación no implementada en fase temprana."""
        from tfm.tools.aggregations import aggregate_sentiment_by_month
        
        with pytest.raises(NotImplementedError):
            aggregate_sentiment_by_month("yelp")


class TestAggregationsByCategory:
    """Tests para aggregate_sentiment_by_category."""
    
    def test_aggregate_by_category_not_implemented(self):
        """Agregación no implementada en fase temprana."""
        from tfm.tools.aggregations import aggregate_sentiment_by_category
        
        with pytest.raises(NotImplementedError):
            aggregate_sentiment_by_category("yelp")


class TestCustomAggregation:
    """Tests para run_custom_aggregation."""
    
    def test_custom_rejects_non_select(self):
        """Custom aggregation debe rechazar queries no-SELECT."""
        from tfm.tools.aggregations import run_custom_aggregation
        
        with pytest.raises(ValueError, match="SELECT"):
            run_custom_aggregation(
                "DELETE FROM reviews",
                dataset="yelp"
            )
    
    def test_custom_rejects_insert(self):
        """Custom aggregation debe rechazar INSERT."""
        from tfm.tools.aggregations import run_custom_aggregation
        
        with pytest.raises(ValueError, match="SELECT"):
            run_custom_aggregation(
                "INSERT INTO reviews VALUES (1, 2, 3)",
                dataset="yelp"
            )
    
    def test_custom_allows_select(self):
        """Custom aggregation debe permitir SELECT (aunque no implementado)."""
        from tfm.tools.aggregations import run_custom_aggregation
        
        # Debe pasar la validación pero fallar por NotImplementedError
        with pytest.raises(NotImplementedError):
            run_custom_aggregation(
                "SELECT * FROM reviews LIMIT 10",
                dataset="yelp"
            )
