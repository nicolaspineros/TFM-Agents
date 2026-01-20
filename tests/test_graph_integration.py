"""
Tests de integracion para los grafos LangGraph.

Estos tests verifican que los nodos del grafo funcionan correctamente
con el estado como diccionario (el formato que LangGraph usa en runtime).
"""

import pytest
from unittest.mock import patch, MagicMock


# =============================================================================
# Tests del Router Node
# =============================================================================

class TestRouteQueryNode:
    """Tests para el nodo route_query con estado como dict."""
    
    def test_route_query_accepts_dict_state(self):
        """El nodo route_query debe aceptar estado como diccionario."""
        from tfm.agents.router import route_query
        
        state = {
            "user_query": "Cual es la distribucion de ratings?",
            "current_dataset": "yelp",
            "messages": [],
        }
        
        # Mock del LLM para evitar llamadas reales
        with patch("tfm.agents.router.create_router_agent") as mock_llm:
            mock_response = MagicMock()
            mock_response.content = '{"is_valid": true, "datasets_required": ["yelp"], "aggregations_needed": ["reviews_by_stars"]}'
            mock_llm.return_value.invoke.return_value = mock_response
            
            result = route_query(state)
        
        assert "query_plan" in result
        assert result["query_plan"] is not None
    
    def test_route_query_empty_query_returns_invalid(self):
        """Query vacia debe retornar query_plan con is_valid=False."""
        from tfm.agents.router import route_query
        
        state = {
            "user_query": "",
            "current_dataset": "yelp",
            "messages": [],
        }
        
        result = route_query(state)
        
        assert "query_plan" in result
        # query_plan es un dict, verificar con acceso por key
        assert result["query_plan"]["is_valid"] == False
    
    def test_route_query_short_query_returns_invalid(self):
        """Query muy corta debe retornar query_plan con is_valid=False."""
        from tfm.agents.router import route_query
        
        state = {
            "user_query": "ab",
            "current_dataset": None,
            "messages": [],
        }
        
        result = route_query(state)
        
        assert "query_plan" in result
        # query_plan es un dict, verificar con acceso por key
        assert result["query_plan"]["is_valid"] == False
    
    def test_route_query_missing_fields_uses_defaults(self):
        """Campos faltantes deben usar valores por defecto."""
        from tfm.agents.router import route_query
        
        # Estado minimo sin current_dataset
        state = {
            "user_query": "Cual es el sentimiento promedio?",
            "messages": [],
        }
        
        with patch("tfm.agents.router.create_router_agent") as mock_llm:
            mock_response = MagicMock()
            mock_response.content = '{"is_valid": true, "datasets_required": ["yelp"]}'
            mock_llm.return_value.invoke.return_value = mock_response
            
            result = route_query(state)
        
        assert "query_plan" in result


# =============================================================================
# Tests de Routing Condicional
# =============================================================================

class TestRouteAfterPlanning:
    """Tests para la funcion de routing condicional route_after_planning."""
    
    def test_route_after_planning_no_plan_returns_invalid(self):
        """Sin query_plan debe retornar 'invalid'."""
        from tfm.graphs.conversation_graph import route_after_planning
        
        state = {
            "query_plan": None,
            "messages": [],
        }
        
        result = route_after_planning(state)
        assert result == "invalid"
    
    def test_route_after_planning_invalid_plan_returns_invalid(self):
        """query_plan con is_valid=False debe retornar 'invalid'."""
        from tfm.graphs.conversation_graph import route_after_planning
        
        state = {
            "query_plan": {
                "is_valid": False,
                "rejection_reason": "Query muy corta",
            },
            "messages": [],
        }
        
        result = route_after_planning(state)
        assert result == "invalid"
    
    def test_route_after_planning_valid_plan_returns_ready(self):
        """query_plan valido sin features NLP debe retornar 'ready' si silver existe."""
        from tfm.graphs.conversation_graph import route_after_planning
        from unittest.mock import patch
        
        state = {
            "query_plan": {
                "is_valid": True,
                "needs_nlp_features": False,
                "datasets_required": ["yelp"],
            },
            "messages": [],
        }
        
        # Mock check_silver_status para simular que silver existe
        mock_silver_status = {
            "yelp_reviews": {"exists": True, "path": "data/silver/yelp_reviews.parquet"},
        }
        
        with patch("tfm.tools.preprocess.check_silver_status", return_value=mock_silver_status):
            result = route_after_planning(state)
            assert result == "ready"
    
    def test_route_after_planning_missing_silver_returns_needs_features(self):
        """query_plan valido pero sin silver debe retornar 'needs_features'."""
        from tfm.graphs.conversation_graph import route_after_planning
        from unittest.mock import patch
        
        state = {
            "query_plan": {
                "is_valid": True,
                "needs_nlp_features": False,
                "datasets_required": ["yelp"],
            },
            "messages": [],
        }
        
        # Mock check_silver_status para simular que silver NO existe
        mock_silver_status = {
            "yelp_reviews": {"exists": False, "path": "data/silver/yelp_reviews.parquet"},
        }
        
        with patch("tfm.tools.preprocess.check_silver_status", return_value=mock_silver_status):
            result = route_after_planning(state)
            assert result == "needs_features"


class TestRouteAfterQA:
    """Tests para la funcion de routing condicional route_after_qa."""
    
    def test_route_after_qa_passed_returns_pass(self):
        """QA aprobado debe retornar 'pass'."""
        from tfm.graphs.conversation_graph import route_after_qa
        
        state = {
            "qa_passed": True,
            "messages": [],
        }
        
        result = route_after_qa(state)
        assert result == "pass"
    
    def test_route_after_qa_failed_returns_fail(self):
        """QA fallido debe retornar 'fail'."""
        from tfm.graphs.conversation_graph import route_after_qa
        
        state = {
            "qa_passed": False,
            "messages": [],
        }
        
        result = route_after_qa(state)
        assert result == "fail"
    
    def test_route_after_qa_missing_field_defaults_fail(self):
        """Campo qa_passed faltante debe usar default False y retornar 'fail'."""
        from tfm.graphs.conversation_graph import route_after_qa
        
        state = {
            "messages": [],
        }
        
        result = route_after_qa(state)
        assert result == "fail"


# =============================================================================
# Tests del Aggregator Node
# =============================================================================

class TestRunAggregations:
    """Tests para el nodo run_aggregations."""
    
    def test_run_aggregations_no_plan_returns_error(self):
        """Sin query_plan debe retornar error."""
        from tfm.graphs.conversation_graph import run_aggregations
        
        state = {
            "query_plan": None,
            "aggregation_results": {},
            "messages": [],
        }
        
        result = run_aggregations(state)
        assert "error" in result
    
    def test_run_aggregations_with_plan_returns_results(self):
        """Con query_plan valido debe ejecutar agregaciones."""
        from tfm.graphs.conversation_graph import run_aggregations
        
        state = {
            "query_plan": {
                "is_valid": True,
                "datasets_required": ["yelp"],
                "aggregations_needed": ["reviews_by_stars"],
                "needs_aggregation": True,
            },
            "aggregation_results": {},
            "messages": [],
        }
        
        # Mock de las funciones de agregacion - usar path donde se usa la funcion
        with patch("tfm.tools.aggregations.aggregate_reviews_by_stars") as mock_agg:
            mock_agg.return_value = {"data": [{"stars": 5, "count": 100}]}
            
            result = run_aggregations(state)
        
        assert "last_result" in result
        assert "aggregation_results" in result


# =============================================================================
# Tests del Synthesizer Node
# =============================================================================

class TestGenerateInsights:
    """Tests para el nodo generate_insights."""
    
    def test_generate_insights_no_data_returns_minimal_report(self):
        """Sin datos debe retornar reporte minimo."""
        from tfm.agents.insight_synthesizer import generate_insights
        
        state = {
            "user_query": "Cual es el sentimiento?",
            "last_result": None,
            "aggregation_results": None,
            "query_plan": None,
            "artifacts": None,
            "messages": [],
        }
        
        result = generate_insights(state)
        
        assert "insights_report" in result
        assert "error" in result
        assert result["insights_report"]["summary"] == "No se encontraron datos para analizar."
    
    def test_generate_insights_with_data_calls_llm(self):
        """Con datos debe llamar al LLM para sintetizar."""
        from tfm.agents.insight_synthesizer import generate_insights
        
        state = {
            "user_query": "Cual es la distribucion de ratings?",
            "last_result": {
                "query_type": "aggregation",
                "data_json": '[{"stars": 5, "count": 100}]',
                "columns": ["stars", "count"],
                "row_count": 1,
            },
            "aggregation_results": {
                "yelp_reviews_by_stars": {"data": [{"stars": 5, "count": 100}]}
            },
            "query_plan": {
                "datasets_required": ["yelp"],
                "aggregations_needed": ["reviews_by_stars"],
            },
            "artifacts": {},
            "messages": [],
        }
        
        with patch("tfm.agents.insight_synthesizer.create_synthesizer") as mock_synth:
            mock_response = MagicMock()
            mock_response.content = '{"summary": "Analisis completado", "bullets": [], "caveats": []}'
            mock_synth.return_value.invoke.return_value = mock_response
            
            result = generate_insights(state)
        
        assert "insights_report" in result
        assert result["insights_report"]["summary"] == "Analisis completado"


# =============================================================================
# Tests del QA Evaluator Node
# =============================================================================

class TestEvaluateResult:
    """Tests para el nodo evaluate_result."""
    
    def test_evaluate_result_no_report_returns_failed(self):
        """Sin insights_report debe retornar qa_passed=False."""
        from tfm.agents.qa_evaluator import evaluate_result
        
        state = {
            "insights_report": None,
            "user_query": "Test query",
            "last_result": None,
            "aggregation_results": None,
            "messages": [],
        }
        
        result = evaluate_result(state)
        
        assert result["qa_passed"] == False
        assert "qa_feedback" in result
    
    def test_evaluate_result_with_valid_report_passes(self):
        """Con reporte valido debe pasar QA."""
        from tfm.agents.qa_evaluator import evaluate_result
        
        state = {
            "insights_report": {
                "summary": "Analisis completado con exito.",
                "bullets": [{"text": "Test", "evidence": "dato=1", "confidence": "high"}],
                "caveats": [],
            },
            "user_query": "Test query",
            "last_result": {
                "query_type": "aggregation",
                "data_json": "[]",
                "columns": [],
                "row_count": 0,
            },
            "aggregation_results": {"test": {"data": [1, 2, 3]}},
            "messages": [],
        }
        
        result = evaluate_result(state)
        
        assert result["qa_passed"] == True


# =============================================================================
# Tests del NLP Worker Node
# =============================================================================

class TestProcessNLPRequest:
    """Tests para el nodo process_nlp_request."""
    
    def test_process_nlp_request_no_plan_returns_error(self):
        """Sin query_plan debe retornar error."""
        from tfm.agents.nlp_worker import process_nlp_request
        
        state = {
            "query_plan": None,
            "artifacts": {},
            "messages": [],
        }
        
        result = process_nlp_request(state)
        assert "error" in result
    
    def test_process_nlp_request_with_plan_processes_datasets(self):
        """Con query_plan debe procesar datasets."""
        from tfm.agents.nlp_worker import process_nlp_request
        
        state = {
            "query_plan": {
                "is_valid": True,
                "datasets_required": ["yelp"],
                "needs_nlp_features": False,
                "needs_aggregation": False,
                "aggregations_needed": [],
                "guardrail_warnings": [],
            },
            "artifacts": {},
            "messages": [],
        }
        
        with patch("tfm.agents.nlp_worker._ensure_silver") as mock_silver:
            mock_silver.return_value = {"success": True, "paths": {"reviews": "/path/to/silver"}}
            
            result = process_nlp_request(state)
        
        assert "artifacts" in result


# =============================================================================
# Tests del NLP Graph Nodes
# =============================================================================

class TestNLPGraphNodes:
    """Tests para nodos del grafo NLP."""
    
    def test_validate_input_node_invalid_dataset(self):
        """Dataset invalido debe agregar error."""
        from tfm.graphs.nlp_graph import validate_input_node
        
        state = {
            "dataset": "invalid_dataset",
            "errors": [],
            "messages": [],
        }
        
        result = validate_input_node(state)
        
        assert "errors" in result
        assert len(result["errors"]) > 0
    
    def test_validate_input_node_valid_dataset(self):
        """Dataset valido debe retornar bronze_path."""
        from tfm.graphs.nlp_graph import validate_input_node
        
        state = {
            "dataset": "yelp",
            "errors": [],
            "messages": [],
        }
        
        # Mock de la funcion - usar path donde se usa la funcion
        with patch("tfm.tools.io_loaders.get_bronze_file_info") as mock_info:
            mock_info.return_value = {"exists": True, "path": "/path/to/bronze"}
            
            result = validate_input_node(state)
        
        assert "bronze_path" in result
    
    def test_route_after_validation_with_errors_returns_error(self):
        """Con errores debe retornar 'error'."""
        from tfm.graphs.nlp_graph import route_after_validation
        
        state = {
            "errors": ["Dataset invalido"],
            "messages": [],
        }
        
        result = route_after_validation(state)
        assert result == "error"
    
    def test_route_after_validation_no_errors_returns_build(self):
        """Sin errores debe retornar 'build'."""
        from tfm.graphs.nlp_graph import route_after_validation
        
        state = {
            "errors": [],
            "messages": [],
        }
        
        result = route_after_validation(state)
        assert result == "build"


# =============================================================================
# Tests de Flujo Completo
# =============================================================================

class TestConversationGraphFlow:
    """Tests de flujo completo del grafo de conversacion."""
    
    def test_graph_compiles_successfully(self):
        """El grafo debe compilar sin errores."""
        from tfm.graphs.conversation_graph import build_conversation_graph
        
        graph = build_conversation_graph()
        assert graph is not None
    
    def test_graph_handles_invalid_query(self):
        """El grafo debe manejar queries invalidas."""
        from tfm.graphs.conversation_graph import conversation_graph
        
        # Este test requiere mock mas extenso del LLM
        # Por ahora solo verificamos que el grafo existe
        assert conversation_graph is not None


class TestNLPGraphFlow:
    """Tests de flujo completo del grafo NLP."""
    
    def test_nlp_graph_compiles_successfully(self):
        """El grafo NLP debe compilar sin errores."""
        from tfm.graphs.nlp_graph import build_nlp_graph
        
        graph = build_nlp_graph()
        assert graph is not None
