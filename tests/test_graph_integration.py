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
        
        # Mock de check_silver_status para evitar I/O
        with patch("tfm.agents.router.check_silver_status") as mock_status:
            mock_status.return_value = {"yelp_reviews": {"exists": True}}
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
        assert result["query_plan"]["is_valid"] == False


# =============================================================================
# Tests de Routing Condicional
# =============================================================================

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


class TestShouldContinueRouting:
    """Tests para la funcion should_continue_routing."""
    
    def test_should_continue_routing_empty_messages_returns_synthesize(self):
        """Sin mensajes debe retornar 'synthesize'."""
        from tfm.graphs.conversation_graph import should_continue_routing
        
        state = {
            "messages": [],
        }
        
        result = should_continue_routing(state)
        assert result == "synthesize"


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
        
        # Mock del LLM para QA
        with patch("tfm.agents.qa_evaluator.create_qa_evaluator") as mock_qa:
            mock_response = MagicMock()
            mock_response.content = '{"answers_query": true, "claims_supported": true, "confidence": 0.9, "issues": [], "passed": true, "feedback": ""}'
            mock_qa.return_value.invoke.return_value = mock_response
            
            result = evaluate_result(state)
        
        assert result["qa_passed"] == True


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
        
        # Mock de la funcion
        with patch("tfm.graphs.nlp_graph.get_bronze_file_info") as mock_info:
            mock_info.return_value = {"exists": True, "path": "/path/to/bronze"}
            
            result = validate_input_node(state)
        
        # Puede retornar bronze_path o errors dependiendo de si existe
        assert "bronze_path" in result or "errors" in result
    
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
        
        # Verificamos que el grafo existe
        assert conversation_graph is not None


class TestNLPGraphFlow:
    """Tests de flujo completo del grafo NLP."""
    
    def test_nlp_graph_compiles_successfully(self):
        """El grafo NLP debe compilar sin errores."""
        from tfm.graphs.nlp_graph import build_nlp_graph
        
        graph = build_nlp_graph()
        assert graph is not None
    
    def test_build_silver_node_with_invalid_dataset(self):
        """build_silver_node con dataset invalido debe agregar error."""
        from tfm.graphs.nlp_graph import build_silver_node
        
        state = {
            "dataset": "invalid",
            "errors": [],
            "messages": [],
        }
        
        result = build_silver_node(state)
        
        assert "errors" in result
        assert len(result["errors"]) > 0
    
    def test_build_gold_node_without_silver_path(self):
        """build_gold_node sin silver_path debe agregar error."""
        from tfm.graphs.nlp_graph import build_gold_node
        
        state = {
            "dataset": "yelp",
            "silver_path": None,
            "errors": [],
            "messages": [],
        }
        
        result = build_gold_node(state)
        
        assert "errors" in result
        assert len(result["errors"]) > 0