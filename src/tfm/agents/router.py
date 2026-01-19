"""
Router / Query Planner Agent.

El Router es el agente principal que:
1. Interpreta la pregunta del usuario
2. Inspecciona schemas disponibles via profiling
3. Aplica guardrails (valida si la query es posible)
4. Genera un QueryPlan con:
   - Datasets necesarios
   - Filtros a aplicar
   - Tools a ejecutar
   - Artefactos a construir (lazy computation)

El Router NO ejecuta tools ni procesa datos; solo planifica.
"""

from typing import Optional, Any, List, Dict
import json

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from tfm.config.settings import get_settings
from tfm.schemas.state import TFMState, QueryPlan
from tfm.schemas.request import UserQuery
from tfm.tools.profiling import profile_dataset, check_artifacts_status
from tfm.tools.preprocess import check_silver_status
from tfm.tools.features import get_gold_status


# System prompt para el Router
ROUTER_SYSTEM_PROMPT = """Eres el Router/Query Planner de un sistema de analisis de reseñas.

IMPORTANTE: Tu trabajo es generar un plan de ejecucion. NO rechaces consultas validas.
SIEMPRE marca is_valid=true si la consulta puede responderse con las agregaciones disponibles.
1. Interpretar la pregunta del usuario
2. Determinar que datasets y filtros son necesarios
3. Verificar si la consulta es posible (guardrails)
4. Generar un plan de ejecucion

DATASETS Y SUS COLUMNAS:

YELP (ingles):
- Columnas: review_id, user_id, business_id, stars (1-5), text, date, year, month
- TIENE fechas: SI
- Agregaciones soportadas: reviews_by_stars, reviews_by_month, user_stats, business_stats, ambiguous_reviews, text_length

ES (espanol):
- Columnas: review_id, product_id, reviewer_id, stars (1-5), text, category
- TIENE fechas: NO
- Agregaciones soportadas: reviews_by_stars, ambiguous_reviews, text_length
- NO soporta: reviews_by_month (sin fechas)

OLIST (portugues):
- Columnas reviews: review_id, order_id, review_score (1-5), text
- Columnas orders: order_id, customer_id, purchase_date, year, month, order_total
- TIENE fechas: SI
- TIENE ventas: SI
- Agregaciones soportadas: reviews_by_stars, reviews_by_month, olist_sales, ambiguous_reviews, text_length

MAPEO DE PREGUNTAS A AGREGACIONES:

"distribucion de ratings" -> reviews_by_stars (TODOS los datasets)
"distribucion de estrellas" -> reviews_by_stars (TODOS los datasets)
"cuantas resenas por estrellas" -> reviews_by_stars (TODOS los datasets)
"resenas por mes" -> reviews_by_month (yelp, olist - NO es)
"tendencia temporal" -> reviews_by_month (yelp, olist - NO es)
"resenas ambiguas" -> ambiguous_reviews (TODOS los datasets)
"resenas de 3 estrellas" -> ambiguous_reviews (TODOS los datasets)
"usuarios influyentes" -> user_stats (solo yelp)
"estadisticas de negocios" -> business_stats (solo yelp)
"ventas" -> olist_sales (solo olist)

REGLAS DE VALIDACION:

1. SIEMPRE marca is_valid=true si la pregunta es sobre distribucion de ratings/estrellas
2. TODOS los datasets tienen columna de rating (stars o review_score)
3. Solo rechaza (is_valid=false) si:
   - La pregunta pide fechas/temporal para dataset ES
   - La pregunta pide ventas para dataset que no es Olist
   - La pregunta es incomprensible o vacia

ESTADO ACTUAL:
{artifacts_status}

FORMATO DE RESPUESTA:
Responde SOLO con JSON valido:
{{
    "is_valid": true/false,
    "datasets_required": ["yelp"],
    "aggregations_needed": ["reviews_by_stars"],
    "needs_nlp_features": true/false,
    "needs_aggregation": true/false,
    "rejection_reason": null o "razon",
    "guardrail_warnings": []
}}

RECUERDA: Para preguntas sobre "distribucion de ratings/estrellas", SIEMPRE is_valid=true y aggregations_needed=["reviews_by_stars"].
"""


QUERY_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "is_valid": {"type": "boolean"},
        "datasets_required": {"type": "array", "items": {"type": "string"}},
        "aggregations_needed": {"type": "array", "items": {"type": "string"}},
        "needs_nlp_features": {"type": "boolean"},
        "needs_aggregation": {"type": "boolean"},
        "rejection_reason": {"type": ["string", "null"]},
        "guardrail_warnings": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["is_valid", "datasets_required"]
}


def create_router_agent():
    """
    Crea instancia del Router agent.
    
    Returns:
        LLM configurado para routing con structured output
    """
    settings = get_settings()
    
    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=0,  # Deterministico para routing
        api_key=settings.openai_api_key,
    )
    
    return llm


def get_artifacts_status_summary() -> str:
    """
    Obtiene resumen del estado de artefactos para el prompt.
    
    Returns:
        String formateado con estado de silver y gold
    """
    lines = []
    
    try:
        # Silver status
        silver_status = check_silver_status()
        lines.append("SILVER LAYER:")
        for name, info in silver_status.items():
            status = "OK" if info.get("exists") else "NO EXISTE"
            size = f" ({info.get('size_mb', 0):.1f} MB)" if info.get("exists") else ""
            lines.append(f"  - {name}: {status}{size}")
    except Exception as e:
        lines.append(f"SILVER LAYER: Error verificando - {e}")
    
    try:
        # Gold status
        gold_status = get_gold_status()
        lines.append("\nGOLD LAYER:")
        for name, info in gold_status.items():
            status = "OK" if info.get("exists") else "NO EXISTE"
            rows = f" ({info.get('rows', 0):,} filas)" if info.get("exists") else ""
            lines.append(f"  - {name}: {status}{rows}")
    except Exception as e:
        lines.append(f"GOLD LAYER: Error verificando - {e}")
    
    return "\n".join(lines)


def route_query(state: TFMState) -> dict[str, Any]:
    """
    Nodo de grafo que ejecuta el routing.
    
    Lee la query del estado, genera QueryPlan, y retorna actualizaciones.
    Tambien verifica que artefactos existen y marca missing_artifacts.
    
    Args:
        state: Estado actual del grafo (diccionario en runtime)
        
    Returns:
        Dict con query_plan y opcionalmente error para actualizar estado
    """
    # Acceso por keys - LangGraph pasa el estado como dict en runtime
    user_query = state.get("user_query", "")
    current_dataset = state.get("current_dataset")
    
    print(f"[route_query] user_query='{user_query}', current_dataset='{current_dataset}'")
    
    # Validacion basica
    if not user_query or len(user_query.strip()) < 3:
        # Convertir a dict para consistencia con el estado
        query_plan = QueryPlan(
            is_valid=False,
            rejection_reason="Query vacia o muy corta. Por favor proporciona una pregunta."
        )
        return {
            "query_plan": query_plan.model_dump(),
            "error": "Query vacia o muy corta"
        }
    
    # Obtener contexto de artefactos
    artifacts_status = get_artifacts_status_summary()
    
    # Crear prompt con contexto
    system_prompt = ROUTER_SYSTEM_PROMPT.format(artifacts_status=artifacts_status)
    
    try:
        # Llamar al LLM
        llm = create_router_agent()
        
        # Incluir dataset preferido si existe
        user_message = f"Pregunta del usuario: {user_query}"
        if current_dataset:
            user_message += f"\nDataset seleccionado: {current_dataset}"
        
        print(f"[route_query] Enviando a LLM: {user_message}")
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message)
        ]
        
        response = llm.invoke(messages)
        
        # Parsear respuesta JSON
        response_text = response.content
        print(f"[route_query] Respuesta LLM: {response_text[:500]}...")
        
        plan_dict = _parse_json_response(response_text)
        print(f"[route_query] Plan parseado: is_valid={plan_dict.get('is_valid')}, datasets={plan_dict.get('datasets_required')}, aggs={plan_dict.get('aggregations_needed')}")
        
        # Si el usuario especifico dataset, usarlo
        if current_dataset and current_dataset in ["yelp", "es", "olist"]:
            plan_dict["datasets_required"] = [current_dataset]
            print(f"[route_query] Override dataset: {current_dataset}")
        
        # CORRECCION: Si la query es sobre distribucion de ratings, forzar agregacion correcta
        query_lower = user_query.lower()
        if any(kw in query_lower for kw in ["distribucion", "distribución", "ratings", "estrellas", "rating"]):
            if "reviews_by_stars" not in plan_dict.get("aggregations_needed", []):
                plan_dict["aggregations_needed"] = ["reviews_by_stars"]
                print(f"[route_query] Override agregacion: reviews_by_stars")
            # Esta query ES valida para todos los datasets
            plan_dict["is_valid"] = True
            plan_dict["rejection_reason"] = None
            print(f"[route_query] Override is_valid=True (query de distribucion)")
        
        # Verificar que artefactos existen y marcar missing_artifacts
        missing_artifacts = _check_missing_artifacts(
            plan_dict.get("datasets_required", ["yelp"]),
            plan_dict.get("needs_nlp_features", False)
        )
        print(f"[route_query] missing_artifacts={missing_artifacts}")
        
        # Crear QueryPlan y convertir a dict para consistencia
        query_plan = QueryPlan(
            is_valid=plan_dict.get("is_valid", True),
            datasets_required=plan_dict.get("datasets_required", ["yelp"]),
            aggregations_needed=plan_dict.get("aggregations_needed", ["reviews_by_stars"]),
            needs_nlp_features=plan_dict.get("needs_nlp_features", False),
            needs_aggregation=plan_dict.get("needs_aggregation", True),
            rejection_reason=plan_dict.get("rejection_reason"),
            guardrail_warnings=plan_dict.get("guardrail_warnings", []),
            missing_artifacts=missing_artifacts,
        )
        
        print(f"[route_query] QueryPlan final: is_valid={query_plan.is_valid}, datasets={query_plan.datasets_required}")
        return {"query_plan": query_plan.model_dump()}
        
    except Exception as e:
        # En caso de error, usar defaults seguros basados en dataset
        default_dataset = current_dataset if current_dataset in ["yelp", "es", "olist"] else "yelp"
        
        # Verificar artefactos incluso en caso de error
        missing_artifacts = _check_missing_artifacts([default_dataset], False)
        
        # Convertir a dict para consistencia
        query_plan = QueryPlan(
            is_valid=True,
            datasets_required=[default_dataset],
            aggregations_needed=["reviews_by_stars"],
            needs_nlp_features=False,
            needs_aggregation=True,
            guardrail_warnings=[f"Usando defaults por error en routing: {str(e)}"],
            missing_artifacts=missing_artifacts,
        )
        return {"query_plan": query_plan.model_dump()}


def _parse_json_response(response_text: str) -> Dict[str, Any]:
    """
    Parsea respuesta JSON del LLM.
    
    Maneja casos donde el JSON esta envuelto en markdown.
    """
    text = response_text.strip()
    
    # Remover markdown code blocks si existen
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    
    if text.endswith("```"):
        text = text[:-3]
    
    text = text.strip()
    
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Intentar extraer JSON con regex
        import re
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        print(f"Error al parsear JSON: {text}")
        # Retornar defaults
        return {
            "is_valid": True,
            "datasets_required": ["yelp"],
            "aggregations_needed": ["reviews_by_stars"],
            "needs_aggregation": True,
        }


def _build_routing_context() -> dict[str, Any]:
    """
    Construye contexto para el Router.
    
    Incluye:
    - Perfiles de datasets disponibles
    - Status de artefactos
    - Capacidades por dataset
    
    Returns:
        Dict con contexto completo
    """
    context = {
        "datasets": {},
        "artifacts_status": {},
    }
    
    for dataset in ["yelp", "es", "olist"]:
        try:
            # Intentar obtener perfil (puede fallar si no existe silver)
            profile = profile_dataset(dataset, layer="silver")
            context["datasets"][dataset] = {
                "exists": profile.get("exists", False),
                "capabilities": profile.get("capabilities", {}),
                "row_count": profile.get("row_count", 0),
            }
        except (NotImplementedError, FileNotFoundError):
            context["datasets"][dataset] = {"exists": False}
        
        # Status de artefactos
        try:
            context["artifacts_status"][dataset] = check_artifacts_status(dataset)
        except:
            context["artifacts_status"][dataset] = {"bronze": False, "silver": False, "gold": False}
    
    return context


def apply_guardrails(
    query: UserQuery,
    context: dict[str, Any]
) -> tuple[bool, list[str]]:
    """
    Aplica guardrails a la query.
    
    Verifica:
    - Query no vacia
    - Dataset solicitado existe
    - Capacidades requeridas disponibles
    - Filtros validos
    
    Args:
        query: Query del usuario
        context: Contexto de datasets
        
    Returns:
        Tuple (is_valid, warnings/errors)
    """
    warnings = []
    
    # Verificar query no vacia
    if not query.text or len(query.text.strip()) < 3:
        return False, ["La pregunta esta vacia o es muy corta"]
    
    # Verificar dataset preferido
    if query.preferred_dataset:
        ds_info = context["datasets"].get(query.preferred_dataset, {})
        if not ds_info.get("exists", False):
            warnings.append(
                f"Dataset {query.preferred_dataset} no tiene silver layer. "
                "Se necesita construir primero."
            )
    
    # Detectar requests temporales en dataset ES
    temporal_keywords = ["mes", "ano", "fecha", "temporal", "tendencia", "evolucion", "monthly", "trend"]
    if any(kw in query.text.lower() for kw in temporal_keywords):
        if query.preferred_dataset == "es":
            return False, [
                "El dataset de espanol NO tiene campo de fecha. "
                "No es posible hacer analisis temporal. "
                "Considera usar el dataset de Yelp o Olist para analisis temporal."
            ]
    
    # Detectar requests de ventas en datasets sin ventas
    sales_keywords = ["ventas", "revenue", "ingresos", "prediccion", "forecast", "sales"]
    if any(kw in query.text.lower() for kw in sales_keywords):
        if query.preferred_dataset in ["yelp", "es"]:
            return False, [
                "Solo el dataset Olist tiene datos de ventas. "
                "Cambia a dataset='olist' para analisis de ventas."
            ]
    
    return True, warnings


def suggest_clarifications(
    query: UserQuery,
    context: dict[str, Any]
) -> list[str]:
    """
    Sugiere clarificaciones si la query es ambigua.
    
    Args:
        query: Query del usuario
        context: Contexto de datasets
        
    Returns:
        Lista de sugerencias
    """
    suggestions = []
    
    # Si no especifico dataset
    if not query.preferred_dataset:
        suggestions.append(
            "No especificaste dataset. Opciones disponibles: "
            "yelp (ingles, con fechas), es (espanol, sin fechas), "
            "olist (portugues, con ventas)"
        )
    
    # Si query muy generica
    if len(query.text.split()) < 5:
        suggestions.append(
            "Tu pregunta es muy general. Ejemplos mas especificos:\n"
            "- 'Cual es el sentimiento promedio de restaurantes en Yelp durante 2022?'\n"
            "- 'Que categorias tienen peor sentimiento en el dataset espanol?'\n"
            "- 'Como correlacionan las ventas con el sentimiento de reviews en Olist?'"
        )
    
    return suggestions


def _check_missing_artifacts(
    datasets_required: List[str],
    needs_nlp_features: bool
) -> List[str]:
    """
    Verifica que artefactos existen y retorna lista de faltantes.
    
    Args:
        datasets_required: Lista de datasets necesarios
        needs_nlp_features: Si requiere gold features
        
    Returns:
        Lista de artefactos faltantes (ej: ["yelp_silver", "yelp_gold"])
    """
    missing = []
    
    try:
        # Verificar silver
        silver_status = check_silver_status()
        
        for dataset in datasets_required:
            # Mapear dataset a key de silver_status
            if dataset == "yelp":
                silver_keys = ["yelp_reviews"]
            elif dataset == "es":
                silver_keys = ["es"]
            elif dataset == "olist":
                silver_keys = ["olist_orders", "olist_reviews"]
            else:
                silver_keys = [f"{dataset}_reviews"]
            
            for key in silver_keys:
                if key in silver_status and not silver_status[key].get("exists", False):
                    missing.append(f"{dataset}_silver")
                    break
        
        # Verificar gold si se necesita
        if needs_nlp_features:
            gold_status = get_gold_status()
            for dataset in datasets_required:
                gold_key = f"{dataset}_features"
                if gold_key in gold_status and not gold_status[gold_key].get("exists", False):
                    missing.append(f"{dataset}_gold")
                    
    except Exception:
        # Si hay error verificando, marcar todo como potencialmente faltante
        for dataset in datasets_required:
            missing.append(f"{dataset}_silver")
    
    return list(set(missing))  # Eliminar duplicados
