# TFM: Sistema Agentic de Análisis de Reseñas con LangGraph

## Visión General

Este proyecto implementa un **sistema agentic real** para análisis inteligente de reseñas y predicción de ventas. Utiliza **LangGraph** como motor de orquestación de agentes y **NLP/ML** como núcleo analítico.

### Diferenciador Clave

El LLM **NO** procesa ni agrega datasets masivos en contexto. En su lugar:

1. **Actúa como planificador/razonador**: inspecciona qué artefactos (features, agregados, modelos) ya existen
2. **Decide dinámicamente**: qué cálculos son necesarios para responder la pregunta del usuario
3. **Ejecuta tools deterministas**: que operan sobre storage estructurado (Parquet + DuckDB)
4. **Comunicación natural con el usuario**: toma las dudas del usuario y tiene el contexto necesarios para decidir y proceder en los casos de uso. (en los EDAs colocamos algunos casos de uso)

Este enfoque permite:
-  Escalar a cientos de miles de reseñas sin saturar el contexto del LLM
-  Mantener reproducibilidad total
-  Demostrar valor agentic con rutas condicionales
-  Implementar lazy computation (calcular solo si no existe)
-  QA y evaluación trazable en LangSmith

---

## Arquitectura

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CONVERSATION GRAPH                                │
│  ┌──────────┐    ┌───────────────┐    ┌────────────────┐    ┌──────────┐    │
│  │  Router  │───▶│  NLP/ML       │───▶│    Insight     │───▶│   QA   │    │
│  │  Agent   │    │  Worker       │    │  Synthesizer   │    │ Evaluator│    │
│  └────┬─────┘    └───────┬───────┘    └────────────────┘    └──────────┘    │
│       │                  │                                                  │
│       │ inspects         │ executes tools                                   │
│       ▼                  ▼                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    TOOLS LAYER (Deterministic)                      │    │
│  │  ┌─────────┐ ┌──────────┐ ┌───────────┐ ┌────────────┐ ┌─────────┐  │    │
│  │  │ Storage │ │ Profiling│ │ Sentiment │ │ Aggregation│ │Retrieval│  │    │
│  │  └─────────┘ └──────────┘ └───────────┘ └────────────┘ └─────────┘  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           STORAGE LAYER                                     │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐   │
│  │   Bronze    │───▶│   Silver    │───▶│    Gold     │───▶│   DuckDB  │    │
│  │  (Raw)      │    │ (Limpio)    │    │ (Features)  │    │ (Warehouse) │   │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘   │
│       Parquet           Parquet           Parquet          tfm.duckdb       │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Estructura del Repositorio

```
tfm-agents/
|-- data/
|   |-- bronze/          # Datos crudos (JSONL, CSV)
|   |   |-- yelp/
|   |   |-- rese_esp/
|   |   +-- olist_ecommerce/
|   |-- silver/          # Datos limpios y normalizados (Parquet)
|   |-- gold/            # Features, agregados, metricas (Parquet + JSON)
|   +-- test/            # Datos de prueba
|       +-- es_annotated_reviews.csv  # Corpus anotado espanol (300 reviews)
|-- models/
|   |-- sentiment/       # Modelos de sentimiento entrenados
|   |   |-- unified_svm_sentiment.joblib
|   |   |-- yelp_svm_sentiment.joblib
|   |   |-- es_logistic_sentiment.joblib
|   |   +-- olist_svm_sentiment.joblib
|   +-- aspects/         # Modelos de aspectos
|       |-- aspect_classifier_lr.joblib
|       |-- aspect_tfidf.joblib
|       |-- aspect_taxonomy.json
|       +-- sentiment_lexicon.json
|-- warehouse/
|   +-- tfm.duckdb       # Base de datos analitica
|-- runs/                # Checkpoints de grafos (SQLite)
|-- docs/
|   +-- data_contracts.md
|-- notebooks/
|   |-- 01_eda_yelp.ipynb
|   |-- 02_eda_es.ipynb
|   |-- 03_eda_olist.ipynb
|   |-- 04_pruebas_fase3.ipynb
|   |-- 05_nlp_sentiment_models.ipynb   # Entrenamiento modelos sentimiento
|   +-- 06_nlp_aspect_extraction.ipynb  # Extraccion de aspectos
|-- src/
|   +-- tfm/
|       |-- config/      # Configuracion y settings
|       |-- schemas/     # Pydantic models (State, Request, Outputs)
|       |-- tools/       # Herramientas deterministas + NLP
|       |-- agents/      # Agentes LLM
|       +-- graphs/      # Grafos LangGraph
|-- tests/
|-- langgraph.json
|-- pyproject.toml
+-- README.md
```

---

## Agentes

| Agente | Responsabilidad |
|--------|-----------------|
| **Router con Tool Binding** | Recibe pregunta, tiene acceso a herramientas via `bind_tools`, decide dinámicamente qué tool invocar |
| **NLP/ML Worker** | Ejecuta tools de NLP (sentimiento, aspectos, clasificación) y ML |
| **Insight Synthesizer** | Consume resultados de herramientas, redacta `InsightsReport` estructurado |
| **QA/Evaluator** | Checks deterministas (schema, faithfulness), evaluaciones ML (F1/MAE) |

### Arquitectura de Tool Binding

El sistema usa **tool binding** de LangChain para que el LLM descubra automáticamente las herramientas disponibles:

```python
# El LLM ve todas las herramientas y sus descripciones
llm_with_tools = llm.bind_tools([
    get_reviews_distribution,  # Distribución de ratings
    get_sales_by_month,        # Ventas por mes (Olist)
    get_reviews_by_month,      # Reviews por mes (temporal)
    ...
])
```

Esto permite:
- **Descubrimiento dinámico**: El LLM "ve" qué herramientas existen
- **Extensibilidad**: Agregar una tool nueva = el LLM la puede usar automáticamente
- **Decisiones inteligentes**: El LLM decide basándose en las descripciones de cada tool

---

## Herramientas Disponibles (Tools)

El sistema expone las siguientes herramientas que el LLM puede invocar:

### Tools de Análisis de Reviews

| Tool | Descripción | Datasets |
|------|-------------|----------|
| `get_reviews_distribution` | Distribución de ratings/estrellas | yelp, es, olist |
| `get_reviews_by_month` | Tendencia temporal de reviews | yelp, olist (NO es) |
| `get_ambiguous_reviews_analysis` | Análisis de reviews de 3 estrellas | yelp, es, olist |
| `get_text_length_analysis` | Análisis por longitud de texto | yelp, es, olist |

### Tools de Ventas (Solo Olist)

| Tool | Descripción | Datasets |
|------|-------------|----------|
| `get_sales_by_month` | Ventas/órdenes por mes | olist |
| `get_sales_by_category` | Ventas por categoría de producto | olist |
| `get_reviews_sales_correlation` | Correlación reviews vs ventas | olist |

### Tools de Usuarios/Negocios (Solo Yelp)

| Tool | Descripción | Datasets |
|------|-------------|----------|
| `get_user_stats` | Estadísticas de usuarios, top reviewers | yelp |
| `get_business_stats` | Estadísticas de negocios | yelp |

### Tools de Utilidad

| Tool | Descripcion | Uso |
|------|-------------|-----|
| `get_dataset_status` | Verifica si los datos silver existen | Diagnostico |
| `build_dataset_silver` | Construye capa silver para un dataset | Preparacion |

### Tools de NLP (Modelos ML Entrenados)

| Tool | Descripcion | Uso |
|------|-------------|-----|
| `get_sentiment_distribution` | Distribucion de sentimiento en todo el dataset o filtrado | Agregacion NLP |
| `get_aspect_distribution` | Aspectos mencionados (calidad, precio, envio) y su frecuencia | Agregacion NLP |
| `get_ambiguous_reviews_sentiment` | Analiza sentimiento real de reseñas de 3 estrellas | Analisis especial |
| `analyze_sentiment` | Analiza sentimiento de texto individual | Consulta puntual |
| `analyze_review_complete` | Analisis completo de una reseña | Consulta puntual |
| `get_nlp_models_status` | Verifica modelos NLP disponibles | Diagnostico |

**Filtros disponibles:**
- `year`: Filtrar por anio (yelp, olist)
- `stars`: Filtrar por estrellas (1-5)
- `sentiment_filter`: Filtrar por sentimiento (positive/negative/neutral)

**Metricas de los modelos (F1 Score):**
- Yelp (SVM): 0.858
- ES (Logistic): 0.712
- Olist (SVM): 0.832
- Unified (SVM): 0.786

---

## Grafos LangGraph

| Grafo | Descripción |
|-------|-------------|
| `nlp_graph` | Construye silver y gold features (sentimiento, aspectos) para Yelp, ES, Olist reviews |
| `prediction_graph` | Features temporales y modelo de predicción de ventas (Olist) |
| `conversation_graph` | Orquestación principal: router → NLP/prediction → agregaciones → síntesis → QA |
| `evaluation_graph` | Runner de evaluación offline (métricas ML + QA + LangSmith evals) |

---

## Datasets

### 1. Yelp Academic Dataset (Inglés)
- Archivo: `yelp_academic_dataset_review.json` (JSONL)
- Campos clave: `review_id`, `business_id`, `stars`, `text`, `date`

### 2. Reseñas en Español
- Archivo: `reviews_dataframe_completo.csv`
- Campos clave: `review_id`, `product_id`, `stars`, `review_body`, `review_title`, `product_category`, `language`

### 3. Olist Brazilian E-commerce (Portugués)
- Múltiples CSV con órdenes, items, reviews, productos
- Incluye timestamps y precios
- Tabla de traduccion de categorias (pt-en)

Ver detalles en: [`docs/data_contracts.md`](docs/data_contracts.md)

---

## Caso Especial: Reviews Ambiguas (stars == 3)

Las reseñas con `stars == 3` son consideradas **ambiguas**. El sistema:
- Marca estas reseñas con un flag `is_ambiguous` en silver/gold
- Permite análisis específico de este segmento
- Usa lógica especial en clasificación de sentimiento

---

## Fases de Desarrollo

### Fase 1: EDA y Prototipos (COMPLETADA)
- [x] EDA Yelp: distribucion de stars, longitud de texto, fechas, usuarios
- [x] EDA ES: exploracion sin fecha, categorias
- [x] EDA Olist: relacion ventas-reviews, temporalidad
- [x] Prototipos de NLP basico en notebooks
- [x] Documentacion de conclusiones

### Fase 2: Tools Deterministas (COMPLETADA)
- [x] Loaders y storage (bronze -> silver)
- [x] Profiling de datasets
- [x] DuckDB como warehouse
- [x] Preprocesamiento: reviews, users, business

### Fase 3: Features y Agregaciones (COMPLETADA)
- [x] Agregaciones SQL (por stars, mes, categoria)
- [x] Analisis de reviews ambiguas
- [x] Correlaciones reviews-ventas (Olist)
- [x] Estadisticas de usuarios/negocios (Yelp)
- [x] Sentimiento VADER para ingles
- [x] Features gold layer

### Fase 4: Agentes y Grafos (COMPLETADA)
- [x] Router con **tool binding** (LLM descubre herramientas automáticamente)
- [x] Tools de análisis con decorador `@tool` y descripciones claras
- [x] Conversation graph con flujo: route → model → tools → synthesizer → QA
- [x] Insight Synthesizer con LLM
- [x] QA Evaluator con checks determinísticos
- [x] Soporte para todos los datasets (yelp, es, olist) en las herramientas

### Fase 5: NLP y ML Avanzado (EN PROGRESO)
- [x] Modelos de sentimiento TF-IDF + SVM/LogisticRegression
- [x] Modelos por idioma (yelp, es, olist) y modelo unificado
- [x] Extraccion de aspectos (calidad, precio, envio, servicio, etc.)
- [x] Sentimiento por aspecto con ventana de contexto
- [x] Corpus anotado ES (300 reviews) para validacion
- [x] Tools NLP integradas en el sistema agentic
- [ ] Correlacion sentimiento - ventas (Olist)

### Fase 6: Prediccion de Ventas y Evaluacion
- [ ] Metricas ML (F1, MAE)
- [ ] Integracion LangSmith
- [ ] Evaluadores de faithfulness
- [ ] Optimizacion de prompts

---

## Quickstart

```bash
# init
uv init

#  Añadir dependencias core
uv add langgraph langchain langchain-openai
uv add "langgraph-cli[inmem]" --dev
uv add polars pyarrow duckdb pydantic pydantic-settings

# Dependencias para notebooks de NLP/ML (05, 06)
uv add scikit-learn pandas numpy joblib
uv add matplotlib seaborn

# Dependencias para análisis de sentimiento
uv add vadersentiment

# Dependencias de desarrollo
uv add pytest jupyterlab ipykernel --dev

# Instalar el paquete local (IMPORTANTE)
uv pip install -e .

# Verificar instalacion
uv run python -c "from tfm.config.settings import get_settings; print('OK')"

# Construir silver layer (primera vez)
uv run python scripts/build_silver.py --limit 10000

# Ejecutar LangGraph Studio
uv run langgraph dev
```

### Instalacion rapida para notebooks de NLP

Si solo necesitas ejecutar los notebooks `05_nlp_sentiment_models.ipynb` y `06_nlp_aspect_extraction.ipynb`:

```bash
# Paquetes mínimos requeridos para notebooks NLP/ML
uv add scikit-learn pandas numpy matplotlib seaborn joblib

# Verificar instalacion
uv run python -c "from sklearn.feature_extraction.text import TfidfVectorizer; print('sklearn OK')"
```

### Notebooks de NLP/ML

| Notebook | Descripcion | Output |
|----------|-------------|--------|
| `05_nlp_sentiment_models.ipynb` | Entrenamiento y comparacion de modelos de sentimiento | `models/sentiment/*.joblib` |
| `06_nlp_aspect_extraction.ipynb` | Extraccion de aspectos y sentimiento por aspecto | `models/aspects/*.joblib` |

**Ejecutar notebooks para entrenar modelos:**

```bash
# Abrir Jupyter Lab
uv run jupyter lab

# Navegar a notebooks/ y ejecutar:
# 1. 05_nlp_sentiment_models.ipynb (entrena modelos de sentimiento)
# 2. 06_nlp_aspect_extraction.ipynb (entrena modelos de aspectos)
```

**Modelos generados:**
- `models/sentiment/unified_svm_sentiment.joblib` - Modelo unificado (recomendado)
- `models/sentiment/yelp_svm_sentiment.joblib` - Especializado ingles
- `models/sentiment/es_logistic_sentiment.joblib` - Especializado espanol
- `models/sentiment/olist_svm_sentiment.joblib` - Especializado portugues
- `models/aspects/aspect_classifier_lr.joblib` - Clasificador multi-label de aspectos
- `models/aspects/aspect_taxonomy.json` - Taxonomia de aspectos multi-idioma

---

## Variables de Entorno

Ver [`.env.example`](.env.example) para la lista completa:

```
OPENAI_API_KEY=         # Requerido para LLM
LANGSMITH_API_KEY=      # Para trazabilidad
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=tfm-agents
```

---

## Dependencias Principales

- `langgraph` - Orquestación de agentes
- `langchain` / `langchain-openai` - Integración LLM
- `duckdb` - Warehouse analítico
- `polars` - Procesamiento de datos eficiente
- `pandas` / `numpy` - Análisis de datos y operaciones numéricas
- `pyarrow` - Soporte Parquet
- `pydantic` - Validación de esquemas
- `scikit-learn` - Machine Learning y métricas
- `vadersentiment` - Análisis de sentimiento (inglés)
- `matplotlib` / `seaborn` - Visualización
- `joblib` - Serialización de modelos

---

## Uso en LangGraph Studio

### Iniciar LangGraph Studio

```bash
cd tfm-agents

# Activar entorno y ejecutar
uv run langgraph dev
```

Esto abrira automaticamente el navegador en `http://127.0.0.1:2024` con LangGraph Studio.

### Grafos Disponibles

| Grafo | Uso | Estado |
|-------|-----|--------|
| **conversation** | Preguntas en lenguaje natural (PRINCIPAL) | Funcional |
| **nlp** | Construccion directa de silver/gold | Funcional |
| **prediction** | Modelo de prediccion ventas Olist | Fase 5 |
| **evaluation** | Evaluacion offline de metricas | Fase 6 |

### Seleccionar Grafo

En la barra superior de LangGraph Studio, hacer clic en el dropdown y seleccionar:
- `conversation` para preguntas en lenguaje natural

### Input para el Grafo `conversation`

En LangGraph Studio, expande "Input" y configura los campos:

| Campo | Valor | Descripcion |
|-------|-------|-------------|
| `user_query` | Tu pregunta aqui | Pregunta en lenguaje natural en espanol |
| `current_dataset` | `yelp`, `es`, u `olist` | Dataset a consultar (requerido) |

**Ejemplo de Input:**

```json
{
  "user_query": "Tu pregunta aquí",
  "current_dataset": "yelp"
}
```

**Valores para `current_dataset`:** `"yelp"`, `"es"`, `"olist"` o vacío para auto-detectar.

### Casos de Uso para Probar

#### Preguntas Básicas (Funcionan con Tool Binding)

| Pregunta | Dataset | Tool Invocada | Resultado |
|----------|---------|---------------|-----------|
| "Cual es la distribucion de ratings?" | yelp | `get_reviews_distribution` | Conteo por estrellas 1-5 |
| "Cual es la distribucion de ratings?" | olist | `get_reviews_distribution` | Conteo por score 1-5 |
| "Cual es la distribucion de ratings?" | es | `get_reviews_distribution` | Conteo por estrellas 1-5 |
| "Cuantas reseñas hay por mes?" | olist | `get_reviews_by_month` | Tendencia temporal |
| "Cuantas reseñas hay por mes?" | yelp | `get_reviews_by_month` | Tendencia temporal |
| "Cuales son las ventas por mes?" | olist | `get_sales_by_month` | Revenue mensual |
| "Cual es la evolucion de ordenes?" | olist | `get_sales_by_month` | Órdenes por mes |
| "Analiza las reseñas de 3 estrellas" | yelp | `get_ambiguous_reviews_analysis` | Reviews ambiguas |
| "Cuales son los usuarios mas influyentes?" | yelp | `get_user_stats` | Top reviewers |
| "Estadisticas de negocios" | yelp | `get_business_stats` | Métricas de negocios |
| "Ventas por categoria" | olist | `get_sales_by_category` | Top categorías |
| "Correlacion reviews y ventas" | olist | `get_reviews_sales_correlation` | Análisis correlación |

#### Guardrails (El LLM detecta automáticamente)

| Pregunta | Dataset | Comportamiento |
|----------|---------|----------------|
| "Cual es la tendencia por mes?" | es | Tool retorna error: "Dataset 'es' NO tiene fechas" |
| "Cuales son las ventas?" | yelp | El LLM sabe que `get_sales_by_month` es solo para olist |
| "hola" | - | Respuesta directa sin invocar tools |

### Ejemplo Paso a Paso en Studio

1. **Seleccionar grafo**: Click en dropdown arriba, elegir `conversation`
2. **Expandir Input**: Click en la seccion "Input" a la izquierda
3. **Configurar campos**:
   - En `User Query`: escribir `"Cual es la distribucion de ratings en Yelp?"`
   - En `Current Dataset`: escribir `"yelp"`
4. **Click Submit**
5. **Observar flujo**: Ver como pasa por `router` - `aggregator` - `synthesizer` - `qa`
6. **Ver resultado**: El nodo final muestra `insights_report` con el resumen

### Input para el Grafo `nlp`

Para construir features directamente sin preguntas:

```json
{
  "dataset": "yelp"
}
```

## Tests

### Tests Unitarios (pytest)

```bash
# Ejecutar todos los tests
uv run pytest

# Tests por modulo
uv run pytest tests/test_loaders.py -v      # Carga de datos bronze
uv run pytest tests/test_preprocess.py -v   # Construccion silver
uv run pytest tests/test_features.py -v     # Construccion gold
uv run pytest tests/test_sentiment.py -v    # Analisis de sentimiento
uv run pytest tests/test_aggregations.py -v # Agregaciones
uv run pytest tests/test_router.py -v       # Router y guardrails
uv run pytest tests/test_graph_integration.py -v  # Integracion grafos
```

### Verificacion de Estado

```bash
# Ver estado completo de todas las capas (bronze, silver, gold) con número de registros
uv run python scripts/data_status.py

# Ver solo una capa específica
uv run python scripts/data_status.py --layer silver
uv run python scripts/data_status.py --layer gold

# Ver con más detalles (nombres de columnas)
uv run python scripts/data_status.py --verbose

# Verificar archivos bronze existen
uv run python -c "from tfm.tools import check_bronze_files; print(check_bronze_files())"

# Verificar estado silver layer
uv run python scripts/build_silver.py --check

# Verificar estado gold layer
uv run python scripts/build_gold.py --check

# Verificar grafo compila
uv run python -c "from tfm.graphs.conversation_graph import conversation_graph; print('OK')"
```

---

## Escenarios capas de almacenamiento

Esta seccion documenta los diferentes escenarios para bronze, silver y gold.

### Escenario 1: Solo Bronze (Primera Ejecución)

**Preparación:**
```bash
# Eliminar silver y gold si existen (para prueba limpia)
rm data/silver/*.parquet
rm data/gold/*.parquet

# Verificar que bronze existe
uv run python scripts/build_silver.py --check
```

**Comportamiento esperado:**
- El LLM intenta invocar `get_reviews_distribution`
- La tool detecta que silver no existe y retorna error con sugerencia
- El LLM puede invocar `build_dataset_silver` para construir los datos
- Luego re-intenta la consulta original

**Input en LangSmith:**
```json
{
  "user_query": "Cual es la distribucion de ratings?",
  "current_dataset": "olist"
}
```

**Flujo del grafo:**
```
START -> route_query -> call_model -> tools (get_reviews_distribution) 
      -> call_model -> synthesizer -> qa -> respond -> END
```

### Escenario 2: Silver Existe, Gold No Existe

**Preparación:**
```bash
# Construir solo silver
uv run python scripts/build_silver.py --limit 1000 --overwrite

# Eliminar gold
rm data/gold/*.parquet
```

**Comportamiento esperado:**
- El LLM invoca directamente `get_sales_by_month` (para órdenes/ventas)
- La tool lee datos de silver y retorna resultados
- El Synthesizer genera el insight
- Gold solo se construye si se necesitan features NLP

**Input en LangSmith:**
```json
{
  "user_query": "Cual es la evolucion de ordenes por mes?",
  "current_dataset": "olist"
}
```

**Tool invocada:** `get_sales_by_month`

### Escenario 3: Silver y Gold Existen (Flujo Optimo)

**Preparacion:**
```bash
# Construir ambas capas
uv run python scripts/build_silver.py --limit 5000 --overwrite
uv run python scripts/build_gold.py --overwrite
```

**Comportamiento esperado:**
- Flujo directo sin construccion
- Lecturas rapidas desde Parquet
- Agregaciones se materializan en memoria

**Input en LangSmith:**
```json
{
  "user_query": "Cuantas reseñas hay por mes en 2021?",
  "current_dataset": "yelp"
}
```

### Escenario 4: Cargar Datos Completos (Produccion)

**Preparacion:**
```bash
uv run python scripts/build_silver.py --overwrite
uv run python scripts/build_gold.py --overwrite
```

### Escenario 5: Distribucion de Sentimiento NLP

**Preparacion (ejecutar notebooks primero):**
```bash
# Verificar modelos existen
ls models/sentiment/*.joblib

# Verificar via tool
uv run python -c "from tfm.tools.nlp_models import get_nlp_models_status; print(get_nlp_models_status.invoke({}))"
```

**Input:**
```json
{
  "user_query": "Cual es la distribucion de sentimiento?",
  "current_dataset": "olist"
}
```

**Respuesta esperada:**
- Total analizado: ~40.000 reseñas
- Distribucion: positive 63%, neutral 5%, negative 32%
- Sentimiento promedio: 0.32 (positivo)

**Tool invocada:** `get_sentiment_distribution`

### Escenario 6: Sentimiento de reseñas de 3 Estrellas

**Input:**
```json
{
  "user_query": "Las reseñas de 3 estrellas son mas positivas o negativas?",
  "current_dataset": "olist"
}
```

**Respuesta esperada:**
- Total reseñas de 3 estrellas: ~3,500
- Distribucion: positive 35%, neutral 19%, negative 46%
- Interpretacion: Tendencia NEGATIVA

**Tool invocada:** `get_ambiguous_reviews_sentiment`

### Escenario 7: Aspectos en reseñas Negativas

**Input:**
```json
{
  "user_query": "Cuales son los problemas mas mencionados en reseñas negativas?",
  "current_dataset": "yelp"
}
```

**Respuesta esperada:**
- Top aspectos: service (55%), product (37%), price (34%)
- Desglose de sentimiento por cada aspecto

**Tool invocada:** `get_aspect_distribution` con filtro sentiment=negative

---

## Casos de Uso para Probar

### Casos Basicos (Agregaciones Simples)

| Pregunta | Dataset | Agregacion | Descripcion |
|----------|---------|------------|-------------|
| "Cual es la distribucion de ratings?" | yelp | `reviews_by_stars` | Conteo por estrellas 1-5 |
| "Cuantas reseñas hay por mes?" | olist | `reviews_by_month` | Tendencia temporal |
| "Cuales son los usuarios mas influyentes?" | yelp | `user_stats` | Top usuarios por influence_score |
| "Cuales son las ventas por mes?" | olist | `olist_sales` | Revenue mensual |
| "Analiza las reseñas de 3 estrellas" | yelp | `ambiguous_reviews` | Reviews ambiguas |

### Casos Avanzados (Requieren Gold o Multiples Agregaciones)

| Pregunta | Dataset | Agregaciones | Descripcion |
|----------|---------|--------------|-------------|
| "Cual es el sentimiento promedio y como se distribuye?" | yelp | `reviews_by_stars` + gold | Usa features de sentimiento |
| "Compara el sentimiento de reseñas largas vs cortas" | yelp | `text_length` | Analisis por longitud |
| "Cuales son las categorias con mas ventas?" | olist | `olist_by_category` | Top categorias |
| "Como correlacionan reviews y ventas?" | olist | `olist_reviews_sales` | Correlacion |
| "Estadisticas de negocios en Yelp" | yelp | `business_stats` | Metricas de negocios |

### Casos de Guardrails (Deben Rechazarse)

| Pregunta | Dataset | Error Esperado |
|----------|---------|----------------|
| "Cual es la tendencia por mes?" | es | "Dataset ES no tiene fechas" |
| "Cuales son las ventas?" | yelp | "Solo Olist tiene datos de ventas" |
| "hola" | - | "Query muy corta" |
| "Predice las ventas del proximo mes" | olist | Fase 5 (no implementado) |

### Casos NLP sobre Datasets (Uso Principal)

| Pregunta | Dataset | Tool | Descripcion |
|----------|---------|------|-------------|
| "Cual es la distribucion de sentimiento?" | yelp | `get_sentiment_distribution` | % positivo/neutral/negativo |
| "Cuantas reseñas positivas hay?" | olist | `get_sentiment_distribution` | Conteo por sentimiento |
| "Cual es el sentimiento promedio?" | yelp | `get_sentiment_distribution` | Media de sentimiento |
| "Que porcentaje de reseñas son negativas?" | olist | `get_sentiment_distribution` | Distribucion |
| "Cual es el sentimiento de las reseñas de 3 estrellas?" | yelp | `get_ambiguous_reviews_sentiment` | Tendencia real |
| "Las reseñas ambiguas son mas positivas o negativas?" | olist | `get_ambiguous_reviews_sentiment` | Analisis ambiguas |
| "Cuales son los aspectos mas mencionados?" | yelp | `get_aspect_distribution` | Frecuencia de aspectos |
| "De que hablan las reseñas negativas?" | olist | `get_aspect_distribution` | Aspectos en negativas |
| "Que problemas mencionan las reseñas de 1 estrella?" | yelp | `get_aspect_distribution` | Aspectos por stars |

### Ejemplos de Inputs NLP

**Distribucion de sentimiento general:**
```json
{
  "user_query": "Cual es la distribucion de sentimiento?",
  "current_dataset": "olist"
}
```

**Sentimiento de reseñas de 3 estrellas:**
```json
{
  "user_query": "Las reseñas de 3 estrellas son mas positivas o negativas?",
  "current_dataset": "olist"
}
```

**Aspectos en reseñas negativas:**
```json
{
  "user_query": "Cuales son los problemas mas mencionados en reseñas negativas?",
  "current_dataset": "yelp"
}
```

**Analisis de reseña individual (uso secundario):**
```json
{
  "user_query": "Analiza esta reseña: 'Excelente producto, muy buena calidad'",
  "current_dataset": "es"
}
```

---

## Comandos de Preparacion de Datos

### Construir Silver Layer

```bash
# Todos los datasets (con limite para pruebas)
uv run python scripts/build_silver.py --limit 5000 --overwrite

# Dataset individual
uv run python scripts/build_silver.py --dataset yelp --limit 10000 --overwrite
uv run python scripts/build_silver.py --dataset es --limit 10000 --overwrite
uv run python scripts/build_silver.py --dataset olist --overwrite  # Olist no necesita limite

# Sin limite (datos completos)
uv run python scripts/build_silver.py --overwrite
```

### Construir Gold Layer (Features NLP)

```bash
# Todos los datasets
uv run python scripts/build_gold.py --overwrite

# Dataset individual
uv run python scripts/build_gold.py --dataset yelp --overwrite
```

### Limpiar Datos (Para Pruebas)

```bash
# Limpiar silver (forzar reconstruccion)
rm data/silver/yelp_reviews.parquet
rm data/silver/es_reviews.parquet
rm data/silver/olist_*.parquet

# Limpiar gold (forzar reconstruccion features)
rm data/gold/*.parquet

# Limpiar warehouse DuckDB
rm warehouse/tfm.duckdb
```

---

## Uso de DuckDB

### Estado Actual
DuckDB esta configurado pero actualmente las agregaciones usan **Polars directamente** sobre archivos Parquet. DuckDB se usa para:

1. **Registro de vistas**: Los archivos Parquet se registran como vistas en DuckDB
2. **Consultas SQL futuras**: Preparado para Fase 5/6 con consultas SQL complejas
3. **Warehouse centralizado**: Un unico punto de acceso a todos los datos

### Cuando se Usara DuckDB (Fase 5+)
- Consultas SQL complejas con JOINs entre tablas
- Agregaciones que requieren multiples tablas
- Busqueda full-text sobre reviews
- Materializacion de vistas para performance

### Verificar Estado DuckDB

```bash
# Verificar tablas registradas
uv run python -c "
from tfm.tools.storage import list_registered_tables
print(list_registered_tables())
"

# Registrar todas las tablas silver
uv run python -c "
from tfm.tools.storage import register_all_silver_tables
register_all_silver_tables()
"
```

---

## Cómo Agregar Nuevas Tools

El sistema usa **tool binding** lo que hace muy fácil agregar nuevas herramientas:

### 1. Crear la Tool en `src/tfm/tools/analysis_tools.py`

```python
from langchain_core.tools import tool

@tool
def mi_nueva_herramienta(dataset: Literal["yelp", "es", "olist"], parametro: int = 10) -> Dict[str, Any]:
    """
    Descripción clara de lo que hace esta herramienta.
    
    Usa esta herramienta cuando el usuario pregunte sobre:
    - Caso de uso 1
    - Caso de uso 2
    
    DISPONIBLE PARA: yelp, olist (lista los datasets soportados)
    
    Args:
        dataset: El dataset a analizar
        parametro: Descripción del parámetro
        
    Returns:
        Diccionario con resultados
    """
    # Implementación
    return {"resultado": "datos"}
```

### 2. Agregar a `get_all_tools()`

```python
def get_all_tools() -> List:
    return [
        # ... tools existentes ...
        mi_nueva_herramienta,  # Agregar aquí
    ]
```

### 3. El LLM la descubre automáticamente

No hay que modificar el Router ni el grafo. El LLM verá la nueva herramienta y podrá usarla basándose en su descripción.

---

## TODO: Mejoras Pendientes

### Chat con Lenguaje Natural (Fase 5+)
Actualmente el input requiere JSON estructurado:
```json
{"user_query": "...", "current_dataset": "yelp"}
```

**Objetivo futuro:** Permitir mensajes como:
> "Cual es la distribucion de ratings del dataset Yelp?"

El sistema debera:
1. Parsear el mensaje con NLP/LLM
2. Extraer `user_query` y `current_dataset` automaticamente
3. Detectar dataset mencionado ("Yelp", "espanol", "Olist", etc.)

### Otras Mejoras Pendientes
- [ ] Soporte para preguntas de seguimiento (memoria de conversacion)
- [ ] Graficos/visualizaciones en la respuesta
- [ ] Exportar resultados a CSV/Excel
- [ ] Filtros por fecha en lenguaje natural ("ultimos 6 meses")
- [ ] Comparaciones entre datasets
- [ ] Correlacion sentimiento-ventas (Olist)

---

## Verificacion de Tools NLP

### Verificar modelos cargados

```bash
uv run python -c "from tfm.tools.nlp_models import get_nlp_models_status; print(get_nlp_models_status.invoke({}))"
```

### Probar distribucion de sentimiento (dataset completo)

```bash
uv run python -c "from tfm.tools.nlp_models import get_sentiment_distribution; import json; print(json.dumps(get_sentiment_distribution.invoke({'dataset': 'olist'}), indent=2))"
```

### Probar sentimiento de reseñas ambiguas

```bash
uv run python -c "from tfm.tools.nlp_models import get_ambiguous_reviews_sentiment; import json; print(json.dumps(get_ambiguous_reviews_sentiment.invoke({'dataset': 'yelp'}), indent=2))"
```

### Probar distribucion de aspectos

```bash
uv run python -c "from tfm.tools.nlp_models import get_aspect_distribution; import json; print(json.dumps(get_aspect_distribution.invoke({'dataset': 'yelp', 'sentiment_filter': 'negative'}), indent=2))"
```

### Probar analisis de reseña individual

```bash
uv run python -c "from tfm.tools.nlp_models import analyze_sentiment; print(analyze_sentiment.invoke({'text': 'Excelente producto', 'model': 'unified'}))"
```

### Listar todas las tools disponibles

```bash
uv run python -c "from tfm.tools.analysis_tools import get_all_tools; [print(t.name) for t in get_all_tools()]"
```

---

## Troubleshooting

### Error: "No module named 'tfm'"

Ejecutar:
```bash
uv pip install -e .
```

### Error: Silver layer no existe

Ejecutar:
```bash
uv run python scripts/build_silver.py --limit 10000
```

### LangGraph Studio no conecta

1. Verificar que `.env` tenga `OPENAI_API_KEY`
2. Verificar que el grafo compile: `uv run python -c "from tfm.graphs.conversation_graph import conversation_graph; print('OK')"`

---

## Licencia

Proyecto académico - TFM Big Data / Data Science

---

## Referencias

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangSmith](https://smith.langchain.com/)
- [Yelp Dataset](https://www.yelp.com/dataset)
- [Olist Dataset](https://www.kaggle.com/olistbr/brazilian-ecommerce)
