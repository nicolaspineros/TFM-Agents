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
├── data/
│   ├── bronze/          # Datos crudos (JSONL, CSV)
│   │   ├── yelp/
│   │   ├── rese_esp/
│   │   └── olist_ecommerce/
│   ├── silver/          # Datos limpios y normalizados (Parquet)
│   └── gold/            # Features, agregados, métricas (Parquet + JSON)
├── warehouse/
│   └── tfm.duckdb       # Base de datos analítica
├── runs/                # Checkpoints de grafos (SQLite)
├── docs/
│   └── data_contracts.md
├── notebooks/
│   ├── 01_eda_yelp.ipynb
│   ├── 02_eda_es.ipynb
│   └── 03_eda_olist.ipynb
├── src/
│   └── tfm/
│       ├── config/      # Configuración y settings
│       ├── schemas/     # Pydantic models (State, Request, Outputs)
│       ├── tools/       # Herramientas deterministas
│       ├── agents/      # Agentes LLM
│       └── graphs/      # Grafos LangGraph
├── tests/
├── langgraph.json
├── pyproject.toml
└── README.md
```

---

## Agentes

| Agente | Responsabilidad |
|--------|-----------------|
| **Router / Query Planner** | Interpreta pregunta, inspecciona schema, aplica guardrails, decide si construir features/agregados |
| **NLP/ML Worker** | Ejecuta tools de NLP (sentimiento, aspectos, clasificación) y ML |
| **Insight Synthesizer** | Consume tablas pequeñas, redacta `InsightsReport` estructurado |
| **QA/Evaluator** | Checks deterministas (schema, faithfulness), evaluaciones ML (F1/MAE) |

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
- Tabla de traducción de categorías (pt→en)

Ver detalles en: [`docs/data_contracts.md`](docs/data_contracts.md)

---

## Caso Especial: Reviews Ambiguas (stars == 3)

Las reseñas con `stars == 3` son consideradas **ambiguas**. El sistema:
- Marca estas reseñas con un flag `is_ambiguous` en silver/gold
- Permite análisis específico de este segmento
- Usa lógica especial en clasificación de sentimiento

---

## Fases de Desarrollo

### Fase 1: EDA y Prototipos (Notebooks)
- [ ] EDA Yelp: distribución de stars, longitud de texto, fechas
- [ ] EDA ES: exploración sin fecha, categorías
- [ ] EDA Olist: relación ventas-reviews, temporalidad
- [ ] Prototipos de NLP básico en notebooks

### Fase 2: Tools Deterministas
- [ ] Loaders y storage (bronze → silver)
- [ ] Profiling de datasets
- [ ] Baseline sentimiento (rule-based + stars==3)
- [ ] Agregaciones SQL con DuckDB

### Fase 3: Agentes y Grafos
- [ ] Router con guardrails
- [ ] NLP Worker con tools
- [ ] Insight Synthesizer
- [ ] QA básico

### Fase 4: Evaluación y Trazabilidad
- [ ] Métricas ML (F1, MAE)
- [ ] Integración LangSmith
- [ ] Evaluadores de faithfulness

### Fase 5: Refinamiento
- [ ] Modelos NLP avanzados (reemplazar baselines)
- [ ] Modelo de predicción de ventas
- [ ] Optimización de prompts

---

## Quickstart

```bash
# init
uv init

#  Añadir dependencias
uv add langgraph langchain langchain-openai
uv add "langgraph-cli[inem]" --dev
uv add polars pyarrow duckdb pydantic pydantic-settings
uv add pytest jupyterlab --dev

# Instalar el paquete local (IMPORTANTE)
uv pip install -e .

# Verificar instalación
uv run python -c "from tfm.config.settings import get_settings; print('OK')"

# Ejecutar notebooks para EDA
uv run jupyter lab

# Ejecutar LangGraph Studio
uv run langgraph dev
```

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
- `pyarrow` - Soporte Parquet
- `pydantic` - Validación de esquemas

---

## Tests

```bash
# Ejecutar todos los tests
uv run pytest

# Tests específicos
uv run pytest tests/test_loaders.py -v
uv run pytest tests/test_sentiment.py -v
uv run python -c "from tfm.tools import check_bronze_files; print(check_bronze_files())"
```

---

## Licencia

Proyecto académico - TFM Big Data / Data Science

---

## Referencias

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangSmith](https://smith.langchain.com/)
- [Yelp Dataset](https://www.yelp.com/dataset)
- [Olist Dataset](https://www.kaggle.com/olistbr/brazilian-ecommerce)
