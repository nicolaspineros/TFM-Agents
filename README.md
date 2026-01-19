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
- [x] Router con guardrails
- [x] NLP Worker con tools
- [x] Conversation graph con rutas condicionales
- [x] Insight Synthesizer con LLM
- [x] QA Evaluator con checks deterministicos

### Fase 5: Prediccion y ML Avanzado
- [ ] Features temporales (lag, rolling)
- [ ] Modelo de prediccion de ventas (Olist)
- [ ] Correlacion sentimiento - ventas
- [ ] Modelos NLP propios + clasificadores

### Fase 6: Evaluacion y Refinamiento
- [ ] Metricas ML (F1, MAE)
- [ ] Integracion LangSmith
- [ ] Evaluadores de faithfulness
- [ ] Optimizacion de prompts

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

#### Preguntas Basicas (Funcionan Ahora)

| Pregunta | Dataset | Que Hace |
|----------|---------|----------|
| "Cual es la distribucion de ratings?" | yelp | Cuenta reviews por estrellas (1-5) |
| "Cuantas reseñas hay por mes?" | olist | Tendencia temporal de reviews |
| "Cual es el sentimiento promedio?" | yelp | Score de sentimiento agregado |
| "Analiza las reseñas de 3 estrellas" | yelp | Estadisticas de reviews ambiguas |
| "Cuales son los usuarios mas influyentes?" | yelp | Top usuarios por influence_score |
| "Cuales son las ventas por mes?" | olist | Revenue mensual agregado |

#### Guardrails (Queries que deben rechazarse)

| Pregunta | Dataset | Error Esperado |
|----------|---------|----------------|
| "Cual es la tendencia por mes?" | es | "Dataset ES no tiene fechas" |
| "Cuales son las ventas?" | yelp | "Solo Olist tiene datos de ventas" |
| "hola" | - | "Query muy corta" |

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

---

## Quickstart

```bash
# init
uv init

#  Añadir dependencias
uv add langgraph langchain langchain-openai
uv add "langgraph-cli[inmem]" --dev
uv add polars pyarrow duckdb pydantic pydantic-settings
uv add pytest jupyterlab --dev

# Instalar el paquete local (IMPORTANTE)
uv pip install -e .

# Verificar instalacion
uv run python -c "from tfm.config.settings import get_settings; print('OK')"

# Construir silver layer (primera vez)
uv run python scripts/build_silver.py --limit 10000

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

## Escenarios de Prueba

Esta seccion documenta los diferentes escenarios de prueba segun el estado de los datos.

### Escenario 1: Solo Bronze (Primera Ejecucion)

**Preparacion:**
```bash
# Eliminar silver y gold si existen (para prueba limpia)
rm data/silver/*.parquet
rm data/gold/*.parquet

# Verificar que bronze existe
uv run python scripts/build_silver.py --check
```

**Comportamiento esperado:**
- El grafo detecta que silver no existe
- Ruta: `router -> nlp_worker -> aggregator -> synthesizer -> qa -> respond`
- `nlp_worker` construye silver layer automaticamente
- Luego ejecuta agregaciones sobre silver

**Input en LangSmith:**
```json
{
  "user_query": "Cual es la distribucion de ratings?",
  "current_dataset": "yelp"
}
```

### Escenario 2: Silver Existe, Gold No Existe

**Preparacion:**
```bash
# Construir solo silver
uv run python scripts/build_silver.py --limit 1000 --overwrite

# Eliminar gold
rm data/gold/*.parquet
```

**Comportamiento esperado:**
- Ruta normal: `router -> aggregator -> synthesizer -> qa -> respond`
- Las agregaciones se ejecutan sobre silver
- Gold se construye solo si `needs_nlp_features=true` en el plan

**Input en LangSmith:**
```json
{
  "user_query": "Cual es el sentimiento promedio de las reseñas?",
  "current_dataset": "yelp"
}
```

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

### Casos NLP (Cuando Gold Existe)

| Pregunta | Dataset | Requiere | Descripcion |
|----------|---------|----------|-------------|
| "Cual es el sentimiento promedio?" | yelp | gold | Score promedio de sentimiento |
| "Cuantas reseñas positivas hay?" | yelp | gold | Conteo por label |
| "Que porcentaje son negativas?" | olist | gold | Distribucion sentimiento |

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
