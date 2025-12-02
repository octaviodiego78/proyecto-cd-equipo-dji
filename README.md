# Servicio de Predicción del Precio del Oro

Un servicio de machine learning listo para producción que predice los precios del oro del día siguiente utilizando datos históricos de oro y S&P 500. El sistema aprovecha redes neuronales de TensorFlow (MLP/CNN/LSTM), MLflow para gestión de modelos, Prefect para orquestación y Docker para containerización.

## Arquitectura

```
┌─────────────────────────────────────────────────────────────────┐
│                  Pipeline de Entrenamiento (Prefect)            │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐      │
│  │  Cargar  │──▶│Ingeniería│──▶│ Entrenar │──▶│ Registrar│      │
│  │  Datos   │   │  Features│   │  Modelos │   │  Champion│      │ 
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘      │
│         │             │                               │         │
│         ▼             ▼                               ▼         │
│   data/raw/    data/processed/              Registro MLflow     │
│   - gold_data.csv  - scaler.pkl           (Databricks)          │
│   - sp500.csv      - feature_columns.json                       │
│                    - model_metadata.json                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Servicio de Predicción (Docker)                │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              Backend FastAPI (Puerto 8000)                 │ │
│  │  ┌──────────┐   ┌──────────┐    ┌──────────┐   ┌────────┐  │ │ 
│  │  │ Obtener  │──▶│ Ingeniería│──▶│ Escalar  │──▶│Predecir│  │ │
│  │  │Yahoo     │   │  Features │   │ Features │   │(Modelo)│  │ │
│  │  │Finance   │   │           │   │          │   │        │  │ │
│  │  └──────────┘   └──────────┘    └──────────┘   └────────┘  │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              │                                  │
│                              ▼                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │          Frontend Streamlit (Puerto 8501)                  │ │
│  │              UI Predicción Precio del Oro                  │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## 📋 Características

- **Integración de Datos en Vivo**: Obtiene precios en tiempo real de oro (GC=F) y S&P 500 (^GSPC) desde Yahoo Finance
- **Modelos ML Avanzados**: Soporta arquitecturas MLP, CNN y LSTM con optimización de hiperparámetros
- **Integración MLflow**: Versionado, seguimiento y registro de modelos con Databricks Unity Catalog
- **Orquestación Prefect**: Pipeline de entrenamiento automatizado con tareas y flujos
- **API Lista para Producción**: Backend FastAPI con health checks y manejo apropiado de errores
- **Interfaz Amigable**: Frontend Streamlit para predicciones fáciles
- **Despliegue Containerizado**: Docker y docker-compose para ambientes consistentes

##  Inicio Rápido

### Prerequisitos

- Python 3.11+
- Docker Desktop instalado y ejecutándose
- Cuenta de Databricks con acceso a MLflow
- Variables de entorno configuradas (ver `src/env.example`)

### 1. Configuración del Entorno

Crea un archivo `.env` en la raíz del proyecto:

```bash
# Copia el archivo de ejemplo
cp src/env.example .env

# Edita con tus credenciales
nano .env
```

Agrega tus credenciales de Databricks:
```bash
DATABRICKS_HOST=https://tu-workspace.databricks.com
DATABRICKS_TOKEN=tu-token-databricks
```

### 2. Pipeline de Entrenamiento

#### Ejecutar Flujo de Entrenamiento Prefect

```bash
# Instalar dependencias
pip install -r src/pipelines/requirements.txt

# Ejecutar el pipeline de entrenamiento
python src/pipelines/train_pipeline.py
```

**Tareas del Pipeline:**
1. Cargar y preparar datos desde archivos CSV
2. Ingeniería de features (lags, promedios móviles, volatilidad)
3. Entrenar modelos base (MLP, CNN, LSTM)
4. Optimización de hiperparámetros con Hyperopt
5. Seleccionar mejor modelo basado en MAPE
6. Registrar modelo campeón en MLflow
7. Guardar artefactos (scaler, columnas de features, metadata)

**Salidas:**
- `data/processed/scaler.pkl` - StandardScaler ajustado
- `data/processed/feature_columns.json` - Nombres de columnas de features
- `data/processed/model_metadata.json` - Tipo de modelo y nombre
- Modelo registrado en Databricks MLflow Registry con alias "champion"

#### Logs del Flujo Prefect

El pipeline proporciona logging detallado para cada tarea:
- Estadísticas de carga de datos
- Progreso de ingeniería de features
- Métricas de entrenamiento (MAPE, RMSE, MAE, R²)
- Detalles de registro del modelo

### 3. Ejecutar el Servicio

#### Opción A: Usando Script de Inicio (Recomendado)

```bash
# Navegar al directorio src
cd src/

# Iniciar ambos servicios
./start.sh
```

Esto hará:
- Construir imágenes Docker
- Iniciar contenedores backend y frontend
- Ejecutar health checks
- Mostrar URLs de los servicios

#### Opción B: Docker Compose Manualmente

```bash
# Navegar al directorio src
cd src/

# Construir e iniciar servicios en modo desacoplado
docker compose up --build -d

# Ver logs
docker compose logs -f
```


### 4. Acceder a los Servicios

Una vez que los contenedores estén ejecutándose (toma ~30-60 segundos para que el backend cargue el modelo):

- **Frontend UI**: http://localhost:8501
- **Documentación API**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

### 5. Verificar que los Servicios Estén Funcionando

```bash
# Verificar estado de contenedores
docker compose ps

# Verificar salud del backend (esperar hasta que el estado sea "healthy")
curl http://localhost:8000/health
```

Respuesta esperada cuando esté listo:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "scaler_loaded": true,
  "feature_cols_loaded": true
}
```

### 6. Hacer Predicciones

#### Via UI Streamlit (Más Fácil)
1. Abrir http://localhost:8501 en tu navegador
2. Hacer clic en **"Predict Tomorrow's Gold Price"**
3. Esperar 15-30 segundos (obtiene datos en vivo de Yahoo Finance)
4. Ver resultados de predicción con:
   - Precio del oro predicho
   - Fecha de predicción
   - Tipo de modelo y detalles

#### Via API (Para Integración)

**Probar Predicción:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"predict_tomorrow": true}'
```

Respuesta esperada:
```json
{
  "prediction": 2650.25,
  "predicted_date": "2024-12-03",
  "today_date": "2024-12-02",
  "model_name": "equipo_dji_gold_prediction_model",
  "model_type": "MLP"
}
```

**Pruebas Interactivas de API:**
- Abrir http://localhost:8000/docs
- Probar endpoints directamente en el navegador

## Configuración Docker

### Vista General de la Arquitectura

```
┌─────────────────────────────────────────────────┐
│           Tu Computadora (Localhost)            │
│                                                 │
│  ┌──────────────────┐      ┌─────────────────┐  │
│  │    Frontend      │─────▶│    Backend      │  │
│  │   (Streamlit)    │ HTTP │    (FastAPI)    │  │
│  │  Contenedor      │      │   Contenedor    │  │
│  │  Puerto: 8501    │      │   Puerto: 8000  │  │
│  └────────┬─────────┘      └────────┬────────┘  │
│           │                         │           │
│           └────────┬────────────────┘           │
│                    │                            │
│        gold-prediction-network                  │
│               (Bridge)                          │
│                    │                            │
│           ┌────────▼────────┐                   │
│           │  Montajes Vol.  │                   │
│           │  ../data/       │                   │
│           │  ../.env        │                   │
│           └─────────────────┘                   │
└─────────────────────────────────────────────────┘
```

### Configuración Docker Compose (`src/docker-compose.yaml`)

**Servicio Backend:**
- **Contenedor**: `gold-prediction-backend`
- **Imagen**: Python 3.11-slim con TensorFlow, FastAPI, MLflow
- **Puertos**: 8000:8000
- **Volúmenes**:
  - `../data/processed:/app/data/processed:ro` (artefactos del modelo, solo lectura)
  - `../.env:/app/.env:ro` (credenciales, solo lectura)
- **Health Check**: Curl a `/health` cada 30s, 40s de período de inicio
- **Variables de Entorno**:
  - `SCALER_PATH=/app/data/processed/scaler.pkl`
  - `FEATURE_COLS_PATH=/app/data/processed/feature_columns.json`
  - `MODEL_METADATA_PATH=/app/data/processed/model_metadata.json`

**Servicio Frontend:**
- **Contenedor**: `gold-prediction-frontend`
- **Imagen**: Python 3.11-slim con Streamlit
- **Puertos**: 8501:8501
- **Depende De**: Backend (espera estado healthy antes de iniciar)
- **Entorno**: `API_URL=http://backend:8000`
- **Red**: Comunica con backend via red bridge de Docker

**Red:**
- **Nombre**: `gold-prediction-network`
- **Tipo**: Bridge (permite comunicación contenedor-a-contenedor)
- **DNS**: Frontend resuelve hostname `backend` a IP del contenedor backend

### Backend Dockerfile (`src/backend/Dockerfile`)

```dockerfile
FROM python:3.11-slim
WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y build-essential curl

# Instalar paquetes Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código backend
COPY *.py ./

# Exponer puerto y configurar health check
EXPOSE 8000
HEALTHCHECK CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Características Clave:**
- Instala TensorFlow, FastAPI, MLflow, yfinance
- Carga artefactos del modelo desde montajes de volumen (no incorporados en la imagen)
- Health checks automáticos cada 30s
- CORS habilitado para comunicación con frontend

### Frontend Dockerfile (`src/frontend/Dockerfile`)

```dockerfile
FROM python:3.11-slim
WORKDIR /app

# Instalar Streamlit y dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar app frontend
COPY app.py .

EXPOSE 8501
ENV API_URL=http://backend:8000

CMD ["streamlit", "run", "app.py", "--server.port=8501", 
     "--server.address=0.0.0.0", "--server.headless=true"]
```

**Características Clave:**
- Aplicación Streamlit ligera
- Conecta al backend via red Docker
- Variable de entorno para configuración de URL API

## Endpoints de la API

### GET /health

Endpoint de health check para verificar el estado del servicio.

**Respuesta:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "scaler_loaded": true,
  "feature_cols_loaded": true
}
```

### POST /predict

Predice el precio del oro de mañana usando datos en vivo.

**Cuerpo de la Solicitud:**
```json
{
  "predict_tomorrow": true
}
```

**Respuesta:**
```json
{
  "prediction": 1950.75,
  "predicted_date": "2024-12-03",
  "today_date": "2024-12-02",
  "model_name": "workspace.default.equipo_dji_gold_prediction_model",
  "model_type": "MLP"
}
```

**Flujo de Predicción:**
1. Obtener últimos 30 días de precios de oro y S&P 500 desde Yahoo Finance
2. Fusionar y ordenar datos por fecha
3. Aplicar ingeniería de features (igual que en entrenamiento):
   - Features de lag (1-2 días) para oro y S&P 500
   - Promedios móviles de 5 días
   - Volatilidad de 5 días (oro)
   - Retornos S&P 500 (lag 1)
4. Extraer features para la fecha más reciente
5. Escalar features usando `StandardScaler` guardado
6. Remodelar basado en tipo de modelo (MLP, CNN, LSTM)
7. Predecir precio de mañana
8. Retornar predicción con metadata

##  Despliegue en HuggingFace Spaces

Este proyecto se despliega como **dos Spaces separados** en HuggingFace:
1. **Backend Space**: API FastAPI en puerto 7860
2. **Frontend Space**: UI Streamlit en puerto 7860 (conecta al backend)

### Arquitectura de Despliegue HuggingFace

```
┌────────────────────────────────────────────────────┐
│              HuggingFace Spaces                    │
│                                                    │
│  ┌─────────────────────┐   ┌──────────────────┐    │
│  │  Frontend Space     │──▶│  Backend Space   │    │
│  │  (Streamlit)        │   │  (FastAPI)       │    │
│  │  Puerto: 7860       │   │  Puerto: 7860    │    │
│  │                     │   │                  │    │
│  │  - app.py           │   │  - api.py        │    │
│  │  - requirements.txt │   │  - model_utils.py│    │
│  │  - Dockerfile       │   │  - data_fetcher.py│   │
│  │                     │   │  - preprocessing.py│  │
│  │                     │   │  - data/         │    │
│  │                     │   │  - requirements  │    │
│  │                     │   │  - Dockerfile    │    │
│  └─────────────────────┘   └──────────────────┘    │
│           │                         │              │
│           ▼                         ▼              │
│  your-frontend.hf.space    your-backend.hf.space   │
└────────────────────────────────────────────────────┘
```

### Paso 1: Crear Backend Space

#### 1.1 Crear Space
1. Ir a https://huggingface.co/spaces
2. Hacer clic en "Create new Space"
3. Nombre: `gold-predictions-backend`
4. SDK: **Docker**
5. Hardware: CPU Basic (mínimo 4GB RAM recomendado)

#### 1.2 Estructura del Backend Space

```
gold-predictions-backend/
├── README.md              # Descripción del backend
├── Dockerfile             # Contenedor Docker
├── requirements.txt       # Dependencias Python
├── api.py                # Aplicación FastAPI
├── preprocessing.py      # Utilidades de preprocesamiento
├── model_utils.py        # Carga de modelo MLflow
├── data_fetcher.py       # Obtención de datos Yahoo Finance
└── data/
    └── processed/
        ├── scaler.pkl
        ├── feature_columns.json
        └── model_metadata.json
```

#### 1.3 Dockerfile Backend

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias del sistema y uv
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | sh

# Agregar uv al PATH
ENV PATH="/root/.local/bin:$PATH"

# Copiar requirements
COPY requirements.txt .

# Instalar dependencias Python con uv
RUN uv pip install --system --no-cache -r requirements.txt

# Copiar código backend
COPY *.py ./

# Copiar datos procesados
COPY data/processed/ /app/data/processed/

# Configurar variables de entorno
ENV PYTHONPATH=/app

# Exponer puerto FastAPI (HuggingFace usa 7860)
EXPOSE 7860

# Iniciar FastAPI en puerto 7860
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
```


### Paso 2: Crear Frontend Space

#### 2.1 Crear Space
1. Ir a https://huggingface.co/spaces
2. Hacer clic en "Create new Space"
3. Nombre: `gold-predictions-frontend`
4. SDK: **Docker**
5. Hardware: CPU Basic

#### 2.2 Estructura del Frontend Space

```
gold-predictions-frontend/
├── README.md              # Descripción del frontend
├── Dockerfile             # Contenedor Docker
├── requirements.txt       # Dependencias Python
├── app.py                # Aplicación Streamlit
└── .streamlit/           # (Opcional) Configuración Streamlit
    └── config.toml
```

#### 2.3 Dockerfile Frontend

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements
COPY requirements.txt .

# Instalar dependencias Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar app
COPY app.py .
COPY .streamlit/ .streamlit/

# Configurar variables de entorno
ENV PYTHONPATH=/app

# Exponer puerto Streamlit (HuggingFace usa 7860)
EXPOSE 7860

# Iniciar Streamlit en puerto 7860
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true"]
```

#### 2.4 Configurar URL del Backend

En `app.py`, asegúrate de que la URL del backend apunte a tu Backend Space:

```python
# API URL - Apunta al Backend Space desplegado
API_URL = os.getenv("API_URL", "https://TU_USUARIO-gold-predictions-backend.hf.space")
```



Una vez desplegados ambos Spaces:

- **Frontend UI**: `https://huggingface.co/spaces/TU_USUARIO/gold-predictions-frontend`
- **Backend API**: `https://huggingface.co/spaces/TU_USUARIO/gold-predictions-backend`
- **API Docs**: `https://TU_USUARIO-gold-predictions-backend.hf.space/docs`


```
proyecto-cd-equipo-dji/
├── src/
│   ├── backend/
│   │   ├── api.py                  # Aplicación FastAPI con CORS
│   │   ├── preprocessing.py        # Utilidades de ingeniería de features
│   │   ├── model_utils.py          # Carga de modelo MLflow
│   │   ├── data_fetcher.py         # Integración Yahoo Finance
│   │   ├── requirements.txt        # Dependencias backend
│   │   ├── Dockerfile              # Definición contenedor backend
│   │   └── .dockerignore           # Optimización de build
│   ├── frontend/
│   │   ├── app.py                  # Aplicación Streamlit
│   │   ├── requirements.txt        # Dependencias frontend
│   │   ├── Dockerfile              # Definición contenedor frontend
│   │   └── .dockerignore           # Optimización de build
│   ├── pipelines/
│   │   └── train_pipeline.py       # Flujo de entrenamiento Prefect
│   ├── docker-compose.yaml         # Orquestación multi-contenedor
│   ├── start.sh                    # Script de inicio rápido (ejecutable)
│   ├── env.example                 # Template variables de entorno
│   └── README.md                   # Guía de despliegue Docker
├── data/
│   ├── raw/
│   │   ├── gold_data.csv           # Precios históricos de oro
│   │   └── sp500.csv               # Datos históricos S&P 500
│   └── processed/
│       ├── scaler.pkl              # StandardScaler ajustado
│       ├── feature_columns.json    # Nombres de features
│       └── model_metadata.json     # Tipo y nombre de modelo
├── notebooks/
│   ├── 01_eda_inicial.ipynb        # Análisis exploratorio de datos
│   └── 02_data_wrangling.ipynb     # Preparación de datos
├── informe_escrito/
│   └── 00_informe_final.ipynb      # Informe final
├── huggingface/
│   ├── gold-predictions-backend/   # Backend para HuggingFace Space
│   │   ├── Dockerfile
│   │   ├── api.py
│   │   ├── requirements.txt
│   │   ├── preprocessing.py
│   │   ├── model_utils.py
│   │   ├── data_fetcher.py
│   │   └── data/processed/
│   └── gold-predictions-frontend/  # Frontend para HuggingFace Space
│       ├── Dockerfile
│       ├── app.py
│       └── requirements.txt
├── .env                            # Variables de entorno (no en git)
├── .gitignore                      # Reglas de ignore de Git
└── README.md                       # Este archivo (documentación principal)
```

##  Dependencias

### Backend
- `fastapi>=0.109.0` - Framework web moderno
- `uvicorn>=0.27.0` - Servidor ASGI
- `tensorflow>=2.15.0` - Modelos de redes neuronales
- `mlflow>=2.10.0` - Seguimiento y registro de modelos
- `yfinance>=0.2.35` - Datos de Yahoo Finance
- `pandas>=2.1.0` - Manipulación de datos
- `scikit-learn>=1.3.0` - Preprocesamiento y métricas

### Frontend
- `streamlit>=1.30.0` - Aplicación web interactiva
- `requests>=2.31.0` - Cliente HTTP

### Pipeline de Entrenamiento
- `prefect>=3.0.0` - Orquestación de flujos de trabajo
- `hyperopt>=0.2.7` - Optimización de hiperparámetros
- Todas las dependencias del backend
