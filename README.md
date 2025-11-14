# Proyecto Final — MLOps con CRISP–DM

**Universidad del Valle de Guatemala**
**Facultad de Ingeniería — Departamento de Ciencias de la Computación**
**Machine Learning Operations (MLOps)**

## 1. Integrantes

* Arturo Argueta — 21527
* Diego Leiva — 21752
* Gustavo González — 21438
* Marta Ramírez — 21342
* Pablo Orellana — 21970

# 2. Descripción del Proyecto

Este proyecto implementa un **pipeline completo de MLOps** para el caso de **Churn Prediction** en Telecomunicaciones, siguiendo la metodología **CRISP–DM**.
Incluye:

* Procesamiento de datos
* Exploración de datos (EDA)
* Entrenamiento y tuning de 4 modelos (Logistic Regression, Random Forest, XGBoost, LightGBM)
* Tracking de experimentos con MLflow
* Logging de métricas, parámetros y artefactos
* Selección automática del mejor modelo
* **Registro de modelos en MLflow Model Registry**
* Evaluación final con métricas de negocio

Todo el flujo está integrado mediante módulos Python en `src/modules` y un notebook principal en `src/notebooks`.

# 3. Requisitos

### **Python 3.10+**

### Librerías principales:

Se instalan desde `requirements.txt`:

```
pandas
scikit-learn
scipy
mlflow
matplotlib
xgboost
lightgbm
seaborn
```

# 4. Instalación del entorno

### 4.1 Crear entorno virtual (recomendado)

#### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### 4.2 Instalar dependencias

Dentro del entorno activado:

```bash
pip install -r requirements.txt
```

# 5. Estructura del proyecto

```
src/
 ├── modules/
 │    ├── modelBaselineMlflow.py
 │    ├── modelEvaluation.py
 │    ├── dataGenerator.py
 │    ├── dataPreparation.py
 ├── notebooks/
 │    ├── trainingEvaluation.ipynb
 │    ├── dataExploration.ipynb
data/
 ├── processed/
 │    ├── train.csv
 │    ├── val.csv
 │    ├── test.csv
 ├── telconnect_churn_synth.csv
mlruns/              ← generado automáticamente
mlflow.db            ← backend del MLflow registry
requirements.txt
README.md
```

# 6. Cómo levantar el MLflow Tracking Server (con Model Registry)

Para habilitar tracking + artifacts + Model Registry, ejecutar:

```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 127.0.0.1 \
  --port 5000
```

Esto inicia un servidor local con:

* MLflow UI
* Experiment Tracking
* Artifact Store
* **Model Registry**

Abrir en el navegador:

```
http://127.0.0.1:5000
```

# 7. Cómo ejecutar el pipeline de entrenamiento y evaluación

Todo el flujo se ejecuta desde el notebook:

```
src/notebooks/training_notebook.ipynb
```

Este notebook:

1. Carga los datos procesados
2. Llama a `modelBaselineMlflow.py` para entrenar 4 modelos + tuning
3. Selecciona el mejor modelo automáticamente por métrica `val_f1`
4. Lo registra como nueva versión en el **Model Registry**
5. Corre el módulo `modelEvaluation.py` para evaluar el modelo seleccionado
6. Genera artefactos:

   * Matriz de confusión
   * Curva ROC
   * Curva Precision–Recall
   * Métricas de test
   * Métricas de impacto de negocio

Todos los runs quedan registrados en MLflow.
