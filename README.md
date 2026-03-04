# Trabajo de ML Challenge - Predicting Loan Payback

## Contexto del Proyecto
Este proyecto fue desarrollado como trabajo final para el módulo de Aprendizaje Automático del Máster en Big Data Analytics en la Universidad Europea de Valencia. Demuestra un pipeline completo de Machine Learning aplicado a un problema real de evaluación de riesgo crediticio bancario, desde la exploración de datos hasta la generación de predicciones para un challenge de Kaggle.

El proyecto aborda el problema de clasificación binaria supervisada para predecir si un cliente bancario pagará o no su préstamo, utilizando información financiera y sociodemográfica de solicitantes de crédito.

| Campo | Detalle |
|---|---|
| **Asignatura** | Aprendizaje Automático |
| **Institución** | Universidad Europea de Valencia |
| **Profesor** | Jorge Fresco |
| **Fecha** | Marzo 2026 |
| **Autores** | Camilo Velasquez, Fernando Pirrongelli, Juan Albornoz |

---

## 🎯 Objetivos del Trabajo

- Diseñar, implementar y evaluar una solución completa de Machine Learning, desde la exploración de los datos hasta la evaluación del modelo final.
- Seleccionar y justificar la elección de modelos de ML supervisados para clasificación binaria.
- Demostrar capacidad de análisis crítico, experimentación, validación y comunicación técnica.
- Generar predicciones de probabilidad de pago para el dataset de test del challenge, evaluadas mediante AUC-ROC.

---

## 📊 Dataset

El dataset corresponde a información financiera y sociodemográfica de solicitantes de crédito, simulando un escenario real de evaluación de riesgo crediticio.

| Característica | Detalle |
|---|---|
| **Registros totales** | ~800,000 |
| **Columnas originales** | 13 |
| **Variable objetivo** | `loan_paid_back` (1: Pagado, 0: No pagado) |
| **Tipo de problema** | Clasificación binaria supervisada |
| **Métrica principal** | AUC-ROC |

**Variables del dataset:**

| Variable | Tipo | Descripción |
|---|---|---|
| `id` | Numérica | Identificador único del cliente |
| `annual_income` | Numérica | Ingreso anual |
| `debt_to_income_ratio` | Numérica | Ratio deuda / ingreso |
| `credit_score` | Numérica | Puntuación crediticia |
| `loan_amount` | Numérica | Monto del préstamo |
| `interest_rate` | Numérica | Tasa de interés |
| `gender` | Categórica | Género |
| `marital_status` | Categórica | Estado civil |
| `education_level` | Categórica | Nivel educativo |
| `employment_status` | Categórica | Situación laboral |
| `loan_purpose` | Categórica | Propósito del préstamo |
| `grade_subgrade` | Categórica | Calificación crediticia |

---

## 🛠️ Pipeline del Proyecto

**1. Análisis y comprensión del problema**
Identificación del tipo de problema, suposiciones iniciales y riesgos como el desbalance de clases entre pagadores y morosos.

**2. Exploración y análisis de datos (EDA)**
Análisis de distribuciones, valores nulos, duplicados y matriz de correlación completa con dummies incluidos.

**3. Preprocesamiento**
- Eliminación de duplicados
- One-Hot Encoding (OHE) sobre 6 variables categóricas → de 13 a 62 columnas
- Reducción de variables por correlación con el target (umbral 0.05) → 6 variables finales
- Detección y eliminación de outliers mediante método IQR
- Train/Val split 70/30 (`random_state=0`)
- StandardScaler aplicado sobre `x_train` y transformado sobre `x_val`

**4. Selección y entrenamiento de modelos**
Tres modelos entrenados representando distintas familias algorítmicas: lineal, ensemble de bagging y ensemble de boosting.

**5. Optimización**
- LR: `LogisticRegressionCV` enfocado en parámetro de regularización C
- LGBM: `RandomizedSearchCV` sobre n_estimators, learning_rate, num_leaves y max_depth
- RF: `RandomizedSearchCV` sobre n_estimators, max_depth, min_samples_leaf y max_features

**6. Evaluación**
Diagnóstico de overfitting/underfitting mediante gráficas comparativas Train vs Val, matrices de confusión y curvas ROC para los tres modelos.

**7. Conclusiones y generación de submission**

---

## 🤖 Modelos

### Logistic Regression
Seleccionado por su alta interpretabilidad y cumplimiento regulatorio bancario (Basilea III). Sus coeficientes β permiten explicar exactamente por qué se aprueba o rechaza un crédito, requisito fundamental en auditorías financieras.

```python
LogisticRegression(max_iter=1000)
# Optimizado con LogisticRegressionCV
```

### LightGBM
Seleccionado por su arquitectura de Gradient Boosting secuencial con corrección de errores, manejo nativo de variables categóricas, algoritmo GOSS optimizado para datasets grandes y regularización L1/L2 incorporada.

```python
LGBMClassifier(random_state=42)
# Optimizado con RandomizedSearchCV
```

### Random Forest
Seleccionado por su robustez ante outliers y capacidad de capturar relaciones no lineales mediante ensemble de árboles independientes. Se aplicaron parámetros de regularización para eliminar el overfitting severo de la configuración por defecto.

```python
RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=10,
    min_samples_split=20,
    max_features='sqrt',
    bootstrap=True,
    random_state=42
)
```

---

## 📈 Resultados

**Variables finales seleccionadas tras reducción por correlación:**

| Variable | Correlación con target |
|---|---|
| `employment_status_Unemployed` | -0.62 |
| `employment_status_Employed` | +0.42 |
| `debt_to_income_ratio` | -0.34 |
| `credit_score` | +0.23 |
| `employment_status_Student` | -0.19 |
| `interest_rate` | -0.13 |

**Comparativa de modelos optimizados:**

| Modelo | Accuracy Train | Accuracy Val | F1 Train | AUC-ROC Val |
|---|---|---|---|---|
| Logistic Regression | 0.8995 | 0.8997 | 0.9393 | 0.9083 |
| Random Forest | 0.9044 | 0.9027 | 0.9427 | 0.9135 |
| **LightGBM** ⭐ | **0.9089** | **0.9053** | **0.9451** | **0.9202** |

**Diagnóstico de overfitting:**

| Modelo | Diferencia Train/Val | Diagnóstico |
|---|---|---|
| Logistic Regression | 0.0002 | ✅ Sin overfitting |
| Random Forest | 0.0016 | ✅ Sin overfitting |
| LightGBM | 0.0036 | ✅ Sin overfitting |

> **Modelo ganador: LightGBM** con AUC-ROC de 0.9202, sin overfitting y mejor rendimiento en todas las métricas evaluadas.

---

## 💡 Decisiones Técnicas Relevantes

**Sobre el desbalance de clases:**
Se evaluó SMOTE como técnica de balanceo pero fue descartado en producción dado que genera registros sintéticos que no reflejan comportamientos reales del mercado, lo que podría sobreestimar la capacidad del modelo ante reguladores. Como alternativa se utilizó `class_weight='balanced'` en Random Forest, más transparente y auditable en contexto bancario regulado.

**Sobre la selección de LR en banca real:**
Aunque LightGBM supera a LR en rendimiento, la Regresión Logística sigue siendo el estándar en banca por requerimientos regulatorios de Basilea III que exigen modelos auditables y explicables. Sus coeficientes β permiten justificar cada decisión de crédito ante reguladores y clientes.

**Sobre el overfitting en Random Forest:**
Sin parámetros de regularización, RF presentaba overfitting severo con diferencia Train/Val de 0.10. La aplicación de `max_depth=10` y `min_samples_leaf=10` redujo esta diferencia a 0.0016.

---

## 📁 Estructura del Repositorio

```
├── Trabajo_Aprendizaje_Automatico_Predict_Loan_Payback.ipynb  # Notebook principal
├── train.csv                                                   # Dataset de entrenamiento
├── test.csv                                                    # Dataset de test del challenge
├── sample.csv                                                  # Predicciones para submission
├── requirements.txt                                            # Dependencias
└── README.md                                                   # Este archivo
```

---

## ▶️ Cómo Ejecutar

1. Clonar el repositorio
```bash
git clone <url-repositorio>
```

2. Instalar dependencias
```bash
pip install -r requirements.txt
```

3. Colocar `train.csv` y `test.csv` en la misma carpeta que el notebook

4. Ejecutar el notebook completo → **Runtime → Run all** en Google Colab o de forma secuencial en VS Code

5. El archivo `sample.csv` con las predicciones se generará automáticamente al ejecutar la última celda

---

## 🛠️ Librerías Utilizadas

```
numpy · pandas · matplotlib · seaborn · scikit-learn · lightgbm
```
