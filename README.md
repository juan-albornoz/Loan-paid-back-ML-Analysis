# 🏦 Predicting Loan Payback - ML Challenge

## Información del Proyecto

| Campo | Detalle |
|---|---|
| **Asignatura** | Aprendizaje Automático |
| **Institución** | Universidad Europea de Valencia |
| **Profesor** | Jorge Fresco |
| **Fecha** | Marzo 2026 |

**Autores:**
- Camilo Velasquez
- Fernando Pirrongelli
- Juan Albornoz

---

## 📋 Descripción

Proyecto de Machine Learning para predecir si un cliente bancario pagará o no su préstamo. El problema es de **clasificación binaria supervisada**:

- `1` → Cliente pagará el préstamo
- `0` → Cliente NO pagará el préstamo

El dataset simula un escenario real de evaluación de riesgo crediticio, con información financiera y sociodemográfica de solicitantes de crédito. La métrica principal de evaluación es el **AUC-ROC**.

---

## 📁 Estructura del Repositorio

```
├── Trabajo_Aprendizaje_Automatico_Predict_Loan_Payback.ipynb  # Notebook principal
├── train.csv                                                   # Dataset de entrenamiento
├── test.csv                                                    # Dataset de test del challenge
├── sample.csv                                                  # Predicciones para submission
└── README.md                                                   # Este archivo
```

---

## 📊 Dataset

| Característica | Detalle |
|---|---|
| **Registros totales** | ~800,000 |
| **Columnas originales** | 13 |
| **Variable objetivo** | `loan_paid_back` |

**Variables del dataset:**

| Variable | Tipo | Descripción |
|---|---|---|
| `id` | Numérica | Identificador único |
| `annual_income` | Numérica | Ingreso anual del cliente |
| `debt_to_income_ratio` | Numérica | Ratio deuda/ingreso |
| `credit_score` | Numérica | Puntuación crediticia |
| `loan_amount` | Numérica | Monto del préstamo |
| `interest_rate` | Numérica | Tasa de interés |
| `gender` | Categórica | Género del cliente |
| `marital_status` | Categórica | Estado civil |
| `education_level` | Categórica | Nivel educativo |
| `employment_status` | Categórica | Situación laboral |
| `loan_purpose` | Categórica | Propósito del préstamo |
| `grade_subgrade` | Categórica | Calificación crediticia |

---

## 🔧 Preprocesamiento

1. **Eliminación de duplicados**
2. **One-Hot Encoding** sobre variables categóricas → de 13 a 62 columnas
3. **Reducción de variables** por correlación con el target (umbral 0.05) → 6 variables relevantes finales
4. **Detección y eliminación de outliers** mediante método IQR
5. **Train/Val split** → 70% entrenamiento / 30% validación (`random_state=0`)
6. **StandardScaler** aplicado sobre conjunto de entrenamiento

**Variables finales seleccionadas:**

| Variable | Correlación con target |
|---|---|
| `employment_status_Unemployed` | -0.62 |
| `employment_status_Employed` | +0.42 |
| `debt_to_income_ratio` | -0.34 |
| `credit_score` | +0.23 |
| `employment_status_Student` | -0.19 |
| `interest_rate` | -0.13 |

---

## 🤖 Modelos Entrenados

### 1. Logistic Regression
Seleccionado por su alta interpretabilidad, auditabilidad y cumplimiento regulatorio bancario (Basilea III). Sus coeficientes β permiten explicar exactamente por qué se aprueba o rechaza un crédito.

```python
LogisticRegression(max_iter=1000)
# Optimizado con LogisticRegressionCV
```

### 2. Random Forest
Seleccionado por su capacidad de capturar relaciones no lineales entre variables y su robustez ante outliers. Se aplicaron parámetros de regularización para eliminar el overfitting severo que presentaba en su configuración por defecto.

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

### 3. LightGBM
Seleccionado por su arquitectura de Gradient Boosting secuencial, manejo nativo de variables categóricas, algoritmo GOSS para datasets grandes y regularización L1/L2 incorporada.

```python
LGBMClassifier(random_state=42)
# Optimizado con RandomizedSearchCV
```

---

## 📈 Resultados

### Modelos Base

| Modelo | Accuracy Train | Accuracy Val | F1 Train | AUC-ROC Val |
|---|---|---|---|---|
| Logistic Regression | 0.8995 | 0.8997 | 0.9393 | 0.9083 |
| Random Forest | 0.9044 | 0.9027 | 0.9427 | 0.9135 |
| **LGBM** | **0.9089** | **0.9053** | **0.9451** | **0.9202** |

### Modelos Optimizados

| Modelo | Accuracy Train | Accuracy Val | F1 Train | AUC-ROC Val |
|---|---|---|---|---|
| Logistic Regression | 0.8995 | 0.8997 | 0.9393 | 0.9083 |
| Random Forest | 0.9044 | 0.9027 | 0.9427 | 0.9135 |
| **LGBM** | **0.9089** | **0.9053** | **0.9451** | **0.9202** |

> **Modelo ganador: LightGBM** con AUC-ROC Val de 0.9202

---

## 🔍 Análisis de Overfitting

| Modelo | Diferencia Train/Val | Diagnóstico |
|---|---|---|
| Logistic Regression | 0.0002 | ✅ Sin overfitting |
| Random Forest | 0.0016 | ✅ Sin overfitting |
| LGBM | 0.0036 | ✅ Sin overfitting |

El Random Forest sin parámetros de regularización presentaba overfitting severo (diferencia de 0.10). La aplicación de `max_depth=10` y `min_samples_leaf=10` redujo esta diferencia a 0.0016.

---

## 🏗️ Estructura del Notebook

```
1. Análisis y comprensión del problema
2. Exploración y análisis de datos (EDA)
3. Preprocesamiento
   ├── Eliminación de duplicados
   ├── One-Hot Encoding
   ├── Selección de variables por correlación
   ├── Detección de outliers (IQR)
   ├── Train/Val split
   └── StandardScaler
4. Selección y entrenamiento de modelos
   ├── Logistic Regression
   ├── Random Forest
   └── LightGBM
5. Optimización
   ├── LogisticRegressionCV
   ├── RandomizedSearchCV para RF
   └── RandomizedSearchCV para LGBM
6. Evaluación
   ├── Matrices de confusión
   ├── Curvas ROC
   ├── Diagnóstico overfitting/underfitting
   └── Tabla comparativa de modelos
7. Conclusiones
```

---

## 🛠️ Librerías Utilizadas

```python
numpy
pandas
matplotlib
seaborn
scikit-learn
lightgbm
math
warnings
```

---

## ▶️ Cómo Ejecutar

1. Clonar el repositorio
```bash
git clone <url-repositorio>
```

2. Instalar dependencias
```bash
pip install numpy pandas matplotlib seaborn scikit-learn lightgbm
```

3. Colocar `train.csv` y `test.csv` en la misma carpeta que el notebook

4. Ejecutar el notebook completo con **Runtime → Run all** en Google Colab o de forma secuencial en VS Code

5. El archivo `sample.csv` con las predicciones se generará automáticamente al final

---

## 📝 Notas Técnicas

- El **StandardScaler** se ajusta únicamente sobre `x_train` y se aplica con `.transform()` sobre `x_val` para evitar data leakage
- **SMOTE** fue evaluado como técnica de balanceo de clases pero descartado por generar registros sintéticos que no reflejan comportamientos reales del mercado, relevante en contexto bancario regulado
- La métrica principal de evaluación del challenge es **AUC-ROC**, no Accuracy
- Las predicciones del `sample.csv` corresponden a probabilidades de pago generadas por el modelo **LightGBM optimizado**
