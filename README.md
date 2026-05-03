# Proyecto Final - Aprendizaje Automático
## Predicción del éxito académico en educación superior
**Asignatura:** Aprendizaje Automático 2025/2026
**Universidad:** Comillas ICAI

---

## Descripción del proyecto

Este proyecto aplica técnicas de Machine Learning sobre un dataset de trayectorias académicas para predecir el abandono, la permanencia y el éxito académico de los estudiantes. El trabajo se estructura en tres tareas:
1. Clasificación supervisada: predicción de si la situación final del estudiante será Abandono, Matriculado o Graduado.
2. Regresión supervisada: predicción de la calificación media del segundo semestre.
3. Aprendizaje no supervisado: identificación de perfiles de estudiantes.

---

## Librerias necesarias
numpy
pandas
matplotlib
seaborn
scikit.learns

Pueden instalarse con:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

---

## Cómo reproducir los resultados
> Es importante ejecutar los notebooks en orden, ya que uno depende del anterior.

### Paso 1.- EDA y preprocesado     
```bash
jupyter notebook eda.ipynb
```
Este notebook carga el CSV original, realiza el análisis exploratorio, transforma las variables categóricas binarias y agrupa las multinomiales, y guarda el dataset preprocesado en 'data/df_limpio.pkl'

### Paso 2.- Clasificación      
```bash
jupyter notebook clasificacion.ipynb
```
Carga 'df_limpio.pkl' y entrena los modelos de KNN, Regresión Logística y Random Forest con validación cruzada y GridSearchCV. Compara los resultados usando F1-macro.

### Paso 3.- Regresión      
```bash
jupyter notebook regresion.ipynb
```
Carga 'df_limpio.pkl' y entre los modelos de regresión Ridge, Lasso, ElasticNet y Randomforest con validación cruzada. Compara los resultados usando MAE, RMSE y R2.

### Paso 4.- Aprendizaje no supervisado     
```bash
jupyter notebook no_supervisado.ipynb
```
Carga 'df_limpio.pkl' y aplica PCA y K-Means implementados a mano y luego los verifica con sklearn. Incluye el método del codo y de la silueta para seleccionar el número óptimo de clusters (k).

---

## Reproducibilidad         
Todos los modelos usan 'random_state=42' para garantizar la reproducibilidad de los resultados.

