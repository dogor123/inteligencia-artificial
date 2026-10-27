# Predicción de Lluvia

Proyecto de la asignatura Inteligencia Artificial.
Predicción de lluvia usando datos climáticos de Cartago, Valle del Cauca (simulados).

## Tecnologías
- Python, NumPy, Matplotlib
- Docker, Docker Compose
- Git, GitHub

## Ejecución
```bash
docker compose build
docker compose up -d
docker exec -it prediccion_lluvia bash
python src/eda_lluvia.py
```


---

## Avance 2 (Segundo Corte)

En este corte se amplió el alcance del proyecto a tres localidades reales del norte del
Valle del Cauca: **El Billar**, **Zaragoza** y **Ansermanuevo**. Ante la imposibilidad de
obtener series históricas completas (temperatura, humedad, presión) para esas tres zonas
específicas desde IDEAM/CVC, se construyó un **dataset simulado anclado en climatología
real** investigada para la región (ver detalle en el documento `Avance2_GrupoAgricultura_PrediccionLluvia.pdf`, sección 2).

### Estructura añadida

```
src/avance2/
├── generar_datos.py        # Genera el dataset simulado (3 localidades, 87 días c/u)
├── preparar_datos.py       # Limpieza y preparación con Pandas
├── visualizaciones.py      # 4 visualizaciones con Seaborn
└── perceptron_proyecto.py  # Perceptrón desde cero con NumPy

data/
├── clima_norte_valle_raw.csv   # Dataset crudo (267 registros, con defectos intencionales)
└── proyecto_limpio.csv         # Dataset limpio (261 registros, 0 nulos, 0 duplicados)
```

### Resultados principales

- **Preparación de datos:** de 267 registros crudos (13 valores nulos, 9 variantes de la
  categoría `localidad`, 6 duplicados) se obtuvo un dataset limpio de 261 registros, 0 nulos
  y 0 duplicados, codificado con One-Hot Encoding.
- **Análisis visual:** El Billar presenta el mayor porcentaje de días con lluvia (67.8%),
  seguido de Ansermanuevo (57.5%) y Zaragoza (52.9%). Existe correlación positiva (0.28)
  entre humedad y precipitación, y la presión promedio es más baja en los días de lluvia
  (1012.4 hPa vs. 1013.0 hPa).
- **Perceptrón (NumPy, sin Scikit-Learn):** entrenado con humedad y presión estandarizadas.
  Pesos finales: w(humedad) = 0.4136, w(presión) = -0.2006, sesgo = 0.4009. Pérdida:
  0.6931 → 0.6491. Precisión: 60.15%.

### Cómo ejecutar el Avance 2

```bash
docker compose build
docker compose up -d
docker exec -it prediccion_lluvia bash
cd /app
python src/avance2/generar_datos.py
python src/avance2/preparar_datos.py
python src/avance2/visualizaciones.py
python src/avance2/perceptron_proyecto.py
exit
```

Las gráficas generadas quedan en `outputs/`.

### Plan hacia el proyecto final

Comparación de regresión logística, árbol de decisión y KNN con Scikit-Learn (Clase 12),
evaluados con F1-score además de accuracy, dado que un falso negativo (predecir "no
llueve" cuando sí llueve) tiene mayor costo práctico para el agricultor que un falso
positivo. Ver cronograma completo en el documento del Avance 2.