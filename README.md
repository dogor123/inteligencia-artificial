# Predicción de Lluvia para Apoyo Agrícola — Norte del Valle del Cauca

Proyecto final de **Inteligencia Artificial** (Ingeniería de Sistemas, Corporación de Estudios Tecnológicos del Norte del Valle). Docente: Jhon James Cano Sánchez.
Integrantes: Esteban Diaz Vargas y Kevin Villegas.

Informe final: `docs/Informe_Final_GrupoAgricultura_PrediccionLluvia.pdf`

---

## 1. Descripción

Sistema de clasificación binaria que estima si **lloverá o no** en un día dado, a partir de humedad relativa, presión atmosférica, temperatura y localidad. Busca apoyar a agricultores del norte del Valle del Cauca (Cartago y municipios vecinos) en decisiones de siembra, fumigación, cosecha y riego. Localidades: **El Billar, Zaragoza y Ansermanuevo**.

El proyecto recorre el ciclo completo: datos → preparación (Pandas) → perceptrón desde cero (NumPy) → modelos de Scikit-Learn → evaluación → modelo guardado y demostración.

> **Aviso sobre los datos:** el dataset es **simulado**, anclado en climatología real de la región (promedios históricos de Ansermanuevo, régimen bimodal de lluvias del Valle y el episodio de El Niño 2026). No se obtuvieron series reales de humedad y presión para estas tres localidades (ver sección 5). Los resultados describen cómo se comporta el método sobre esos datos, no el clima real.

## 2. Estructura del repositorio

```
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── ejecutar_todo.sh              # corre todo el proyecto en el contenedor
├── data/
│   ├── clima_cartago.csv         # Avance 1: dataset de ejemplo (20 registros)
│   ├── clima_norte_valle_raw.csv # datos originales simulados (267 registros)
│   └── proyecto_limpio.csv       # datos limpios (261 registros, 9 columnas)
├── outputs/                      # gráficos y registros de ejecución
└── src/
    ├── eda_lluvia.py             # Avance 1: EDA con NumPy y Matplotlib
    ├── modelo_proyecto.py        # compara modelos de Scikit-Learn y guarda el mejor
    ├── usar_modelo.py            # carga el modelo guardado y predice casos nuevos
    ├── modelo_final.joblib       # modelo guardado (se regenera al entrenar)
    └── avance2/
        ├── generar_datos.py      # genera el dataset simulado
        ├── preparar_datos.py     # limpieza con Pandas → proyecto_limpio.csv
        ├── visualizaciones.py    # 4 gráficos de análisis con Seaborn
        └── perceptron_proyecto.py# perceptrón desde cero con NumPy
```

## 3. Cómo ejecutar (paso a paso con Docker)

Requisitos: Docker Desktop abierto. En Windows se usa la terminal de Ubuntu (WSL).

```bash
git clone https://github.com/dogor123/inteligencia-artificial.git
cd inteligencia-artificial          # o la carpeta donde quedó el proyecto
docker compose up -d --build        # construye la imagen y enciende el contenedor
```

**Opción A — todo con un comando** (guarda la salida en `outputs/log_ejecucion.txt`):

```bash
bash ejecutar_todo.sh
```

**Opción B — paso a paso dentro del contenedor:**

```bash
docker exec -it prediccion_lluvia bash
cd /app
python src/avance2/preparar_datos.py       # 1. limpieza  -> data/proyecto_limpio.csv
python src/avance2/visualizaciones.py      # 2. 4 gráficos de Seaborn -> outputs/
python src/avance2/perceptron_proyecto.py  # 3. perceptrón desde cero
python src/modelo_proyecto.py              # 4. compara modelos y guarda el mejor
python src/usar_modelo.py                  # 5. demostración: predice 3 días nuevos
exit
```

`usar_modelo.py` **no entrena**: solo carga `src/modelo_final.joblib`.

Si aparece `Conflict. The container name "/prediccion_lluvia" is already in use`, ejecutar `docker stop prediccion_lluvia && docker rm prediccion_lluvia` y repetir `docker compose up -d --build`.

## 4. Resultados principales

### Preparación de datos

| Métrica | Antes | Después |
|---|---|---|
| Filas | 267 | 261 |
| Columnas | 7 | 9 (localidad en One-Hot) |
| Valores nulos | 13 | 0 |
| Duplicados exactos | 5 (6 eliminados tras normalizar `localidad`) | 0 |
| Variantes de `localidad` | 9 | 3 |

Los nulos se imputan con la mediana **por localidad**; la humedad mayor a 100 % se trata como error de sensor.

### Perceptrón desde cero (humedad y presión, 208 para entrenar y 53 para probar)

| Parámetro | Valor |
|---|---|
| w humedad | 0.3697 |
| w presión | -0.1063 |
| Sesgo (b) | 0.4454 |
| Pérdida (log loss) | 0.6931 → 0.6528 |
| Exactitud entrenamiento / prueba | 61.54 % / 58.49 % |

Los datos se escalan con la media y desviación **del entrenamiento** y esos mismos valores se aplican a la prueba. Un modelo lineal de una neurona con dos variables es limitado: la exactitud modesta indica que las clases se solapan.

### Modelos de Scikit-Learn (entrada: humedad, presión, temperatura, localidad; 80/20 estratificado, `Pipeline` con `StandardScaler`)

| Modelo | Exactitud | Precisión | Recall | F1 | F1 val. cruzada (5) |
|---|---|---|---|---|---|
| **Regresión logística** | 0.830 | 0.867 | 0.839 | 0.852 | **0.833** |
| KNN (5 vecinos) | 0.774 | 0.806 | 0.806 | 0.806 | 0.803 |
| Árbol de decisión | 0.811 | 0.920 | 0.742 | 0.821 | 0.727 |
| Red neuronal (MLP) | 0.868 | 0.929 | 0.839 | 0.881 | 0.815 |

**Modelo elegido: regresión logística**, por tener el mejor F1 en validación cruzada. Con solo 53 registros de prueba, un acierto más o menos cambia mucho el porcentaje, por eso se prefiere la validación cruzada. Se usa **F1** porque un falso negativo (decir "no lloverá" cuando sí llueve) le cuesta más al agricultor que una falsa alarma. Matriz de confusión del modelo elegido: `[[18, 4], [5, 26]]` (`outputs/matriz_confusion.png`).

`precipitacion_mm` **no** se usa como entrada: `llovio` se define a partir de ella (`llovio = 1` si hay más de 0 mm), y usarla sería fuga de información.

### Gráficos (`outputs/`)

Análisis (Seaborn): `distribucion_precipitacion_localidad.png`, `pct_dias_lluvia_localidad.png`, `humedad_vs_precipitacion.png`, `presion_por_lluvia.png`.
Evaluación: `curva_aprendizaje.png`, `frontera_decision.png`, `matriz_confusion.png`.

## 5. Datos: fuente y diccionario

**Fuente.** Se intentó obtener datos reales de IDEAM (portal DHIME), CVC y Kaggle. DHIME solo entregó precipitación de la estación PALMASOLA (sin humedad ni presión); el portal de la CVC publica reportes en PDF; el archivo de Kaggle es solo un índice de estaciones. Por eso `generar_datos.py` genera 87 días por localidad (1 de agosto a 26 de octubre de 2026) con semilla fija, e introduce a propósito nulos, duplicados, inconsistencias de texto y un valor imposible para practicar la limpieza.

| Variable | Descripción |
|---|---|
| `fecha` | Día (AAAA-MM-DD) |
| `localidad` | El Billar, Zaragoza o Ansermanuevo (One-Hot en el dataset limpio) |
| `temperatura` | °C |
| `humedad` | Humedad relativa (%) |
| `presion` | Presión atmosférica (hPa) |
| `precipitacion_mm` | Precipitación diaria (mm) |
| `llovio` | Variable objetivo: 1 si `precipitacion_mm` > 0, 0 en caso contrario |

## 6. Limitaciones

- Datos simulados: los pesos aprendidos reflejan las reglas del simulador, no física real. En particular, la temperatura pesa mucho (-1.83) porque el simulador hace los días de lluvia 1.5 °C más fríos, y la presión pesa poco.
- Muestra pequeña (261 registros, 53 de prueba) y solo 87 días: no cubre un año completo.
- No se sabe cómo respondería el modelo con otra zona u otro año.

**Trabajo futuro:** validar con una estación real del IDEAM o sensores propios, y probar `GridSearchCV`.

## 7. Historial del proyecto

- **Avance 1:** EDA con NumPy y Matplotlib (`src/eda_lluvia.py`).
- **Avance 2:** dataset simulado, limpieza con Pandas, análisis con Seaborn y perceptrón desde cero.
- **Proyecto final:** comparación de modelos de Scikit-Learn, modelo guardado con `joblib` y `usar_modelo.py`.

## Tecnologías

Python 3.12 · NumPy · Pandas · Matplotlib · Seaborn · Scikit-Learn · joblib · Docker y Docker Compose · Git y GitHub
