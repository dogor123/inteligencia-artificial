#!/bin/bash
# Ejecuta todo el proyecto dentro del contenedor y guarda la salida en outputs/log_ejecucion.txt
# Uso (desde la raiz del proyecto, en Ubuntu):  bash ejecutar_todo.sh
docker compose up -d
docker exec prediccion_lluvia bash -c "cd /app \
 && python src/avance2/preparar_datos.py \
 && python src/avance2/visualizaciones.py \
 && python src/avance2/perceptron_proyecto.py \
 && python src/modelo_proyecto.py \
 && python src/usar_modelo.py" 2>&1 | tee outputs/log_ejecucion.txt
