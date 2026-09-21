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