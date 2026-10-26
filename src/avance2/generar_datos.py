"""
generar_datos.py
Genera un dataset climático simulado (pero basado en climatología real)
para tres localidades del norte del Valle del Cauca: El Billar, Zaragoza
y Ansermanuevo (corregimientos/cabecera del municipio de Ansermanuevo,
vecino de Cartago).

Base real usada para anclar la simulación (no son valores inventados al azar):
- Climatología histórica de Ansermanuevo (weatherapi.com / IDEAM): mayo
  ~659 mm/mes, 28 días de lluvia, temp. 18-25°C; diciembre ~398 mm/mes,
  25 días de lluvia, temp. 16-24°C, presión ~1012-1013 hPa.
- Régimen bimodal de lluvias del Valle del Cauca: "veranillo" (época más
  seca) a mitad de año, segunda temporada de lluvias entre septiembre
  y noviembre.
- Reportes hidroclimatológicos de la CVC (portal-hidroclimatologico.cvc.gov.co):
  2026 presenta un episodio de El Niño fortaleciéndose (ONI +1.8°C en el
  trimestre junio-julio-agosto 2026), lo que históricamente reduce la
  precipitación en la región Andina colombiana durante ese periodo.

El dataset se genera "crudo" (con nulos, duplicados y una inconsistencia
de categoría a propósito) para poder practicar la limpieza con Pandas,
tal como se hizo en la Clase 7 con el dataset de ventas.
"""

import numpy as np
import csv
from datetime import date, timedelta

np.random.seed(42)

LOCALIDADES = {
    # nombre: (temp_base, humedad_base, presion_base, factor_lluvia)
    "Ansermanuevo": {"temp_base": 21.5, "humedad_base": 78, "presion_base": 1013, "elevacion_factor": 1.0},
    "El Billar":    {"temp_base": 19.5, "humedad_base": 82, "presion_base": 1011, "elevacion_factor": 1.15},
    "Zaragoza":     {"temp_base": 22.0, "humedad_base": 75, "presion_base": 1014, "elevacion_factor": 0.9},
}

FECHA_INICIO = date(2026, 8, 1)
FECHA_FIN = date(2026, 10, 26)  # fecha actual del proyecto


def factor_estacional(dia_del_periodo, total_dias):
    """
    Modela el régimen bimodal: agosto más seco (veranillo + El Niño 2026),
    incremento gradual hacia la segunda temporada de lluvias en octubre.
    Devuelve un multiplicador de probabilidad/intensidad de lluvia.
    """
    progreso = dia_del_periodo / total_dias
    # Curva suave que va de 0.55 (agosto, seco) a 1.25 (fin de octubre, más lluvioso)
    return 0.55 + 0.70 * progreso


def generar_fila(fecha, localidad, params, dia_del_periodo, total_dias):
    factor = factor_estacional(dia_del_periodo, total_dias)

    # Presión: oscila con ruido pequeño, ligeramente más baja en días húmedos
    presion = np.random.normal(params["presion_base"], 2.5)

    # Humedad relativa: con tendencia estacional y ruido
    humedad = np.random.normal(params["humedad_base"] * factor if factor < 1 else params["humedad_base"], 6)
    humedad = np.clip(humedad, 45, 98)

    # Probabilidad de lluvia: sube con humedad alta y baja con presión alta
    prob_lluvia = 0.08 + 0.55 * factor
    prob_lluvia += max(0, (humedad - 80)) * 0.01
    prob_lluvia -= max(0, (presion - 1014)) * 0.03
    prob_lluvia = np.clip(prob_lluvia, 0.03, 0.9)

    llovio = 1 if np.random.random() < prob_lluvia else 0

    if llovio:
        precipitacion = float(np.round(np.random.gamma(shape=2.0, scale=6.0 * factor) + 0.5, 1))
        temp_ajuste = -1.5  # días lluviosos ligeramente más frescos
    else:
        precipitacion = 0.0
        temp_ajuste = 0.8

    temperatura = np.random.normal(
        params["temp_base"] * params["elevacion_factor"] + temp_ajuste, 1.4
    )
    temperatura = float(np.round(temperatura, 1))

    return {
        "fecha": fecha.isoformat(),
        "localidad": localidad,
        "temperatura": temperatura,
        "humedad": round(float(humedad), 1),
        "presion": round(float(presion), 1),
        "precipitacion_mm": precipitacion,
        "llovio": llovio,
    }


def generar_dataset():
    filas = []
    total_dias = (FECHA_FIN - FECHA_INICIO).days + 1

    for localidad, params in LOCALIDADES.items():
        fecha = FECHA_INICIO
        dia = 0
        while fecha <= FECHA_FIN:
            fila = generar_fila(fecha, localidad, params, dia, total_dias)
            filas.append(fila)
            fecha += timedelta(days=1)
            dia += 1

    return filas


def ensuciar_dataset(filas):
    """
    Introduce problemas reales de calidad de datos a propósito, para
    practicar limpieza con Pandas (Avance 2, requisito B):
    - Algunos valores nulos en humedad y temperatura (sensor con falla)
    - Algunas filas duplicadas exactas (error de doble registro)
    - Inconsistencia de categoría en 'localidad' (mayúsculas/espacios)
    - Un valor atípico/imposible de humedad (error de captura)
    """
    filas_sucias = [dict(f) for f in filas]
    n = len(filas_sucias)

    # 1. Nulos en humedad (simula falla de sensor en ~3% de los registros)
    idx_nulos_humedad = np.random.choice(n, size=max(1, n // 33), replace=False)
    for i in idx_nulos_humedad:
        filas_sucias[i]["humedad"] = ""

    # 2. Nulos en temperatura (~2%)
    idx_nulos_temp = np.random.choice(n, size=max(1, n // 50), replace=False)
    for i in idx_nulos_temp:
        filas_sucias[i]["temperatura"] = ""

    # 3. Duplicar algunas filas completas (error de doble registro)
    idx_duplicar = np.random.choice(n, size=max(1, n // 40), replace=False)
    for i in idx_duplicar:
        filas_sucias.append(dict(filas_sucias[i]))

    # 4. Inconsistencia de categoría en 'localidad'
    for i in np.random.choice(len(filas_sucias), size=max(1, len(filas_sucias) // 25), replace=False):
        valor = filas_sucias[i]["localidad"]
        variante = np.random.choice(["MAYUS", "espacios", "minus"])
        if variante == "MAYUS":
            filas_sucias[i]["localidad"] = valor.upper()
        elif variante == "espacios":
            filas_sucias[i]["localidad"] = f"  {valor}  "
        else:
            filas_sucias[i]["localidad"] = valor.lower()

    # 5. Un valor imposible de humedad (error de captura, ej. sensor corrupto)
    idx_outlier = np.random.choice(len(filas_sucias), size=2, replace=False)
    for i in idx_outlier:
        filas_sucias[i]["humedad"] = 145.0  # imposible: >100%

    np.random.shuffle(filas_sucias)
    return filas_sucias


def guardar_csv(filas, archivo_salida):
    campos = ["fecha", "localidad", "temperatura", "humedad", "presion", "precipitacion_mm", "llovio"]
    with open(archivo_salida, "w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=campos)
        escritor.writeheader()
        for fila in filas:
            escritor.writerow(fila)
    print(f"Dataset guardado: {archivo_salida} ({len(filas)} registros)")


if __name__ == "__main__":
    datos_limpios = generar_dataset()
    datos_crudos = ensuciar_dataset(datos_limpios)
    guardar_csv(datos_crudos, "clima_norte_valle_raw.csv")