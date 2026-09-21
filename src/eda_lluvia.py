"""
eda_lluvia.py
Script para realizar un Analisis Exploratorio de Datos (EDA) basico
sobre un archivo CSV de clima, utilizando NumPy y Matplotlib.

Proyecto: Prediccion de Lluvia (Cartago, Valle del Cauca)
Sigue la misma estructura de eda_cultivos.py (Clase 4).
"""

import numpy as np
import matplotlib.pyplot as plt
import csv


def cargar_datos(archivo_csv):
    """
    Carga un archivo CSV de clima y retorna:
    - un array de NumPy con las columnas numericas (temperatura, humedad, precipitacion, presion)
    - la lista completa de registros (diccionarios)
    - los nombres de las columnas
    """
    datos = []
    nombres_columnas = []

    try:
        with open(archivo_csv, 'r', encoding='utf-8') as archivo:
            lector = csv.DictReader(archivo)
            nombres_columnas = lector.fieldnames

            for fila in lector:
                registro = {
                    "fecha": fila["fecha"],
                    "temperatura": float(fila["temperatura"]),
                    "humedad": float(fila["humedad"]),
                    "precipitacion_mm": float(fila["precipitacion_mm"]),
                    "presion": float(fila["presion"]),
                    "llovio": int(fila["llovio"])
                }
                datos.append(registro)

        print(f"Datos cargados correctamente: {len(datos)} registros.")

    except FileNotFoundError:
        print(f"Error: El archivo '{archivo_csv}' no existe.")
        return None, None, None
    except Exception as e:
        print(f"Error al leer el archivo: {e}")
        return None, None, None

    datos_numericos = np.array(
        [[d["temperatura"], d["humedad"], d["precipitacion_mm"], d["presion"]] for d in datos],
        dtype=float
    )

    return datos_numericos, datos, nombres_columnas


def dias_con_lluvia(datos):
    """
    Funcion propia del proyecto.
    Cuenta cuantos dias del dataset tuvieron lluvia (llovio == 1).
    """
    if not datos:
        return 0
    return sum(1 for d in datos if d["llovio"] == 1)


def analizar_datos(datos_numericos):
    """Calcula estadisticas descriptivas de temperatura, humedad, precipitacion y presion."""
    if datos_numericos is None or len(datos_numericos) == 0:
        return None

    temperatura = datos_numericos[:, 0]
    humedad = datos_numericos[:, 1]
    precipitacion = datos_numericos[:, 2]
    presion = datos_numericos[:, 3]

    estadisticas = {
        'temperatura': {
            'media': np.mean(temperatura),
            'mediana': np.median(temperatura),
            'desviacion': np.std(temperatura),
            'minimo': np.min(temperatura),
            'maximo': np.max(temperatura)
        },
        'humedad': {
            'media': np.mean(humedad),
            'mediana': np.median(humedad),
            'desviacion': np.std(humedad),
            'minimo': np.min(humedad),
            'maximo': np.max(humedad)
        },
        'precipitacion': {
            'media': np.mean(precipitacion),
            'mediana': np.median(precipitacion),
            'desviacion': np.std(precipitacion),
            'minimo': np.min(precipitacion),
            'maximo': np.max(precipitacion)
        },
        'presion': {
            'media': np.mean(presion),
            'mediana': np.median(presion),
            'desviacion': np.std(presion),
            'minimo': np.min(presion),
            'maximo': np.max(presion)
        }
    }
    return estadisticas


def generar_visualizaciones(datos_numericos, datos_completos):
    """Genera graficos para el analisis exploratorio del clima."""
    if datos_numericos is None or len(datos_numericos) == 0:
        return

    humedad = datos_numericos[:, 1]
    precipitacion = datos_numericos[:, 2]
    fechas = [d["fecha"] for d in datos_completos]

    plt.figure(figsize=(10, 5))
    plt.plot(fechas, precipitacion, marker='o', linestyle='-', color='b', label='Precipitacion')
    plt.title('Precipitacion diaria (mm)')
    plt.xlabel('Fecha')
    plt.ylabel('Precipitacion (mm)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('outputs/precipitacion_diaria.png')
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.scatter(humedad, precipitacion, color='green', alpha=0.7)
    plt.title('Relacion: Humedad vs Precipitacion')
    plt.xlabel('Humedad (%)')
    plt.ylabel('Precipitacion (mm)')
    plt.grid(True)
    plt.savefig('outputs/humedad_vs_precipitacion.png')
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.hist(precipitacion, bins=5, color='orange', edgecolor='black')
    plt.title('Distribucion de la Precipitacion')
    plt.xlabel('Precipitacion (mm)')
    plt.ylabel('Frecuencia')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.savefig('outputs/distribucion_precipitacion.png')
    plt.show()


def main():
    """Funcion principal del programa."""
    print("=" * 50)
    print(" ANALISIS EXPLORATORIO DE DATOS (EDA) - LLUVIA")
    print("=" * 50)

    datos_numericos, datos_completos, columnas = cargar_datos('data/clima_cartago.csv')
    if datos_numericos is None:
        return

    total_lluvia = dias_con_lluvia(datos_completos)
    print(f"\nDias con lluvia registrados: {total_lluvia} de {len(datos_completos)}")

    estadisticas = analizar_datos(datos_numericos)
    if estadisticas:
        print("\nESTADISTICAS DESCRIPTIVAS:")
        print("-" * 40)
        for variable, valores in estadisticas.items():
            print(f"\n{variable.upper()}:")
            for key, value in valores.items():
                print(f"  {key.capitalize()}: {value:.2f}")

    print("\nGENERANDO VISUALIZACIONES...")
    generar_visualizaciones(datos_numericos, datos_completos)
    print("Visualizaciones guardadas como archivos PNG.")

    print("\nAnalisis completado exitosamente.")


if __name__ == "__main__":
    main()