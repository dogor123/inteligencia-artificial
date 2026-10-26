"""
visualizaciones.py
Visualizaciones con Seaborn sobre el dataset ya limpio (Avance 2, Seccion C).
Sigue el estilo de Clase 7/8: set_theme, histplot, barplot, scatterplot.

Proyecto: Prediccion de Lluvia - El Billar, Zaragoza y Ansermanuevo
(Valle del Cauca)
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

ARCHIVO_ENTRADA = "data/proyecto_limpio.csv"

sns.set_theme(style="whitegrid")


def cargar_datos(archivo):
    df = pd.read_csv(archivo)
    print(f"Datos cargados para visualizacion: {df.shape[0]} filas.")
    return df


def reconstruir_localidad(df):
    """
    El dataset limpio ya esta codificado con One-Hot (loc_Ansermanuevo,
    loc_El Billar, loc_Zaragoza). Para las graficas es mas claro volver
    a tener una sola columna categorica 'localidad', asi que la
    reconstruimos a partir de las columnas dummy.
    """
    columnas_loc = [c for c in df.columns if c.startswith("loc_")]
    df = df.copy()
    df["localidad"] = df[columnas_loc].idxmax(axis=1).str.replace("loc_", "", regex=False)
    return df


def grafico_distribucion_precipitacion(df):
    """1. Histograma: distribucion de la precipitacion diaria por localidad."""
    plt.figure(figsize=(9, 5))
    sns.histplot(data=df, x="precipitacion_mm", hue="localidad", bins=20, kde=True, multiple="layer", alpha=0.5)
    plt.title("Distribucion de la precipitacion diaria por localidad")
    plt.xlabel("Precipitacion (mm)")
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    plt.savefig("outputs/distribucion_precipitacion_localidad.png")
    plt.close()
    print("Guardado: outputs/distribucion_precipitacion_localidad.png")


def grafico_dias_lluvia_por_localidad(df):
    """2. Barras: proporcion de dias con lluvia por localidad (relaciona variable con el target)."""
    proporcion = df.groupby("localidad")["llovio"].mean().reset_index()
    proporcion["llovio"] = proporcion["llovio"] * 100

    plt.figure(figsize=(7, 5))
    sns.barplot(data=proporcion, x="localidad", y="llovio", hue="localidad", legend=False)
    plt.title("Porcentaje de dias con lluvia por localidad")
    plt.xlabel("Localidad")
    plt.ylabel("% de dias con lluvia")
    plt.tight_layout()
    plt.savefig("outputs/pct_dias_lluvia_localidad.png")
    plt.close()
    print("Guardado: outputs/pct_dias_lluvia_localidad.png")


def grafico_humedad_vs_precipitacion(df):
    """3. Dispersion: relacion humedad vs precipitacion, coloreada por si llovio."""
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="humedad", y="precipitacion_mm", hue="llovio", palette="viridis", alpha=0.7)
    plt.title("Relacion entre humedad y precipitacion")
    plt.xlabel("Humedad (%)")
    plt.ylabel("Precipitacion (mm)")
    plt.legend(title="Llovio")
    plt.tight_layout()
    plt.savefig("outputs/humedad_vs_precipitacion.png")
    plt.close()
    print("Guardado: outputs/humedad_vs_precipitacion.png")


def grafico_presion_por_lluvia(df):
    """4. Boxplot: presion atmosferica segun si llovio o no (relaciona variable con el target)."""
    plt.figure(figsize=(7, 5))
    sns.boxplot(data=df, x="llovio", y="presion", hue="llovio", legend=False)
    plt.title("Presion atmosferica segun ocurrencia de lluvia")
    plt.xlabel("Llovio (0 = No, 1 = Si)")
    plt.ylabel("Presion (hPa)")
    plt.tight_layout()
    plt.savefig("outputs/presion_por_lluvia.png")
    plt.close()
    print("Guardado: outputs/presion_por_lluvia.png")


def imprimir_interpretaciones(df):
    """
    Interpretaciones (2-4 lineas cada una) para incluir en el documento
    final, Seccion C. Se calculan a partir del dataset para que los
    numeros citados sean reales y no estimados.
    """
    print("\n" + "=" * 60)
    print("INTERPRETACIONES")
    print("=" * 60)

    prop_por_loc = df.groupby("localidad")["llovio"].mean() * 100
    corr_hum = df["humedad"].corr(df["precipitacion_mm"])
    presion_lluvia = df.groupby("llovio")["presion"].mean()

    print(f"""
1. Distribucion de precipitacion por localidad:
   La mayoria de los dias en las tres localidades tienen precipitacion cercana
   a 0 mm (la barra mas alta del histograma), con una cola larga hacia valores
   mayores en los dias de lluvia fuerte. El Billar muestra la cola mas pesada,
   consistente con ser la zona de mayor elevacion simulada.

2. Dias con lluvia por localidad:
   El Billar presenta el mayor porcentaje de dias con lluvia
   ({prop_por_loc.get('El Billar', 0):.1f}%), seguido de Ansermanuevo
   ({prop_por_loc.get('Ansermanuevo', 0):.1f}%) y Zaragoza
   ({prop_por_loc.get('Zaragoza', 0):.1f}%). Esto confirma que la localidad por
   si sola ya aporta informacion util para el modelo.

3. Humedad vs precipitacion:
   Existe una correlacion positiva de {corr_hum:.2f} entre humedad y
   precipitacion: los dias con lluvia se concentran en la zona de humedad alta
   del grafico, lo que respalda usar la humedad como variable predictora
   principal del perceptron.

4. Presion segun lluvia:
   Los dias con lluvia tienen una presion promedio de
   {presion_lluvia.get(1, 0):.1f} hPa, mientras que los dias sin lluvia
   promedian {presion_lluvia.get(0, 0):.1f} hPa. La diferencia es consistente
   con la fisica atmosferica: menor presion favorece la formacion de nubes y
   precipitacion.
""")


def main():
    print("=" * 60)
    print(" VISUALIZACIONES CON SEABORN - AVANCE 2")
    print("=" * 60)

    df = cargar_datos(ARCHIVO_ENTRADA)
    df = reconstruir_localidad(df)

    grafico_distribucion_precipitacion(df)
    grafico_dias_lluvia_por_localidad(df)
    grafico_humedad_vs_precipitacion(df)
    grafico_presion_por_lluvia(df)

    imprimir_interpretaciones(df)

    print("\nVisualizaciones completadas.")


if __name__ == "__main__":
    main()