"""
preparar_datos.py
Preparacion y limpieza de datos con Pandas (Avance 2, Seccion B).
Sigue la misma estructura vista en Clase 7 (dataset de ventas):
exploracion inicial -> deteccion de problemas -> tratamiento justificado
-> codificacion de categorias -> exportacion del dataset limpio.

Proyecto: Prediccion de Lluvia - El Billar, Zaragoza y Ansermanuevo
(Valle del Cauca)
"""

import pandas as pd

ARCHIVO_ENTRADA = "data/clima_norte_valle_raw.csv"
ARCHIVO_SALIDA = "data/proyecto_limpio.csv"


def cargar_datos(archivo):
    """Carga el CSV crudo en un DataFrame de Pandas."""
    df = pd.read_csv(archivo)
    print(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas.")
    return df


def explorar_datos(df):
    """
    Exploracion inicial del dataset, igual a como se hizo en clase
    con el dataset de ventas: forma, tipos, nulos y duplicados.
    """
    print("\n" + "=" * 60)
    print("EXPLORACION INICIAL")
    print("=" * 60)

    print("\n--- df.info() ---")
    df.info()

    print("\n--- df.head() ---")
    print(df.head())

    print("\n--- Valores nulos por columna ---")
    print(df.isnull().sum())

    print("\n--- Filas duplicadas ---")
    print(f"Duplicados exactos: {df.duplicated().sum()}")

    print("\n--- Valores unicos en 'localidad' (antes de limpiar) ---")
    print(df["localidad"].unique())

    return {
        "filas": df.shape[0],
        "columnas": df.shape[1],
        "nulos": df.isnull().sum().to_dict(),
        "duplicados": int(df.duplicated().sum()),
    }


def limpiar_datos(df):
    """
    Trata cada problema detectado, con su justificacion:

    1. Duplicados exactos -> se eliminan (son el mismo registro repetido
       por un error de doble captura; conservarlos duplicaria el peso de
       esos dias en el analisis y en el entrenamiento del perceptron).

    2. Inconsistencia de categoria en 'localidad' (mayusculas/espacios) ->
       se normaliza con .str.strip().str.title() para que
       "ANSERMANUEVO", "  Ansermanuevo  " y "ansermanuevo" se traten
       como la misma localidad.

    3. Nulos en 'humedad' y 'temperatura' -> se imputan con la MEDIANA
       de esa misma localidad (no la media global), porque cada localidad
       tiene un clima base distinto (elevacion/microclima) y la mediana es
       mas robusta a los valores atipicos que la media.

    4. Valor imposible de humedad (>100%) -> se trata como dato erroneo
       de sensor (no como un valor real extremo) y se reemplaza por la
       mediana de humedad de esa localidad, igual que un nulo.

    5. Tipos de dato -> se fuerza 'llovio' a entero y 'fecha' a tipo
       datetime, para poder ordenar y derivar variables de fecha mas
       adelante si se necesita.
    """
    df = df.copy()

    # 2. Normalizar categoria 'localidad'
    df["localidad"] = df["localidad"].str.strip().str.title()

    # 1. Eliminar duplicados exactos (ya con localidad normalizada)
    antes = len(df)
    df = df.drop_duplicates()
    print(f"\nDuplicados eliminados: {antes - len(df)}")

    # 4. Humedad imposible (>100%) -> tratar como nulo antes de imputar
    df.loc[df["humedad"] > 100, "humedad"] = pd.NA

    # 3. Imputar nulos de humedad y temperatura con la mediana POR LOCALIDAD
    for columna in ["humedad", "temperatura"]:
        df[columna] = df.groupby("localidad")[columna].transform(
            lambda serie: serie.fillna(serie.median())
        )

    # 5. Tipos de dato correctos
    df["fecha"] = pd.to_datetime(df["fecha"])
    df["llovio"] = df["llovio"].astype(int)

    # Verificacion de outliers logicos (precipitacion no puede ser negativa,
    # humedad debe estar entre 0 y 100 tras la limpieza)
    assert (df["precipitacion_mm"] >= 0).all(), "Hay precipitacion negativa"
    assert df["humedad"].between(0, 100).all(), "Humedad fuera de rango tras limpieza"

    return df


def codificar_categorias(df):
    """
    Codifica la variable categorica 'localidad' con One-Hot Encoding
    (pd.get_dummies), tal como se hizo en Clase 7 con la columna de
    categoria de producto. Esto permite usar la localidad como entrada
    numerica en modelos posteriores (perceptron, Scikit-Learn).
    """
    df_codificado = pd.get_dummies(df, columns=["localidad"], prefix="loc")
    return df_codificado


def resumen_antes_despues(resumen_inicial, df_final):
    """Genera la tabla 'antes vs despues' pedida en el Avance 2."""
    print("\n" + "=" * 60)
    print("RESUMEN ANTES / DESPUES")
    print("=" * 60)
    print(f"{'Metrica':<25}{'Antes':<15}{'Despues'}")
    print(f"{'Filas':<25}{resumen_inicial['filas']:<15}{df_final.shape[0]}")
    print(f"{'Columnas':<25}{resumen_inicial['columnas']:<15}{df_final.shape[1]}")
    total_nulos_antes = sum(resumen_inicial["nulos"].values())
    print(f"{'Valores nulos totales':<25}{total_nulos_antes:<15}{int(df_final.isnull().sum().sum())}")
    print(f"{'Duplicados':<25}{resumen_inicial['duplicados']:<15}{int(df_final.duplicated().sum())}")


def main():
    print("=" * 60)
    print(" PREPARACION DE DATOS CON PANDAS - AVANCE 2")
    print("=" * 60)

    df_crudo = cargar_datos(ARCHIVO_ENTRADA)
    resumen_inicial = explorar_datos(df_crudo)

    df_limpio = limpiar_datos(df_crudo)
    df_final = codificar_categorias(df_limpio)

    resumen_antes_despues(resumen_inicial, df_final)

    df_final.to_csv(ARCHIVO_SALIDA, index=False)
    print(f"\nDataset limpio guardado en: {ARCHIVO_SALIDA}")
    print(f"Forma final: {df_final.shape}")


if __name__ == "__main__":
    main()