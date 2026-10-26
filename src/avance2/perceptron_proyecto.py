"""
perceptron_proyecto.py
Perceptron simple implementado desde cero con NumPy (sin Scikit-Learn),
siguiendo exactamente la estructura vista en clase: suma ponderada
vectorizada, activacion sigmoide, funcion de perdida de entropia
cruzada binaria (log loss) y descenso de gradiente.

Proyecto: Prediccion de Lluvia - El Billar, Zaragoza y Ansermanuevo
(Valle del Cauca)

Variable objetivo: 'llovio' (0 = no llovio, 1 = llovio). El umbral ya
viene definido desde la generacion del dataset: llovio = 1 cuando la
precipitacion registrada en el dia fue mayor a 0 mm.

Variables de entrada (2, numericas, estandarizadas):
- humedad (%)
- presion (hPa)
Se eligieron estas dos porque en el analisis exploratorio (Seccion C)
mostraron la relacion mas clara con la ocurrencia de lluvia: la humedad
tiene correlacion positiva con la precipitacion, y la presion es en
promedio mas baja los dias que llovio, consistente con la fisica
atmosferica.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ARCHIVO_ENTRADA = "data/proyecto_limpio.csv"


def cargar_datos(archivo):
    df = pd.read_csv(archivo)
    X = df[["humedad", "presion"]].to_numpy(dtype=float)
    y = df["llovio"].to_numpy(dtype=float)
    print(f"Datos cargados: {X.shape[0]} registros, 2 variables de entrada.")
    return X, y


def estandarizar(X):
    """
    Estandariza cada columna a media 0 y desviacion 1.
    Se guardan media y desviacion para poder transformar datos nuevos
    con los mismos parametros (no se vuelve a calcular con datos nuevos).
    """
    medias = X.mean(axis=0)
    desviaciones = X.std(axis=0)
    X_est = (X - medias) / desviaciones
    return X_est, medias, desviaciones


def sigmoide(z):
    return 1 / (1 + np.exp(-z))


def calcular_perdida(y_real, y_predicho):
    """Entropia cruzada binaria (log loss). Epsilon evita log(0)."""
    epsilon = 1e-10
    y_predicho = np.clip(y_predicho, epsilon, 1 - epsilon)
    return -np.mean(y_real * np.log(y_predicho) + (1 - y_real) * np.log(1 - y_predicho))


def entrenar_perceptron(X, y, tasa_aprendizaje=0.1, epocas=1000):
    """
    Entrena un perceptron de una sola neurona con descenso de gradiente.
    w: vector de 2 pesos (humedad, presion). b: sesgo (bias).
    """
    n_muestras, n_variables = X.shape
    w = np.zeros(n_variables)
    b = 0.0
    historial_perdida = []

    for epoca in range(epocas):
        # Paso hacia adelante (vectorizado)
        z = np.dot(X, w) + b
        y_predicho = sigmoide(z)

        perdida = calcular_perdida(y, y_predicho)
        historial_perdida.append(perdida)

        # Gradientes (derivadas de la log loss respecto a w y b)
        error = y_predicho - y
        gradiente_w = np.dot(X.T, error) / n_muestras
        gradiente_b = np.mean(error)

        # Actualizacion de parametros
        w -= tasa_aprendizaje * gradiente_w
        b -= tasa_aprendizaje * gradiente_b

        if epoca % 100 == 0:
            print(f"Epoca {epoca:4d}  -  Perdida: {perdida:.4f}")

    return w, b, historial_perdida


def calcular_precision(X, y, w, b):
    z = np.dot(X, w) + b
    y_predicho = sigmoide(z)
    y_clase = (y_predicho >= 0.5).astype(int)
    precision = np.mean(y_clase == y)
    return precision


def graficar_curva_aprendizaje(historial_perdida):
    plt.figure(figsize=(8, 5))
    plt.plot(historial_perdida, color="darkred")
    plt.title("Curva de aprendizaje del perceptron")
    plt.xlabel("Epoca")
    plt.ylabel("Perdida (log loss)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("outputs/curva_aprendizaje.png")
    plt.close()
    print("Guardado: outputs/curva_aprendizaje.png")


def graficar_frontera_decision(X, y, w, b, medias, desviaciones):
    """
    Dibuja la frontera de decision (donde z = 0, es decir sigmoide = 0.5)
    sobre los datos ya estandarizados.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(X[y == 0, 0], X[y == 0, 1], color="orange", label="No llovio", alpha=0.6)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], color="steelblue", label="Llovio", alpha=0.6)

    # Frontera: w0*x0 + w1*x1 + b = 0  ->  x1 = -(w0*x0 + b) / w1
    x0_vals = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    if w[1] != 0:
        x1_vals = -(w[0] * x0_vals + b) / w[1]
        plt.plot(x0_vals, x1_vals, color="black", linestyle="--", label="Frontera de decision")

    plt.title("Frontera de decision del perceptron (datos estandarizados)")
    plt.xlabel("Humedad (estandarizada)")
    plt.ylabel("Presion (estandarizada)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/frontera_decision.png")
    plt.close()
    print("Guardado: outputs/frontera_decision.png")


def main():
    print("=" * 60)
    print(" PERCEPTRON DESDE CERO CON NUMPY - AVANCE 2")
    print("=" * 60)

    X, y = cargar_datos(ARCHIVO_ENTRADA)
    X_est, medias, desviaciones = estandarizar(X)

    print(f"\nMedias (humedad, presion): {medias}")
    print(f"Desviaciones (humedad, presion): {desviaciones}")

    # Perdida inicial (antes de entrenar, con w=0, b=0)
    perdida_inicial = calcular_perdida(y, sigmoide(np.zeros(len(y))))

    print("\nEntrenando...")
    w, b, historial_perdida = entrenar_perceptron(X_est, y, tasa_aprendizaje=0.1, epocas=1000)

    precision = calcular_precision(X_est, y, w, b)

    print("\n" + "=" * 60)
    print("RESULTADOS")
    print("=" * 60)
    print(f"Peso humedad (w1): {w[0]:.4f}")
    print(f"Peso presion (w2): {w[1]:.4f}")
    print(f"Sesgo (b):         {b:.4f}")
    print(f"Perdida inicial:   {perdida_inicial:.4f}")
    print(f"Perdida final:     {historial_perdida[-1]:.4f}")
    print(f"Precision (accuracy): {precision * 100:.2f}%")

    graficar_curva_aprendizaje(historial_perdida)
    graficar_frontera_decision(X_est, y, w, b, medias, desviaciones)

    print(f"""
INTERPRETACION:
- w1 (humedad) = {w[0]:.4f}: signo {'positivo' if w[0] > 0 else 'negativo'},
  es decir que a mayor humedad, {'mayor' if w[0] > 0 else 'menor'} es la
  probabilidad estimada de lluvia. Esto coincide con la correlacion
  positiva observada en el EDA (Seccion C).
- w2 (presion) = {w[1]:.4f}: signo {'positivo' if w[1] > 0 else 'negativo'},
  es decir que a mayor presion, {'mayor' if w[1] > 0 else 'menor'} es la
  probabilidad estimada de lluvia. Esto coincide con la fisica
  atmosferica (menor presion favorece la lluvia).
- La precision de {precision * 100:.1f}% para un modelo lineal tan simple
  (una sola neurona, 2 variables) indica que el problema es parcialmente
  pero no totalmente separable de forma lineal con estas dos variables:
  hay zonas donde dias con y sin lluvia se mezclan (visible en la
  frontera de decision), lo que sugiere que un modelo mas complejo
  (red con mas capas, o mas variables de entrada) podria mejorar el
  resultado en el proyecto final.
""")


if __name__ == "__main__":
    main()