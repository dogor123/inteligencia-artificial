"""
modelo_proyecto.py
Compara modelos de Scikit-Learn para predecir si llovera (llovio = 1) y
guarda el mejor con joblib. Sigue el esquema de la Clase 12.

Entradas: humedad, presion, temperatura y localidad (One-Hot).
NO se usa precipitacion_mm como entrada: llovio se define a partir de
ella (llovio = 1 si precipitacion_mm > 0), asi que usarla seria fuga de
informacion (el modelo "haria trampa").

Genera: outputs/matriz_confusion.png, outputs/comparacion_modelos.csv
        src/modelo_final.joblib
Ejecutar desde la raiz del proyecto (/app):  python src/modelo_proyecto.py
"""
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

ARCHIVO_DATOS = "data/proyecto_limpio.csv"
ARCHIVO_MODELO = "src/modelo_final.joblib"
COLUMNAS_ENTRADA = ["humedad", "presion", "temperatura",
                    "loc_Ansermanuevo", "loc_El Billar", "loc_Zaragoza"]

# 1. Cargar datos
df = pd.read_csv(ARCHIVO_DATOS)
for c in ["loc_Ansermanuevo", "loc_El Billar", "loc_Zaragoza"]:
    df[c] = df[c].astype(int)          # True/False -> 1/0
X = df[COLUMNAS_ENTRADA]
y = df["llovio"]

# 2. Separar: 80 % entrenamiento, 20 % prueba (misma proporcion de 0 y 1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Registros para entrenar: {len(X_train)} | para probar: {len(X_test)}")

# 3. Modelos (cada uno escala DENTRO del Pipeline, solo con datos de entrenamiento)
modelos = {
    "Regresión logística": Pipeline([
        ("escalado", StandardScaler()),
        ("modelo", LogisticRegression()),
    ]),
    "KNN (5 vecinos)": Pipeline([
        ("escalado", StandardScaler()),
        ("modelo", KNeighborsClassifier(n_neighbors=5)),
    ]),
    "Árbol de decisión": Pipeline([
        ("escalado", StandardScaler()),
        ("modelo", DecisionTreeClassifier(max_depth=3, random_state=42)),
    ]),
    "Red neuronal (MLP)": Pipeline([
        ("escalado", StandardScaler()),
        ("modelo", MLPClassifier(hidden_layer_sizes=(8,), max_iter=2000,
                                 random_state=42)),
    ]),
}

# 4. Entrenar con entrenamiento, medir con prueba + validacion cruzada (F1)
resultados = []
for nombre, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    resultados.append({
        "Modelo": nombre,
        "Exactitud": accuracy_score(y_test, y_pred),
        "Precisión": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "Exactitud entrenamiento": modelo.score(X_train, y_train),
        "F1 val. cruzada": cross_val_score(modelo, X, y, cv=5,
                                           scoring="f1").mean(),
    })

tabla = pd.DataFrame(resultados)
print("\nComparación de modelos (datos de prueba):")
print(tabla.round(3).to_string(index=False))
tabla.round(3).to_csv("outputs/comparacion_modelos.csv", index=False)

# 5. Elegir por F1 de validacion cruzada (mas estable que una sola prueba)
nombre_mejor = tabla.loc[tabla["F1 val. cruzada"].idxmax(), "Modelo"]
mejor_modelo = modelos[nombre_mejor]
print(f"\nMejor modelo según F1 de validación cruzada: {nombre_mejor}")

if nombre_mejor == "Regresión logística":
    reg = mejor_modelo.named_steps["modelo"]
    print("Pesos aprendidos (variables estandarizadas):")
    for nom, peso in zip(COLUMNAS_ENTRADA, reg.coef_[0]):
        print(f"  {nom}: {peso:+.3f}")
    print(f"  sesgo (b): {reg.intercept_[0]:+.3f}")

# 6. Matriz de confusion del mejor modelo
matriz = confusion_matrix(y_test, mejor_modelo.predict(X_test))
print("\nMatriz de confusión (filas = real, columnas = predicción):")
print(matriz)
plt.figure(figsize=(6, 5))
sns.heatmap(matriz, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Predijo 0", "Predijo 1"],
            yticklabels=["Real 0", "Real 1"])
plt.title(f"Matriz de confusión - {nombre_mejor}")
plt.tight_layout()
plt.savefig("outputs/matriz_confusion.png", dpi=150)
plt.close()
print("Gráfico guardado: outputs/matriz_confusion.png")

# 7. Guardar el mejor modelo (incluye el escalado)
joblib.dump(mejor_modelo, ARCHIVO_MODELO)
print(f"Modelo guardado: {ARCHIVO_MODELO}")
