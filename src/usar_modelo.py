"""
usar_modelo.py
Carga el modelo guardado (src/modelo_final.joblib) y predice dias nuevos.
NO entrena nada. El Pipeline ya incluye el escalado.
Ejecutar desde la raiz del proyecto (/app):  python src/usar_modelo.py
"""
import joblib
import pandas as pd

modelo = joblib.load("src/modelo_final.joblib")

# Dias nuevos (mismos nombres de columna que en el entrenamiento).
# Localidad en formato One-Hot: 1 en la que corresponde, 0 en las demas.
dias_nuevos = pd.DataFrame({
    "localidad": ["El Billar", "Zaragoza", "Ansermanuevo"],
    "humedad": [88.0, 55.0, 74.0],
    "presion": [1009.5, 1016.0, 1012.5],
    "temperatura": [18.5, 24.0, 21.0],
})
for loc in ["Ansermanuevo", "El Billar", "Zaragoza"]:
    dias_nuevos[f"loc_{loc}"] = (dias_nuevos["localidad"] == loc).astype(int)

columnas = ["humedad", "presion", "temperatura",
            "loc_Ansermanuevo", "loc_El Billar", "loc_Zaragoza"]
probabilidad = modelo.predict_proba(dias_nuevos[columnas])[:, 1]
prediccion = modelo.predict(dias_nuevos[columnas])

dias_nuevos["prob_lluvia_%"] = (probabilidad * 100).round(1)
dias_nuevos["resultado"] = ["Lloverá" if p == 1 else "No lloverá" for p in prediccion]

print("Predicciones con el modelo guardado:")
print(dias_nuevos[["localidad", "humedad", "presion", "temperatura",
                   "prob_lluvia_%", "resultado"]].to_string(index=False))
