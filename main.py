import streamlit as st
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from xgboost import XGBRegressor



model = joblib.load("xgboost_log.pkl")
encoder = joblib.load("encoder.pkl")

st.title("Estimador de Precio de Departamentos en Buenos Aires.")
st.markdown("Este estimador fue entrenado con algoritmos de aprendizaje automático sobre un conjunto de datos de propiedades en CABA (2016).  El modelo final utilizado es **XGBoost**.")
st.markdown("Para más información al respecto consulte este [enlace](https://colab.research.google.com/drive/1mSAg3AsKneUCFINGvCq4FjNmQmywJV6K?usp=sharing).")

st.divider()
# Entradas


barrio = st.selectbox("Barrio",
                      ['PALERMO', 'VILLA CRESPO', 'SAAVEDRA', 'BOEDO',
       'VILLA URQUIZA', 'COLEGIALES', 'CABALLITO', 'ALMAGRO', 'BALVANERA',
       'CONSTITUCION', 'RETIRO', 'MONTE CASTRO', 'FLORES',
       'SAN CRISTOBAL', 'BELGRANO', 'PARQUE PATRICIOS', 'RECOLETA',
       'VILLA SANTA RITA', 'VILLA DEVOTO', 'CHACARITA', 'SAN NICOLAS',
       'NUÑEZ', 'PARQUE CHAS', 'VILLA REAL', 'VILLA LURO',
       'PUERTO MADERO', 'BOCA', 'MONSERRAT', 'PARQUE CHACABUCO',
       'VILLA ORTUZAR', 'VILLA PUEYRREDON', 'SAN TELMO', 'COGHLAN',
       'VILLA LUGANO', 'FLORESTA', 'BARRACAS', 'VILLA GRAL. MITRE',
       'MATADEROS', 'LINIERS', 'PARQUE AVELLANEDA', 'VILLA DEL PARQUE',
       'VERSALLES', 'AGRONOMIA', 'VELEZ SARSFIELD', 'NUEVA POMPEYA',
       'VILLA SOLDATI', 'PATERNAL'])
m2cub = st.number_input("Metros cuadrados cubiertos", min_value=10.0, max_value=500.0, value=80.0, step=1.0)
m2desc= st.number_input("Metros cuadrados descubiertos", min_value=10.0, max_value=500.0, value=60.0, step=1.0)
ambientes = st.slider("Ambientes", min_value=1, max_value=10, value=2, step=1)
antiguedad = st.slider("Antiguedad", min_value=0, max_value=100, value=30, step=1)
baños = st.slider("Baños", min_value=1, max_value=10, value=1, step=1)

with st.expander("¿Cómo obtener la latitud y longitud de una dirección?"):
    st.markdown("""
Para encontrar las coordenadas de tu departamento:

1. Abrí [Google Maps](https://www.google.com/maps).
2. Escribe la dirección del departamento (por ejemplo, *Av. Santa Fe 1200, CABA*).
3. Hace click derecho sobre el punto en el mapa.
4. Aparecerá la información con la **latitud y longitud** (por ejemplo: `-34.5931, -58.4108`).
5. Copiá esos números y pegálos en los campos correspondientes.

💡 *Recordá que la latitud suele empezar con -34 y la longitud con -58 en CABA.*
""")

latitud= st.number_input("Latitud", min_value=-200.0, max_value=200.0, value=-34.0)
longitud = st.number_input("Longitud", min_value=-200.0, max_value=200.0, value=-58.0)





input_df = pd.DataFrame([{
    "M2CUB": m2cub,
    "AMBIENTES": ambientes,
    "ANTIGUEDAD": antiguedad,
    "BAÑOS": baños,
    "BARRIO": barrio,
    "LATITUD": latitud,
    "LONGITUD": longitud,
    "M2DESC": m2desc     
}])

input_df["BARRIO"] = encoder.transform(input_df[["BARRIO"]]).ravel()


if st.button("Predecir precio"):
       
       y_pred = model.predict(input_df)[0]
       st.success(f"El precio del departamento es de: ${np.expm1(y_pred):,.2f}")
       
       import streamlit as st

