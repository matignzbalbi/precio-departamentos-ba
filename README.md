# 🏙️ Estimador de Precios de Departamentos en Buenos Aires / Buenos Aires Apartment Price Estimator

[🌐 Acceder a la aplicación / Access the App](https://precio-departamentos-ba.streamlit.app/)

---

## 📌 Español


Aplicación interactiva desarrollada con **Streamlit** que predice el precio estimado de un departamento en la Ciudad de Buenos Aires, utilizando un modelo de machine learning entrenado con **XGBoost**.

🔍 El modelo se entrena con un dataset real del mercado inmobiliario porteño, incorporando variables como:
- Superficie total y cubierta
- Antigüedad
- Ambientes y baños
- Barrio y coordenadas geográficas

📊 El pipeline incluye:
- Limpieza y transformación de datos
- Análisis exploratorio
- Codificación de variables categóricas con `OrdinalEncoder`
- Comparación de modelos y selección del mejor (XGBoost)


