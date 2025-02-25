import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Configurar la página
st.set_page_config(page_title="Análisis de Datos K-Means", layout="wide")

# Cargar el dataset
file_path = "/Users/santiagobarrera/PycharmProjects/PythonProject1/dataset_kmeans_19022025.csv"
df = pd.read_csv(file_path, sep=';')

# Título principal
st.title("📊 Análisis de Datos para K-Means")

# Mostrar dataset inicial
st.subheader("📂 Dataset Original")
st.dataframe(df.head(11))

# Mostrar estadísticas descriptivas
st.subheader("📈 Estadísticas Descriptivas")
st.write(df.describe())

# Función para detectar outliers usando IQR
def detectar_outliers_iqr(df, columna):
    Q1 = df[columna].quantile(0.25)
    Q3 = df[columna].quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    outliers = df[(df[columna] < limite_inferior) | (df[columna] > limite_superior)]
    return outliers

# Detectar valores atípicos
outliers_edad = detectar_outliers_iqr(df, "Edad")
outliers_gasto = detectar_outliers_iqr(df, "Gasto Mensual (USD)")
outliers_compras = detectar_outliers_iqr(df, "Compras Mensuales")

# Mostrar valores atípicos
st.subheader("⚠️ Valores Atípicos Detectados")
outliers_count = {
    "Edad": outliers_edad.shape[0],
    "Gasto Mensual (USD)": outliers_gasto.shape[0],
    "Compras Mensuales": outliers_compras.shape[0],
}
st.write(outliers_count)

# Visualización de outliers
st.subheader("📌 Boxplots para Identificación de Outliers")
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
sns.boxplot(y=df["Edad"], ax=ax[0], color="lightblue")
ax[0].set_title("Edad")

sns.boxplot(y=df["Gasto Mensual (USD)"], ax=ax[1], color="lightgreen")
ax[1].set_title("Gasto Mensual (USD)")

sns.boxplot(y=df["Compras Mensuales"], ax=ax[2], color="lightcoral")
ax[2].set_title("Compras Mensuales")

st.pyplot(fig)

# Calcular correlaciones
st.subheader("🔗 Correlaciones entre Variables")
correlaciones = df[["Edad", "Gasto Mensual (USD)", "Compras Mensuales"]].corr()
st.write(correlaciones)

# Mapa de calor de correlaciones
st.subheader("🌡️ Heatmap de Correlaciones")
fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(correlaciones, annot=True, cmap="coolwarm", linewidths=0.5, ax=ax)
st.pyplot(fig)

# Limpiar datos eliminando duplicados
df_cleaned = df.drop_duplicates()

# Normalizar los datos para K-Means
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_cleaned[["Edad", "Gasto Mensual (USD)", "Compras Mensuales"]])
df_scaled = pd.DataFrame(X_scaled, columns=["Edad", "Gasto Mensual (USD)", "Compras Mensuales"])

# Mostrar dataset limpio y normalizado
st.subheader("✅ Dataset Limpio y Normalizado")
st.dataframe(df_scaled.head())

# Gráficos de distribución de variables normalizadas
st.subheader("📊 Distribución de Variables Normalizadas")
fig, ax = plt.subplots(1, 3, figsize=(15, 5))

sns.histplot(df_scaled["Edad"], bins=30, kde=True, ax=ax[0], color="lightblue")
ax[0].set_title("Distribución de Edad (Normalizada)")

sns.histplot(df_scaled["Gasto Mensual (USD)"], bins=30, kde=True, ax=ax[1], color="lightgreen")
ax[1].set_title("Distribución de Gasto Mensual (Normalizada)")

sns.histplot(df_scaled["Compras Mensuales"], bins=30, kde=True, ax=ax[2], color="lightcoral")
ax[2].set_title("Distribución de Compras Mensuales (Normalizada)")

st.pyplot(fig)