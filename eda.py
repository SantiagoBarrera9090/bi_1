import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate

# Cargar el dataset
file_path = "dataset_kmeans_19022025.csv"
df = pd.read_csv(file_path, sep=';')

# Información del dataset
df_info = df.info()
df_head = df.head()
df_description = df.describe()
df_nulls = df.isnull().sum()

# Mostrar los primeros datos
print("\n🔍 **Primeras 11 Filas del Dataset:**")
print(tabulate(df.head(11), headers='keys', tablefmt='fancy_grid'))

print("\n📊 **Estadísticas Descriptivas:**")
print(tabulate(df_description, headers='keys', tablefmt='fancy_grid'))

# Función para detectar outliers usando IQR
def detectar_outliers_iqr(df, columna):
    Q1 = df[columna].quantile(0.25)
    Q3 = df[columna].quantile(0.75)
    IQR = Q3 - Q1

    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR

    outliers = df[(df[columna] < limite_inferior) | (df[columna] > limite_superior)]
    return outliers

# Identificar valores atípicos
outliers_edad = detectar_outliers_iqr(df, "Edad")
outliers_gasto = detectar_outliers_iqr(df, "Gasto Mensual (USD)")
outliers_compras = detectar_outliers_iqr(df, "Compras Mensuales")

outliers_count = {
    "Valores Atípicos en Edad": outliers_edad.shape[0],
    "Valores Atípicos en Gasto Mensual": outliers_gasto.shape[0],
    "Valores Atípicos en Compras Mensuales": outliers_compras.shape[0],
}

print("\n⚠️ **Valores Atípicos Detectados:**")
print(tabulate(outliers_count.items(), headers=["Categoría", "Cantidad"], tablefmt="fancy_grid"))

# Calcular correlaciones
correlacion_edad_compras = df["Edad"].corr(df["Compras Mensuales"])
correlacion_gasto_compras = df["Gasto Mensual (USD)"].corr(df["Compras Mensuales"])
correlacion_edad_gasto = df["Edad"].corr(df["Gasto Mensual (USD)"])

print("\n📈 **Coeficientes de Correlación:**")
correlaciones = [
    ["Edad vs Compras Mensuales", correlacion_edad_compras],
    ["Gasto Mensual vs Compras Mensuales", correlacion_gasto_compras],
    ["Edad vs Gasto Mensual", correlacion_edad_gasto],
]
print(tabulate(correlaciones, headers=["Relación", "Coeficiente"], tablefmt="fancy_grid"))

# Limpiar datos eliminando duplicados
df_cleaned = df.drop_duplicates()

# Normalizar los datos para K-Means
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_cleaned[["Edad", "Gasto Mensual (USD)", "Compras Mensuales"]])

# Convertir a DataFrame después de la normalización
df_scaled = pd.DataFrame(X_scaled, columns=["Edad", "Gasto Mensual (USD)", "Compras Mensuales"])

# Mostrar el nuevo dataset limpio y normalizado de forma visual
print("\n✅ **Dataset Limpio y Normalizado:**")
print(tabulate(df_scaled.head(), headers='keys', tablefmt='fancy_grid'))