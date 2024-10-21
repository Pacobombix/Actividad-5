
import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from funpymodeling.exploratory import freq_tbl 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt

categorical_columns = ['host_name', 'neighbourhood', 'property_type']
limited_columns = ["beds", "accommodates", "bathrooms", "bedrooms"]

@st.cache_resource
def load_data():
    df1 = pd.read_csv("CLEANED_MILAN_FINAL.csv")
    
    for col in categorical_columns:
        if col not in df1.columns:
            st.error(f"La columna '{col}' no existe en el dataset.")
            return None, None, None
    
    for col in categorical_columns:
        if not pd.api.types.is_categorical_dtype(df1[col]):
            df1[col] = df1[col].astype('category')
    
    freq_tables = {}
    for col in categorical_columns:
        table = freq_tbl(df1[col])
        if 'frequency' not in table.columns:
            st.error(f"La columna 'frequency' no existe en la tabla de frecuencias para {col}.")
            return None, None, None
        freq_tables[col] = table[table['frequency'] > 1].set_index(col)
    
    numeric_cols = df1.select_dtypes(include=[np.number]).columns.tolist()
    
    return df1, freq_tables, numeric_cols

df1, freq_tables, numeric_cols = load_data()

if freq_tables is None:
    st.stop()

st.sidebar.title("DASHBOARD")
st.sidebar.header("Sidebar")
st.sidebar.subheader("Panel de selección")

Frames = st.selectbox(label="Frames", options=["Frame 1: Extracción de Características", "Frame 2: Análisis Avanzado", "Frame 3: Análisis de Regresiones"])

if Frames == "Frame 1: Extracción de Características":
    st.title("Ciudad de Milán")
    st.header("Extracción de Características")
    
    show_full_data = st.sidebar.checkbox(label="Mostrar y filtrar Dataset Completo")
    if show_full_data:
        score_rate_filter = st.sidebar.slider("Filtrar por calificación (score_rate)", min_value=int(df1['score_rate'].min()), max_value=int(df1['score_rate'].max()))
        filtered_data = df1[df1['score_rate'] >= score_rate_filter]
        st.write(filtered_data)
    
    st.write("Tabla de Frecuencia - property_type")
    st.write(freq_tables['property_type'].head(15))
    
    st.write("Tabla de Frecuencia - host_name")
    st.write(freq_tables['host_name'].head(15))
    
    st.write("Tabla de Frecuencia - neighbourhood")
    st.write(freq_tables['neighbourhood'].head(15))

elif Frames == "Frame 2: Análisis Avanzado":
    st.title("Análisis Avanzado")
    
    numeric_df = df1.select_dtypes(include=[np.number])
    fig, ax = plt.subplots()
    sns.heatmap(numeric_df.corr(), ax=ax, cmap='coolwarm', annot=True)
    st.pyplot(fig)
    
    selected_col = st.sidebar.selectbox(label="Seleccionar columna para Pieplot", options=limited_columns)
    fig = px.pie(df1, names=selected_col, title="Pieplot")
    st.plotly_chart(fig)
    
    selected_boxplot_col = st.sidebar.selectbox(label="Seleccionar columna para Boxplot", options=numeric_df.columns)
    fig = px.box(df1, y=selected_boxplot_col, title="Boxplot")
    st.plotly_chart(fig)

elif Frames == "Frame 3: Análisis de Regresiones":
    st.title("Análisis de Regresiones")
    
    regression_type = st.sidebar.selectbox(label="Tipo de Regresión", options=["Regresión Lineal Simple", "Regresión Lineal Múltiple", "Regresión No Lineal", "Regresión Logística"])
    
    if regression_type == "Regresión Lineal Simple":
        x_var = st.sidebar.selectbox(label="Variable Independiente (X)", options=numeric_cols)
        y_var = st.sidebar.selectbox(label="Variable Dependiente (Y)", options=numeric_cols)
        if x_var and y_var:
            X = df1[[x_var]].values
            y = df1[y_var].values
            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)
            
            fig = px.scatter(df1, x=x_var, y=y_var, title="Regresión Lineal Simple")
            fig.add_scatter(x=df1[x_var], y=y_pred, mode='lines', name='Regresión Lineal')
            st.plotly_chart(fig)
    
    elif regression_type == "Regresión Lineal Múltiple":
        x_vars = st.sidebar.multiselect(label="Variables Independientes (X)", options=numeric_cols)
        y_var = st.sidebar.selectbox(label="Variable Dependiente (Y)", options=numeric_cols)
        if x_vars and y_var:
            X = df1[x_vars].values
            y = df1[y_var].values
            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)
            
            fig = px.scatter(df1, x=x_vars[0], y=y_var, title="Regresión Lineal Múltiple")
            fig.add_scatter(x=df1[x_vars[0]], y=y_pred, mode='lines', name='Regresión Lineal')
            st.plotly_chart(fig)
    
    elif regression_type == "Regresión No Lineal":
        x_var = st.sidebar.selectbox(label="Variable Independiente (X)", options=numeric_cols)
        y_var = st.sidebar.selectbox(label="Variable Dependiente (Y)", options=numeric_cols)
        degree = st.sidebar.slider("Grado del Polinomio", 2, 5)
        if x_var and y_var:
            poly = PolynomialFeatures(degree=degree)
            X_poly = poly.fit_transform(df1[[x_var]])
            model = LinearRegression()
            model.fit(X_poly, df1[y_var])
            y_pred = model.predict(X_poly)
            
            fig = px.scatter(df1, x=x_var, y=y_var, title="Regresión No Lineal")
            fig.add_scatter(x=df1[x_var], y=y_pred, mode='lines', name='Regresión No Lineal')
            st.plotly_chart(fig)
    
    elif regression_type == "Regresión Logística":
        x_vars = st.sidebar.multiselect(label="Variables Independientes (X)", options=numeric_cols)
        y_var = st.sidebar.selectbox(label="Variable Dependiente (Y)", options=numeric_cols)
        if x_vars and y_var:
            X = df1[x_vars].values
            y = df1[y_var].values
            model = LogisticRegression()
            model.fit(X, y)
            y_pred = model.predict(X)
            
            fig = px.scatter(df1, x=x_vars[0], y=y_var, title="Regresión Logística")
            fig.add_scatter(x=df1[x_vars[0]], y=y_pred, mode='lines', name='Regresión Logística')
            st.plotly_chart(fig)
