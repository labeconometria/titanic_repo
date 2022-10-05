import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

data = pd.read_csv('clean_titanic.csv')
modelo = pickle.load(open('modelo.pickle', 'rb'))

st.title('¿Habrías sobrevivido al Titanic?')

st.header('Exploración inicial')

st.header('Visualización')

st.header('Verifica si habrías sobrevivido')

col1, col2 = st.columns(2)

with col1:
  st.header('Características')
  sexo = st.selectbox('Genero', ('M', 'F'))
  if sexo == 'M':
    sexo = 1
  else:
    sexo = 0
  par_hijos = st.slider('Número entre padres e hijos', 0, 10)
  hermanos_esposos = st.slider('Número entre hermanos(as) y esposo(a)', 0, 10)

with col2:
  st.header('Boleto')
  edad = st.slider('Edad', 0, 99)
  clase = st.selectbox('Clase', (1, 2, 3))
  fare = st.slider('Disposición a pagar por el boleto', 0, 1000)
 
if st.button('Predecir'):
  
  pred = modelo.predict_proba(np.array([[clase, edad, hermanos_esposos, par_hijos, fare, sexo]]))
  st.text(f'Habrías sobrevivido con una probabilidad de {round(pred, 3) * 100}%.'
else:
  st.text('Seleccione entre las opciones y oprima Predecir')

  
