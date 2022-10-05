import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import sklearn

data = pd.read_csv('clean_titanic.csv')
modelo = pickle.load(open('modelo.pickle', 'rb'))

st.title('¿Habrías sobrevivido al Titanic?')

st.header('Exploración inicial')

st.dataframe(data.describe)
st.dataframe(data.head())

st.header('Visualización')
       
fig, ax = plt.subplots(1, 4, sharey=True, figsize=(16, 4))
ax[0].set_ylabel('%')

for idx, col in enumerate(['pclass','sibsp','parch','sex_male']):
  data[col].value_counts(normalize=True).plot(kind='bar', ax=ax[idx], title=col)  
st.pyplot(fig)

fig, ax = plt.subplots(1, 4, sharey=True, figsize=(16, 4))

for idx, col in enumerate(['pclass','sibsp','parch','sex_male']):
  pd.crosstab(data[col], data['survived']).plot(kind='bar', ax=ax[idx], title=col) 
        
st.pyplot(fig)
        
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
  clase = st.selectbox('Clase', (1, 2, 3))
  edad = st.slider('Edad', 0, 99)
  fare = st.slider('Disposición a pagar por el boleto', 0, 500)
 
if st.button('Predecir'):
  
  pred = modelo.predict_proba(np.array([[clase, edad, hermanos_esposos, par_hijos, fare, sexo]]))
  st.text(f'Habrías sobrevivido con una probabilidad de {round(pred[0][0], 3) * 100}%.')
else:
  st.text('Seleccione entre las opciones y oprima Predecir')

  
