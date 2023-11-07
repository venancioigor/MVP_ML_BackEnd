import pytest
import pickle
from sklearn.metrics import accuracy_score
import pandas as pd

# Carregar o modelo
with open('../model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('../scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

def test_should_predict_cure_when_correct_ingredients_provided():
    
    data = {
    'Pena de Fenix': [22.1],
    'Chifre de Unicornio': [10.8],
    'Sangue de Dragão': [16.4],
    'Lágrimas de Sereia': [10.5],
    'Pó de Fada': [22.0],
    'Dedos de Duende': [23.4],
    'Poção da Bruxa': [2.6],
    'Garra de Grifo': [18.2],
    'Cabelo de Troll': [23.8],
    'Tinta de Kraken': [11.3],
    'Chifre de Minotauro': [5.5],
    'Escama de Basilisco': [16.8],
    'Presa de Quimera': [16.2]
}
    atributos = [
                    'Pena de Fenix', 'Chifre de Unicornio', 'Sangue de Dragão','Lágrimas de Sereia',
                    'Pó de Fada','Dedos de Duende','Poção da Bruxa', 'Poção da Bruxa', 'Garra de Grifo',
                    'Cabelo de Troll',  'Tinta de Kraken','Chifre de Minotauro','Escama de Basilisco',
                    'Presa de Quimera'
                    ]
    
    entrada = pd.DataFrame(data, columns=atributos)

    array_entrada = entrada.values
    X_entrada = array_entrada[:,0:13].astype(float)

# Padronização nos dados de entrada usando o scaler utilizado em X
    rescaledEntradaX = scaler.transform(X_entrada)
    prediction = model.predict(rescaledEntradaX)
    
    expected_output = [0]  
    assert list(prediction) == expected_output

