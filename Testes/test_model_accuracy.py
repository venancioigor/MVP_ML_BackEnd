import pytest
import pickle
from sklearn.metrics import accuracy_score

# Carregar o modelo
with open('../model.pkl', 'rb') as f:
    model = pickle.load(f)


def test_should_predict_cure_when_correct_ingredients_provided():
    
    test_input = [[2.4, 18.7, 18.4, 27.9, 7.9, 9.6, 18.3, 13.2, 2.5, 26.0, 10.5, 26.2, 12.5]]
    expected_output = [1]  
    prediction = model.predict(test_input)
    assert list(prediction) == expected_output

